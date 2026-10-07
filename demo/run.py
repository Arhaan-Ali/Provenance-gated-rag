"""Local RAG with provenance gating: every document carries the trust tier of its origin."""

import argparse
import shutil
import sys
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)


from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from pgrag.core.registry import (
    discover_origins,
    load_latest_decisions,
    latest_decisions_path,
    load_rules,
    save_decisions,
)

from pgrag.cli import run_review_gate

from pgrag.adapters.llamaindex import stamp_documents, TIER_KEY, ORIGIN_KEY

DATA_DIR = Path("data")
PERSIST_DIR = Path("storage")
RESULTS_DIR = Path("results")
TRUST_FILE = Path("trust.yaml")
DECISIONS_DIR = Path("decisions")
OLLAMA_URL = "http://localhost:11434"



Settings.llm = Ollama(
    model="qwen3:30b-a3b",
    base_url=OLLAMA_URL,
    request_timeout=600.0,
    context_window=32768,
    thinking=False,
    temperature=0,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_URL,
)

Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)




def build_index() -> VectorStoreIndex:
    """Review every origin, then embed only what was approved."""
    if not DATA_DIR.is_dir():
        sys.exit(f"Put your documents in ./{DATA_DIR}/<origin-folder>/ first.")

    rules = load_rules(TRUST_FILE)
    origins = discover_origins(DATA_DIR, rules)
    if not origins:
        sys.exit(f"No origin folders found in ./{DATA_DIR}/. Each subfolder is one origin.")

    decisions = run_review_gate(origins)
    decisions_path = save_decisions(decisions, DECISIONS_DIR)
    print(f"\nDecisions saved to {decisions_path}")

    approved = load_latest_decisions(DECISIONS_DIR)
    documents = SimpleDirectoryReader(str(DATA_DIR), recursive=True).load_data()
    kept, dropped, provenance = stamp_documents(documents, DATA_DIR, approved)

    if dropped:
        print(f"Dropped {len(dropped)} document(s) from excluded origins.")
    if not kept:
        sys.exit("Nothing left to index - every origin was excluded.")

    by_tier = {}
    for doc in kept:
        by_tier[doc.metadata[TIER_KEY]] = by_tier.get(doc.metadata[TIER_KEY], 0) + 1
    print("Indexing " + ", ".join(f"{n} tier {t}" for t, n in sorted(by_tier.items())))

    index = VectorStoreIndex.from_documents(kept, show_progress=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    (PERSIST_DIR / ".decisions_version").write_text(decisions_path.name, encoding="utf-8")
    return index


def get_index(force_reindex: bool = False) -> VectorStoreIndex:
    """Load the cached index, but only if it was built from the current decisions.

    The old version returned the cache whenever the folder existed, which meant
    adding a document or changing a tier silently had no effect. The marker file
    ties the index to the decision set it came from.
    """
    if force_reindex and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    marker = PERSIST_DIR / ".decisions_version"
    if PERSIST_DIR.is_dir() and marker.exists():
        latest = latest_decisions_path(DECISIONS_DIR)
        current = latest.name if latest else None
        if current and marker.read_text(encoding="utf-8").strip() == current:
            print(f"Loading cached index (built from {current}).")
            return load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
            )
        print("Decisions changed since the index was built - re-running the review gate.")
        shutil.rmtree(PERSIST_DIR)

    return build_index()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def save_last(last: dict, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"MODEL:    {Settings.llm.model}\n")
        f.write(f"QUESTION: {last['q']}\n\n")
        f.write(f"{last['a']}\n\n")
        f.write("--- Sources ---\n")
        for score, tier, origin, fname, text in last["s"]:
            f.write(f"\n[{score:.3f}] [tier {tier}] {origin}/{fname}\n{text[:500].strip()}\n")
    print(f"Saved to {path}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local RAG with provenance gating.")
    ap.add_argument("--reindex", action="store_true",
                    help="discard the cached index and re-run the review gate")
    args = ap.parse_args()

    index = get_index(force_reindex=args.reindex)
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        streaming=False,
    )

    last = None
    print("\nAsk a question. Type  save <name>  to keep the last answer.")
    print("Blank line or Ctrl-C to quit.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break

        if line.startswith("save "):
            name = line[5:].strip()
            if last is None:
                print("Nothing to save yet.\n")
            elif not name:
                print("Give it a name: save my_test_1\n")
            else:
                save_last(last, name)
            continue

        response = query_engine.query(line)
        print(f"\n{response}\n")

        srcs = []
        print("--- Sources ---")
        for node in response.source_nodes:
            meta = node.metadata
            tier = meta.get(TIER_KEY, "?")
            origin = meta.get(ORIGIN_KEY, "?")
            fname = meta.get("file_name", "?")
            score = float(node.score or 0)
            print(f"[{score:.3f}] [tier {tier}] {origin}/{fname}: {node.text[:120].strip()}...")
            srcs.append((score, tier, origin, fname, node.text))
        print()

        last = {"q": line, "a": str(response), "s": srcs}


if __name__ == "__main__":
    main()