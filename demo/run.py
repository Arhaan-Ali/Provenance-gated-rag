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
    origins_needing_review,
)

from datetime import datetime, timezone

from pgrag.core.sanitizer import sanitize_documents

from pgrag.cli import run_review_gate

from pgrag.adapters.llamaindex import stamp_documents, to_nodes, TIER_KEY, ORIGIN_KEY

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

    previous = load_latest_decisions(DECISIONS_DIR)
    decisions = run_review_gate(origins, previous)
    decisions_path = save_decisions(decisions, DECISIONS_DIR)
    print(f"\nDecisions saved to {decisions_path}")

    approved = load_latest_decisions(DECISIONS_DIR)
    documents = SimpleDirectoryReader(str(DATA_DIR), recursive=True).load_data()

    # Before stamping, and before chunking: a document that instructs a model
    # never reaches the index at all. Has to happen here, on whole documents -
    # see pgrag/core/sanitizer.py for why doing it per chunk is bypassable.
    documents, instructing = sanitize_documents(documents)
    if instructing:
        print(f"Sanitizer dropped {len(instructing)} document(s) that instructed the model:")
        for doc in instructing:
            print(f"  - {doc.metadata.get('file_path', '?')}")

    kept, dropped, provenance = stamp_documents(documents, DATA_DIR, approved)

    if dropped:
        print(f"Dropped {len(dropped)} document(s) from excluded origins.")
    if not kept:
        sys.exit("Nothing left to index - every document was excluded or sanitized away.")

    by_tier = {}
    for doc in kept:
        by_tier[doc.metadata[TIER_KEY]] = by_tier.get(doc.metadata[TIER_KEY], 0) + 1
    print("Indexing " + ", ".join(f"{n} tier {t}" for t, n in sorted(by_tier.items())))

    nodes = to_nodes(kept, provenance, [Settings.node_parser])
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    (PERSIST_DIR / ".decisions_version").write_text(decisions_path.name, encoding="utf-8")
    return index


def get_index(force_reindex: bool = False) -> VectorStoreIndex:
    """Load the cached index only if it matches the current decisions and no review is due."""
    if force_reindex and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    marker = PERSIST_DIR / ".decisions_version"
    if PERSIST_DIR.is_dir() and marker.exists():
        latest = latest_decisions_path(DECISIONS_DIR)
        current = latest.name if latest else None
        if current and marker.read_text(encoding="utf-8").strip() == current:
            origins = discover_origins(DATA_DIR, load_rules(TRUST_FILE))
            pending = origins_needing_review(
                origins, load_latest_decisions(DECISIONS_DIR), datetime.now(timezone.utc))
            if not pending:
                print(f"Loading cached index (built from {current}).")
                return load_index_from_storage(
                    StorageContext.from_defaults(persist_dir=str(PERSIST_DIR)))
            print(f"Review needed again for {len(pending)} origin(s) - re-running the gate.")
        else:
            print("Decisions changed since the index was built - re-running the review gate.")
        shutil.rmtree(PERSIST_DIR)

    return build_index()

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