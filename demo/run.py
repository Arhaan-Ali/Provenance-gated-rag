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
from llama_index.readers.file import PDFReader

from pgrag.adapters.llamaindex import (
    ORIGIN_KEY,
    SANITIZER_FLAG_KEY,
    TIER_KEY,
    InstructionSanitizerPostprocessor,
    run_gated_turn,
    stamp_documents,
    to_nodes,
)
from pgrag.cli import run_review_gate, show_home
from pgrag.core.freshness import stale_reason, write_manifest
from pgrag.core.registry import (
    DecisionsError,
    discover_origins,
    latest_decisions_path,
    load_latest_decisions,
    load_rules,
    save_decisions,
)
from pgrag.core.capability_gate import TOOL_REQUIRED_TIER
from pgrag.core.sanitizer import DEFAULT_THRESHOLD, sanitize_documents

from demo.tools import TOOLS

# Anchored to the project, not to the folder the command was typed in, so
# `python -m demo.run` finds data/ and decisions/ from anywhere.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PERSIST_DIR = ROOT / "storage"
RESULTS_DIR = ROOT / "results"
TRUST_FILE = ROOT / "trust.yaml"
DECISIONS_DIR = ROOT / "decisions"
OLLAMA_URL = "http://localhost:11434"


Settings.llm = Ollama(
    model="mistral",
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


def make_reader() -> SimpleDirectoryReader:
    """Built on demand, so importing this module never needs data/ to exist."""
    return SimpleDirectoryReader(
        str(DATA_DIR), recursive=True,
        required_exts=[".txt", ".md", ".pdf"],
        file_extractor={".pdf": PDFReader(return_full_document=True)},   # whole document, not per page
    )


def screening_label(classifier) -> str:
    """What the documents are screened with, recorded in the index manifest."""
    if classifier is None:
        return "patterns"
    return f"patterns+{classifier.describe} (threshold {DEFAULT_THRESHOLD})"


def build_index(classifier=None) -> VectorStoreIndex:
    """Review every origin, then embed only what was approved."""
    if not DATA_DIR.is_dir():
        sys.exit(f"Put your documents in {DATA_DIR}/<origin-folder>/ first.")

    rules = load_rules(TRUST_FILE)
    origins = discover_origins(DATA_DIR, rules)
    if not origins:
        sys.exit(f"No origin folders found in {DATA_DIR}/. Each subfolder is one origin.")

    previous = load_latest_decisions(DECISIONS_DIR)
    decisions = run_review_gate(origins, previous)
    decisions_path = save_decisions(decisions, DECISIONS_DIR)
    print(f"\nDecisions saved to {decisions_path}")

    approved = load_latest_decisions(DECISIONS_DIR)
    documents = make_reader().load_data()

    # Before stamping, and before chunking: a document that instructs a model
    # never reaches the index at all. Has to happen here, on whole documents -
    # see pgrag/core/sanitizer.py for why doing it per chunk is bypassable.
    # Both stages run here, not just at query time: the query-time screen only
    # sees the chunks a question retrieved, so a directive in any other chunk
    # of a pattern-clean document would never be scored at all.
    documents, instructing = sanitize_documents(documents, classifier=classifier)
    if instructing:
        print(f"Sanitizer dropped {len(instructing)} document(s) that instructed the model:")
        for doc in instructing:
            print(f"  - {doc.metadata.get('file_path', '?')}")

    kept, dropped, provenance = stamp_documents(documents, DATA_DIR, approved)

    if dropped:
        print(f"Dropped {len(dropped)} document(s): excluded origin, or not inside an origin folder.")
    if not kept:
        sys.exit("Nothing left to index - every document was excluded or sanitized away.")

    by_tier = {}
    for doc in kept:
        by_tier[doc.metadata[TIER_KEY]] = by_tier.get(doc.metadata[TIER_KEY], 0) + 1
    print("Indexing " + ", ".join(f"{n} tier {t}" for t, n in sorted(by_tier.items())))

    nodes = to_nodes(kept, provenance, [Settings.node_parser])
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    write_manifest(PERSIST_DIR, decisions_path.name, DATA_DIR, screening_label(classifier))
    return index


def get_index(force_reindex: bool = False, classifier=None) -> VectorStoreIndex:
    """Reuse the saved index only if nothing it was built from has changed."""
    if force_reindex and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    if PERSIST_DIR.is_dir() and DATA_DIR.is_dir():
        origins = discover_origins(DATA_DIR, load_rules(TRUST_FILE))
        why = stale_reason(PERSIST_DIR, DECISIONS_DIR, DATA_DIR, origins,
                           load_latest_decisions(DECISIONS_DIR),
                           screening=screening_label(classifier))
        if why is None:
            print("Loading cached index.")
            return load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(PERSIST_DIR)))
        print(f"Rebuilding: {why}.")
        shutil.rmtree(PERSIST_DIR)

    return build_index(classifier)


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


def make_classifier():
    """The second-stage injection model. Cheap to build: it loads on first use.

    Only the packages are checked here (without importing them), so a missing
    install fails now with the fix, not halfway through the first question.
    """
    import importlib.util

    missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"The injection model needs: pip install {' '.join(missing)}\n"
                 "(or run with --no-classifier to use the pattern check only)")
    from pgrag.adapters.injection_classifier import InjectionClassifier
    return InjectionClassifier()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local RAG with provenance gating.")
    ap.add_argument("--reindex", action="store_true",
                    help="discard the cached index and re-run the review gate")
    ap.add_argument("--no-classifier", action="store_true",
                    help="patterns only: skip the local injection model")
    args = ap.parse_args()

    # Automatic: each document (at indexing) and each retrieved chunk (at query
    # time) goes through the pattern check first and reaches the model only if
    # the patterns found nothing. The model loads on that first need, not at
    # startup. A missing package fails now, loudly, because quietly running
    # without stage 2 would look like "nothing found".
    classifier = None if args.no_classifier else make_classifier()

    if latest_decisions_path(DECISIONS_DIR) is None:
        show_home()

    try:
        index = get_index(force_reindex=args.reindex, classifier=classifier)
    except DecisionsError as e:
        sys.exit(f"Cannot continue: {e}")

    # Every retrieved chunk is screened before the model reads it: the pattern
    # check first, then the classifier unless --no-classifier was given.
    retriever = index.as_retriever(similarity_top_k=5)
    sanitizer = InstructionSanitizerPostprocessor(classifier=classifier)

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

        nodes = sanitizer.postprocess_nodes(retriever.retrieve(line), query_str=line)
        # The capability gate: the tools this turn may use come from the
        # weakest tier among the chunks just retrieved, and nothing else.
        turn = run_gated_turn(Settings.llm, line, nodes, TOOLS)

        print(f"\n{turn.answer}\n" if turn.answer else "")
        for tool, args, output in turn.ran:
            print(f"  ran {tool}: {output}")
        for tool, args, why in turn.refused:
            print(f"  REFUSED {tool}({args}): {why}")
        withheld = ", ".join(f"{t} (needs {TOOL_REQUIRED_TIER[t].name})" for t in turn.withheld)
        print(f"Tools this turn: {', '.join(turn.allowed) or 'none'}"
              + (f"   withheld: {withheld}" if withheld else ""))
        print()

        srcs = []
        print("--- Sources ---")
        for node in nodes:
            meta = node.metadata
            tier = meta.get(TIER_KEY, "?")
            origin = meta.get(ORIGIN_KEY, "?")
            fname = meta.get("file_name", "?")
            score = float(node.score or 0)
            quarantined = " [QUARANTINED]" if meta.get(SANITIZER_FLAG_KEY) else ""
            print(f"[{score:.3f}] [tier {tier}]{quarantined} {origin}/{fname}: "
                  f"{node.text[:120].strip()}...")
            srcs.append((score, tier, origin, fname, node.text))
        print()

        last = {"q": line, "a": turn.answer, "s": srcs}


if __name__ == "__main__":
    main()