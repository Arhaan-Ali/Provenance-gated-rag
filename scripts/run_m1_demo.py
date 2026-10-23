"""M1 demo: origin registry + review gate before indexing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import DECISIONS, M1_CORPUS, TRUST_RULES

from provenance_gated_rag.config import require_ollama
from provenance_gated_rag.ingest import build_plain_query_engine

QUERY = "Who is the CEO of Meridian Robotics?"


def main():
    require_ollama()
    # The plain engine: this demo shows what the review gate does on its own.
    query_engine = build_plain_query_engine(M1_CORPUS, DECISIONS, TRUST_RULES)

    print(f"\nQuery: {QUERY}\n")
    response = query_engine.query(QUERY)

    print("--- Answer ---")
    print(response)

    print("\n--- Retrieved sources ---")
    for node in response.source_nodes:
        origin = node.node.metadata.get("origin")
        tier = node.node.metadata.get("trust_tier")
        print(f"[{node.score:.3f}] origin={origin} tier={tier}")
        print(node.node.get_content()[:200].replace("\n", " ") + "...")
        print()


if __name__ == "__main__":
    main()
