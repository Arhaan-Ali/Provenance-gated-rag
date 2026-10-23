"""M0 demo: a plain RAG pipeline confidently gives a wrong answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import M0_CORPUS

from provenance_gated_rag.baseline import build_query_engine
from provenance_gated_rag.config import require_ollama

QUERY = "Who is the CEO of Meridian Robotics?"


def main():
    require_ollama()
    print(f"Query: {QUERY}\n")
    query_engine = build_query_engine(M0_CORPUS)
    response = query_engine.query(QUERY)

    print("--- Answer ---")
    print(response)

    print("\n--- Retrieved sources ---")
    for node in response.source_nodes:
        print(f"[{node.score:.3f}] {node.node.metadata.get('file_name')}")
        print(node.node.get_content()[:200].replace("\n", " ") + "...")
        print()


if __name__ == "__main__":
    main()
