"""M4 demo: instruction sanitizer, run on top of the M0 baseline pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import M0_CORPUS

from provenance_gated_rag.baseline import build_query_engine
from provenance_gated_rag.config import require_ollama
from provenance_gated_rag.sanitizer import InstructionSanitizerPostprocessor

QUERY = "Who is the CEO of Meridian Robotics?"


def run(node_postprocessors, drop_instruction_documents=False):
    query_engine = build_query_engine(
        M0_CORPUS,
        node_postprocessors=node_postprocessors,
        drop_instruction_documents=drop_instruction_documents,
    )
    response = query_engine.query(QUERY)
    print("Answer:", response)
    print("Retrieved sources:")
    for node in response.source_nodes:
        flagged = node.node.metadata.get("sanitizer_flagged", False)
        print(f"  [{node.score:.3f}] flagged={flagged} {node.node.get_content()[:90]!r}")


def main():
    require_ollama()
    print(f"Query: {QUERY}\n")

    print("=== Sanitizer OFF ===")
    run(node_postprocessors=[])

    print("\n=== Sanitizer ON ===")
    run(
        node_postprocessors=[InstructionSanitizerPostprocessor()],
        drop_instruction_documents=True,
    )


if __name__ == "__main__":
    main()
