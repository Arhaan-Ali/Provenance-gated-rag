"""M0 baseline: a plain RAG pipeline with no provenance controls."""
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

from provenance_gated_rag.config import get_embed_model, get_llm
from provenance_gated_rag.ingest import FILE_EXTRACTOR
from provenance_gated_rag.sanitizer import sanitize_documents

# Relative to the caller's folder; see ingest.DEFAULT_CORPUS_DIR.
DEFAULT_CORPUS_DIR = Path("corpus")


def build_query_engine(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    similarity_top_k: int = 4,
    node_postprocessors: list | None = None,
    drop_instruction_documents: bool = False,
):
    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()

    documents = SimpleDirectoryReader(
        str(corpus_dir), exclude_hidden=False, file_extractor=FILE_EXTRACTOR
    ).load_data()
    if drop_instruction_documents:
        documents, _ = sanitize_documents(documents)
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=node_postprocessors or [],
    )
