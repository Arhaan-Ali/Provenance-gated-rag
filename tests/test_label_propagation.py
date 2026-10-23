from pathlib import Path

import pytest
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from provenance_gated_rag.ingest import load_documents
from provenance_gated_rag.registry import ReviewDecision, TrustTier

M0_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "m0_corpus"
M1_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "m1_corpus"

QUERY = "Who is the CEO of Meridian Robotics?"


@pytest.fixture(autouse=True)
def _mock_embedding():
    """Labels surviving retrieval does not depend on ranking, so no model is
    needed -- which lets M2's claim run in CI rather than only on a machine
    with Ollama up."""
    previous = Settings._embed_model
    Settings.embed_model = MockEmbedding(embed_dim=8)
    yield
    Settings._embed_model = previous


def _retrieve_m1(decisions):
    # Keeps the planted document in the index on purpose: this test is about
    # labels surviving retrieval, and needs a D-tier node to check. Dropping
    # it would leave only B-tier nodes and the test would still pass while
    # silently no longer exercising D. Content filtering is M4's job and is
    # covered in tests/test_sanitizer.py.
    documents = load_documents(M1_CORPUS_DIR, decisions, drop_instruction_documents=False)
    index = VectorStoreIndex.from_documents(documents)
    return index.as_retriever(similarity_top_k=4).retrieve(QUERY)


def _retrieve_m0():
    documents = SimpleDirectoryReader(str(M0_CORPUS_DIR), exclude_hidden=False).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_retriever(similarity_top_k=4).retrieve(QUERY)


def test_m1_pipeline_preserves_origin_and_tier_metadata():
    decisions = [
        ReviewDecision("public_inbox_unverified", TrustTier.D, False, False, "test"),
        ReviewDecision("verified_internal_wiki", TrustTier.B, False, False, "test"),
    ]
    expected_tier = {"public_inbox_unverified": "D", "verified_internal_wiki": "B"}

    nodes = _retrieve_m1(decisions)
    assert nodes

    for node in nodes:
        origin = node.node.metadata.get("origin")
        tier = node.node.metadata.get("trust_tier")
        assert origin in expected_tier
        assert tier == expected_tier[origin]


def test_stock_pipeline_has_no_provenance_metadata():
    nodes = _retrieve_m0()
    assert nodes

    for node in nodes:
        assert "origin" not in node.node.metadata
        assert "trust_tier" not in node.node.metadata
