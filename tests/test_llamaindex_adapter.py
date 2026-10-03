from llama_index.core import Document

from pgrag.adapters.llamaindex import (
    ORIGIN_KEY,
    RESERVED_KEYS,
    TIER_KEY,
    stamp_documents,
)
from pgrag.core.registry import DEFAULT_TIER, ReviewDecision, TrustTier


def _doc(tmp_path, origin, filename="doc.txt", **extra_metadata):
    path = tmp_path / origin / filename if origin else tmp_path / filename
    return Document(text="content", metadata={"file_path": str(path), **extra_metadata})


def _decisions(**by_origin):
    return {
        name: ReviewDecision(name, tier, excluded, False, "reason")
        for name, (tier, excluded) in by_origin.items()
    }


# --- keeping and dropping -------------------------------------------------

def test_kept_document_is_stamped_with_its_tier_and_origin(tmp_path):
    decisions = _decisions(wiki=(TrustTier.B, False))

    kept, dropped, _ = stamp_documents([_doc(tmp_path, "wiki")], tmp_path, decisions)

    assert dropped == []
    assert kept[0].metadata[TIER_KEY] == "B"
    assert kept[0].metadata[ORIGIN_KEY] == "wiki"


def test_excluded_origin_is_dropped_and_never_stamped(tmp_path):
    decisions = _decisions(public_inbox=(TrustTier.D, True))

    kept, dropped, _ = stamp_documents([_doc(tmp_path, "public_inbox")], tmp_path, decisions)

    assert kept == []
    assert len(dropped) == 1


def test_document_with_no_origin_folder_is_dropped(tmp_path):
    kept, dropped, _ = stamp_documents([_doc(tmp_path, None)], tmp_path, {})

    assert kept == []
    assert len(dropped) == 1


def test_document_outside_the_corpus_is_dropped(tmp_path):
    outside = Document(text="x", metadata={"file_path": str(tmp_path.parent / "elsewhere.txt")})

    kept, dropped, _ = stamp_documents([outside], tmp_path, {})

    assert kept == []
    assert len(dropped) == 1


def test_document_with_no_file_path_is_dropped(tmp_path):
    # Fail closed: an unattributable document must not slip through untagged.
    kept, dropped, _ = stamp_documents([Document(text="x", metadata={})], tmp_path, {})

    assert kept == []
    assert len(dropped) == 1


def test_unreviewed_origin_is_kept_at_the_default_tier(tmp_path):
    kept, dropped, _ = stamp_documents([_doc(tmp_path, "brand_new")], tmp_path, {})

    assert dropped == []
    assert kept[0].metadata[TIER_KEY] == DEFAULT_TIER.name


def test_mixed_batch_is_split_correctly(tmp_path):
    decisions = _decisions(wiki=(TrustTier.B, False), public_inbox=(TrustTier.D, True))
    documents = [
        _doc(tmp_path, "wiki", "a.txt"),
        _doc(tmp_path, "public_inbox", "b.txt"),
        _doc(tmp_path, "wiki", "c.txt"),
    ]

    kept, dropped, _ = stamp_documents(documents, tmp_path, decisions)

    assert len(kept) == 2
    assert len(dropped) == 1
    assert all(d.metadata[ORIGIN_KEY] == "wiki" for d in kept)


def test_empty_batch_returns_two_empty_lists(tmp_path):
    assert stamp_documents([], tmp_path, {}) == ([], [], {})


# --- spoofing defence -----------------------------------------------------

def test_a_document_cannot_smuggle_in_its_own_tier(tmp_path):
    # A hostile file whose front-matter sets _prov_tier: A must still end up
    # at the tier its origin actually earned.
    decisions = _decisions(public_inbox=(TrustTier.D, False))
    doc = _doc(tmp_path, "public_inbox", **{TIER_KEY: "A", ORIGIN_KEY: "verified_internal_wiki"})

    kept, _, _ = stamp_documents([doc], tmp_path, decisions)

    assert kept[0].metadata[TIER_KEY] == "D"
    assert kept[0].metadata[ORIGIN_KEY] == "public_inbox"


def test_spoofed_keys_are_stripped_even_when_the_document_is_dropped(tmp_path):
    decisions = _decisions(public_inbox=(TrustTier.D, True))
    doc = _doc(tmp_path, "public_inbox", **{TIER_KEY: "A"})

    stamp_documents([doc], tmp_path, decisions)

    assert TIER_KEY not in doc.metadata


def test_unrelated_metadata_is_left_alone(tmp_path):
    decisions = _decisions(wiki=(TrustTier.B, False))
    doc = _doc(tmp_path, "wiki", author="someone", page=7)

    kept, _, _ = stamp_documents([doc], tmp_path, decisions)

    assert kept[0].metadata["author"] == "someone"
    assert kept[0].metadata["page"] == 7


# --- the tier must never reach the model or the embedding -----------------

def test_reserved_keys_are_hidden_from_the_llm(tmp_path):
    decisions = _decisions(wiki=(TrustTier.B, False))

    kept, _, _ = stamp_documents([_doc(tmp_path, "wiki")], tmp_path, decisions)

    for key in RESERVED_KEYS:
        assert key in kept[0].excluded_llm_metadata_keys


def test_reserved_keys_are_hidden_from_the_embedding(tmp_path):
    decisions = _decisions(wiki=(TrustTier.B, False))

    kept, _, _ = stamp_documents([_doc(tmp_path, "wiki")], tmp_path, decisions)

    for key in RESERVED_KEYS:
        assert key in kept[0].excluded_embed_metadata_keys


def test_tier_does_not_appear_in_the_text_the_llm_sees(tmp_path):
    decisions = _decisions(wiki=(TrustTier.B, False))

    kept, _, _ = stamp_documents([_doc(tmp_path, "wiki")], tmp_path, decisions)

    assert TIER_KEY not in kept[0].get_content(metadata_mode="llm")


def test_restamping_does_not_duplicate_the_exclusion_entries(tmp_path):
    decisions = _decisions(wiki=(TrustTier.B, False))
    doc = _doc(tmp_path, "wiki")

    stamp_documents([doc], tmp_path, decisions)
    stamp_documents([doc], tmp_path, decisions)

    assert doc.excluded_llm_metadata_keys.count(TIER_KEY) == 1
    assert doc.excluded_embed_metadata_keys.count(TIER_KEY) == 1


def test_a_retier_is_reflected_on_a_second_pass(tmp_path):
    doc = _doc(tmp_path, "wiki")
    stamp_documents([doc], tmp_path, _decisions(wiki=(TrustTier.D, False)))

    kept, _, _ = stamp_documents([doc], tmp_path, _decisions(wiki=(TrustTier.A, False)))

    assert kept[0].metadata[TIER_KEY] == "A"
