"""Cache freshness: when may a saved index be reused, and when must it be rebuilt?"""
import os
from datetime import UTC, datetime, timedelta

import pytest

from provenance_gated_rag.freshness import (
    MANIFEST_NAME,
    data_fingerprint,
    read_manifest,
    stale_reason,
    write_manifest,
)
from provenance_gated_rag.registry import (
    Origin,
    ReviewDecision,
    TrustTier,
    save_decisions,
)

NOW = datetime.now(UTC)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "data"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "notes" / "b.txt").write_text("beta", encoding="utf-8")
    return root


def origins():
    return [Origin("notes", TrustTier.A, "mine", 2, editability="immutable")]


def decisions(reviewed_at=NOW):
    return {"notes": ReviewDecision("notes", TrustTier.A, False, False, "mine",
                                    editability="immutable", reviewed_at=reviewed_at,
                                    files={})}


@pytest.fixture
def built(tmp_path, corpus):
    """An index that was just built: decisions saved, manifest written."""
    persist, dec_dir = tmp_path / "storage", tmp_path / "decisions"
    saved = save_decisions(list(decisions().values()), dec_dir)
    write_manifest(persist, saved.name, corpus)
    return persist, dec_dir


def why(built, corpus, **kw):
    persist, dec_dir = built
    return stale_reason(persist, dec_dir, corpus, kw.get("origins", origins()),
                        kw.get("decisions", decisions()), kw.get("now", NOW))


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_is_stable(corpus):
    assert data_fingerprint(corpus) == data_fingerprint(corpus)


def test_fingerprint_changes_when_a_file_is_edited(corpus):
    before = data_fingerprint(corpus)
    (corpus / "notes" / "a.txt").write_text("ALPHA", encoding="utf-8")   # same size
    assert data_fingerprint(corpus) != before


def test_fingerprint_changes_when_a_file_is_added_or_deleted(corpus):
    before = data_fingerprint(corpus)
    (corpus / "notes" / "c.txt").write_text("gamma", encoding="utf-8")
    added = data_fingerprint(corpus)
    assert added != before
    (corpus / "notes" / "c.txt").unlink()
    assert data_fingerprint(corpus) == before


def test_fingerprint_changes_when_a_file_is_renamed(corpus):
    before = data_fingerprint(corpus)
    (corpus / "notes" / "a.txt").rename(corpus / "notes" / "z.txt")
    assert data_fingerprint(corpus) != before


def test_fingerprint_ignores_modified_time(corpus):
    before = data_fingerprint(corpus)
    p = corpus / "notes" / "a.txt"
    os.utime(p, (1_000_000_000, 1_000_000_000))                          # touch, content same
    assert data_fingerprint(corpus) == before


# --- manifest ----------------------------------------------------------------

def test_manifest_round_trip(built):
    persist, _ = built
    m = read_manifest(persist)
    assert m["decisions_file"] == "decisions_v1.json"
    assert len(m["data_fingerprint"]) == 64


def test_missing_or_corrupt_manifest_reads_as_none(tmp_path):
    assert read_manifest(tmp_path / "nope") is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    assert read_manifest(tmp_path / "bad") is None


# --- stale_reason ------------------------------------------------------------

def test_fresh_index_is_reused(built, corpus):
    assert why(built, corpus) is None


def test_no_manifest_is_stale(tmp_path, corpus):
    assert "no index manifest" in stale_reason(
        tmp_path / "storage", tmp_path / "decisions", corpus, origins(), decisions(), NOW)


def test_new_decisions_file_makes_it_stale(built, corpus):
    _, dec_dir = built
    save_decisions(list(decisions().values()), dec_dir)                  # now v2
    assert "decisions changed" in why(built, corpus)


def test_edited_document_makes_it_stale(built, corpus):
    (corpus / "notes" / "a.txt").write_text("changed", encoding="utf-8")
    assert "documents in data/ changed" in why(built, corpus)


def test_added_document_makes_it_stale(built, corpus):
    (corpus / "notes" / "new.txt").write_text("hello", encoding="utf-8")
    assert "documents in data/ changed" in why(built, corpus)


def test_review_due_makes_it_stale(built, corpus):
    reason = why(built, corpus, now=NOW + timedelta(days=120))
    assert reason.startswith("review due")
    assert "notes" in reason


def test_new_origin_makes_it_stale(built, corpus):
    extra = origins() + [Origin("forum", TrustTier.D, "unknown", 1)]
    assert "review due" in why(built, corpus, origins=extra)