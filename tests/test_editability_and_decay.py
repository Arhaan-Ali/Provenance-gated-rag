"""Tests for the editability caps and review decay added in eae04e3.

Both features shipped without tests. These cover the cap table, the note the
review screen shows, and every branch of needs_review.
"""
from datetime import datetime, timedelta, timezone

import pytest

from pgrag.core.registry import (
    EDITABILITY,
    EDITABILITY_CAP,
    REVIEW_MAX_AGE_DAYS,
    Origin,
    ReviewDecision,
    Rule,
    TrustTier,
    apply_editability_cap,
    discover_origins,
    load_rules,
    needs_review,
)

NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def _origin(tier=TrustTier.B, editability=None, name="wiki"):
    return Origin(name, tier, "reason", 3, editability)


def _previous(tier=TrustTier.B, *, editability=None, overridden=False,
              days_ago=1, reviewed=True):
    return ReviewDecision(
        "wiki", tier, False, overridden, "reason", editability,
        NOW - timedelta(days=days_ago) if reviewed else None,
    )


# --- apply_editability_cap -------------------------------------------------

@pytest.mark.parametrize(
    ("editability", "expected"),
    [
        ("immutable", TrustTier.A),
        ("authenticated-few", TrustTier.A),
        ("org-wide", TrustTier.C),
        ("anonymous", TrustTier.D),
        (None, TrustTier.D),
    ],
)
def test_editability_caps_the_tier(editability, expected):
    assert apply_editability_cap(TrustTier.A, editability)[0] == expected


def test_undeclared_editability_caps_as_hard_as_anonymous():
    # Silence is not evidence that few people can edit a source. Only the
    # note differs, so a reviewer can still tell the two cases apart.
    undeclared, _ = apply_editability_cap(TrustTier.A, None)
    anonymous, _ = apply_editability_cap(TrustTier.A, "anonymous")

    assert undeclared == anonymous == TrustTier.D


def test_a_cap_never_raises_a_tier():
    # A source already at D stays D; the cap only ever demotes.
    assert apply_editability_cap(TrustTier.D, "org-wide") == (TrustTier.D, None)


def test_no_note_when_nothing_was_capped():
    assert apply_editability_cap(TrustTier.B, "immutable") == (TrustTier.B, None)


def test_the_note_names_both_tiers_and_the_reason():
    _, note = apply_editability_cap(TrustTier.A, "org-wide")

    assert "A -> C" in note
    assert "org-wide" in note


def test_the_note_distinguishes_an_undeclared_source():
    _, note = apply_editability_cap(TrustTier.A, None)

    assert "not declared" in note


def test_every_declared_editability_value_is_handled():
    # A value accepted by load_rules but missing from the cap table would
    # silently mean "no cap", which is the unsafe direction to fail in.
    for value in EDITABILITY:
        tier, _ = apply_editability_cap(TrustTier.A, value)
        assert isinstance(tier, TrustTier)


def test_cap_table_only_contains_known_editability_values():
    assert set(EDITABILITY_CAP) - {None} <= set(EDITABILITY)


# --- discover_origins applies the cap --------------------------------------

def test_discover_origins_applies_the_cap(tmp_path):
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "doc.txt").write_text("content")

    origin = discover_origins(tmp_path, {"wiki": Rule(TrustTier.B, "reason", "org-wide")})[0]

    assert origin.tier == TrustTier.C
    assert origin.editability == "org-wide"
    assert "B -> C" in origin.cap_note


def test_discover_origins_leaves_an_uncapped_origin_alone(tmp_path):
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "doc.txt").write_text("content")

    origin = discover_origins(tmp_path, {"wiki": Rule(TrustTier.B, "reason", "immutable")})[0]

    assert origin.tier == TrustTier.B
    assert origin.cap_note is None


# --- load_rules reads and validates editability -----------------------------

def test_load_rules_reads_editability(tmp_path):
    path = tmp_path / "trust.yaml"
    path.write_text("origins:\n  wiki:\n    tier: B\n    editability: ORG-WIDE\n")

    assert load_rules(path)["wiki"].editability == "org-wide"


def test_load_rules_defaults_editability_to_none(tmp_path):
    path = tmp_path / "trust.yaml"
    path.write_text("origins:\n  wiki:\n    tier: B\n")

    assert load_rules(path)["wiki"].editability is None


def test_load_rules_rejects_unknown_editability(tmp_path):
    path = tmp_path / "trust.yaml"
    path.write_text("origins:\n  wiki:\n    tier: B\n    editability: whoever\n")

    with pytest.raises(ValueError, match="unknown editability"):
        load_rules(path)


# --- needs_review ----------------------------------------------------------

def test_a_settled_origin_is_not_re_reviewed():
    assert needs_review(_origin(), _previous(), NOW) is None


def test_an_origin_with_no_prior_decision_is_new():
    assert needs_review(_origin(), None, NOW) == "new origin"


def test_a_decision_with_no_timestamp_is_re_reviewed():
    # Decisions written before reviewed_at existed cannot be aged, so they
    # are asked again rather than trusted indefinitely.
    assert needs_review(_origin(), _previous(reviewed=False), NOW) == "no review date on record"


def test_a_decision_older_than_the_max_age_is_re_reviewed():
    reason = needs_review(_origin(), _previous(days_ago=REVIEW_MAX_AGE_DAYS + 1), NOW)

    assert reason is not None
    assert "days ago" in reason


def test_a_decision_exactly_at_the_max_age_still_stands():
    assert needs_review(_origin(), _previous(days_ago=REVIEW_MAX_AGE_DAYS), NOW) is None


def test_a_changed_editability_triggers_re_review():
    # The tier did not move, but who can edit the source did -- exactly the
    # change a tier comparison cannot see.
    previous = _previous(editability="authenticated-few")

    assert needs_review(_origin(editability="org-wide"), previous, NOW) == (
        "editability changed in trust.yaml"
    )


def test_a_changed_computed_tier_triggers_re_review():
    assert needs_review(_origin(tier=TrustTier.C), _previous(tier=TrustTier.B), NOW) == (
        "tier changed in trust.yaml"
    )


def test_an_override_is_not_treated_as_drift():
    # The reviewer deliberately set A against a computed B. Asking again
    # every run would be nagging about a decision they already made.
    previous = _previous(tier=TrustTier.A, overridden=True)

    assert needs_review(_origin(tier=TrustTier.B), previous, NOW) is None


def test_an_overridden_decision_still_ages_out():
    # An override is deliberate, but not permanent.
    previous = _previous(tier=TrustTier.A, overridden=True, days_ago=REVIEW_MAX_AGE_DAYS + 30)

    assert needs_review(_origin(tier=TrustTier.B), previous, NOW) is not None


def test_age_is_checked_before_tier_and_editability():
    # A stale decision reports its age rather than a field change, so the
    # reviewer is told the most important reason first.
    previous = _previous(tier=TrustTier.D, editability="anonymous",
                         days_ago=REVIEW_MAX_AGE_DAYS + 5)

    assert "days ago" in needs_review(_origin(), previous, NOW)
