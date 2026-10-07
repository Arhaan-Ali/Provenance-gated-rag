import json
from datetime import datetime, timedelta, timezone

import pytest

from pgrag.core.registry import (
    DEFAULT_TIER,
    EDITABILITY,
    REVIEW_MAX_AGE_DAYS,
    Origin,
    ReviewDecision,
    Rule,
    TrustTier,
    apply_editability_cap,
    discover_origins,
    latest_decisions_path,
    load_latest_decisions,
    load_rules,
    needs_review,
    origin_of,
    origins_needing_review,
    save_decisions,
    tier_for_document,
)

NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)

def _make_origin_dir(tmp_path, name, file_count):
    origin_dir = tmp_path / name
    origin_dir.mkdir()
    for i in range(file_count):
        (origin_dir / f"doc_{i}.txt").write_text(f"content {i}")
    return origin_dir


# --- TrustTier ------------------------------------------------------------

def test_tiers_are_ordered_lowest_trust_to_highest():
    # The capability gate is min(tiers), so this ordering is load-bearing.
    assert TrustTier.D < TrustTier.C < TrustTier.B < TrustTier.A


def test_default_tier_is_the_lowest():
    assert DEFAULT_TIER is TrustTier.D
    assert DEFAULT_TIER is min(TrustTier)


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        (TrustTier.A, "User Authored"),
        (TrustTier.B, "Internal Documents"),
        (TrustTier.C, "External Verified"),
        (TrustTier.D, "External Un-verified"),
    ],
)
def test_every_tier_has_a_description(tier, expected):
    assert tier.description == expected


def test_str_shows_the_letter_not_the_number():
    # Without __str__, an IntEnum prints its value and the review screen
    # would show "3" instead of "B".
    assert str(TrustTier.B) == "B"


def test_format_shows_the_letter_in_an_fstring():
    assert f"{TrustTier.A}" == "A"
    assert f"{TrustTier.C:>3}" == "  C"


# --- load_rules -----------------------------------------------------------

def test_load_rules_creates_template_on_first_run(tmp_path):
    rules_path = tmp_path / "trust.yaml"

    rules = load_rules(rules_path)

    assert rules_path.exists()
    assert rules["verified_internal_wiki"].tier == TrustTier.B


def test_load_rules_creates_missing_parent_directories(tmp_path):
    rules_path = tmp_path / "nested" / "deeper" / "trust.yaml"

    load_rules(rules_path)

    assert rules_path.exists()


def test_load_rules_reads_existing_yaml(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text(
        'origins:\n  some_partner_feed:\n    tier: C\n    reason: "TLS-verified partner feed"\n'
    )

    rules = load_rules(rules_path)

    assert rules["some_partner_feed"] == Rule(TrustTier.C, "TLS-verified partner feed", None)


@pytest.mark.parametrize("written_tier", ["b", " B ", "B\n"])
def test_load_rules_normalises_tier_case_and_whitespace(tmp_path, written_tier):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text(f'origins:\n  wiki:\n    tier: "{written_tier}"\n')

    rules = load_rules(rules_path)

    assert rules["wiki"].tier == TrustTier.B


def test_load_rules_supplies_a_default_reason_when_omitted(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  wiki:\n    tier: B\n")

    rules = load_rules(rules_path)

    assert rules["wiki"].reason == "Set in trust.yaml"


def test_load_rules_rejects_unknown_tier(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  bad_origin:\n    tier: Z\n")

    with pytest.raises(ValueError, match="unknown tier"):
        load_rules(rules_path)


def test_load_rules_rejects_missing_tier_key(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text('origins:\n  bad_origin:\n    reason: "no tier given"\n')

    with pytest.raises(ValueError, match="needs a 'tier' key"):
        load_rules(rules_path)


def test_load_rules_rejects_origins_that_are_not_a_mapping(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  - just_a_list_item\n")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_rules(rules_path)


def test_load_rules_returns_empty_for_an_empty_file(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("")

    assert load_rules(rules_path) == {}


def test_load_rules_returns_empty_when_origins_key_is_absent(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("something_else: true\n")

    assert load_rules(rules_path) == {}

# --- Load Editability Rules  ----------------------------------------------

def test_load_rules_reads_editability(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  wiki:\n    tier: B\n    editability: org-wide\n")
 
    assert load_rules(rules_path)["wiki"].editability == "org-wide"
 
 
def test_load_rules_editability_is_none_when_not_written(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  wiki:\n    tier: B\n")
 
    assert load_rules(rules_path)["wiki"].editability is None
 
 
@pytest.mark.parametrize("written", ["Org-Wide", " org-wide ", "ORG-WIDE"])
def test_load_rules_normalises_editability_case_and_whitespace(tmp_path, written):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text(f'origins:\n  wiki:\n    tier: B\n    editability: "{written}"\n')
 
    assert load_rules(rules_path)["wiki"].editability == "org-wide"
 
 
def test_load_rules_rejects_unknown_editability(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  wiki:\n    tier: B\n    editability: everyone\n")
 
    with pytest.raises(ValueError, match="unknown editability"):
        load_rules(rules_path)
 
 
def test_the_starter_template_declares_editability(tmp_path):
    rules = load_rules(tmp_path / "trust.yaml")
 
    assert rules["verified_internal_wiki"].editability == "org-wide"

# --- discover_origins -----------------------------------------------------

def test_discover_origins_known_origin_gets_rubric_tier(tmp_path):
    _make_origin_dir(tmp_path, "verified_internal_wiki", 3)
    rules = {"verified_internal_wiki": Rule(TrustTier.B, "reason", "authenticated-few")}

    origins = discover_origins(tmp_path, rules)

    assert len(origins) == 1
    assert origins[0].tier == TrustTier.B
    assert origins[0].document_count == 3


def test_discover_origins_unknown_origin_defaults_to_lowest_tier(tmp_path):
    _make_origin_dir(tmp_path, "some_random_new_source", 2)

    origins = discover_origins(tmp_path, rules={})

    assert origins[0].tier == DEFAULT_TIER
    assert "not listed in trust.yaml" in origins[0].reason


def test_discover_origins_ignores_loose_files_at_top_level(tmp_path):
    _make_origin_dir(tmp_path, "verified_internal_wiki", 1)
    (tmp_path / "stray_file.txt").write_text("not an origin")

    origins = discover_origins(tmp_path, rules={})

    assert [o.name for o in origins] == ["verified_internal_wiki"]


def test_discover_origins_counts_nested_files_recursively(tmp_path):
    origin_dir = _make_origin_dir(tmp_path, "wiki", 2)
    nested = origin_dir / "subfolder"
    nested.mkdir()
    (nested / "extra.txt").write_text("nested content")

    origins = discover_origins(tmp_path, rules={})

    assert origins[0].document_count == 3


def test_discover_origins_returns_origins_sorted_by_name(tmp_path):
    for name in ["zebra", "alpha", "middle"]:
        _make_origin_dir(tmp_path, name, 1)

    origins = discover_origins(tmp_path, rules={})

    assert [o.name for o in origins] == ["alpha", "middle", "zebra"]


def test_discover_origins_empty_corpus_returns_empty_list(tmp_path):
    assert discover_origins(tmp_path, rules={}) == []

#--- Editability Cap  -----------------------------------------------------
@pytest.mark.parametrize(
    ("declared", "editability", "expected"),
    [
        # wide editing caps the tier
        (TrustTier.A, "org-wide", TrustTier.C),
        (TrustTier.B, "org-wide", TrustTier.C),
        (TrustTier.A, "anonymous", TrustTier.D),
        (TrustTier.B, "anonymous", TrustTier.D),
        (TrustTier.C, "anonymous", TrustTier.D),
        # not declared = assume the worst (fail closed)
        (TrustTier.A, None, TrustTier.D),
        # control: a narrow editor set keeps the declared tier
        (TrustTier.A, "immutable", TrustTier.A),
        (TrustTier.A, "authenticated-few", TrustTier.A),
        (TrustTier.B, "immutable", TrustTier.B),
        # already at or below the cap: untouched, never raised
        (TrustTier.C, "org-wide", TrustTier.C),
        (TrustTier.D, "org-wide", TrustTier.D),
        (TrustTier.D, "anonymous", TrustTier.D),
        (TrustTier.D, None, TrustTier.D),
    ],
)
def test_editability_cap_table(declared, editability, expected):
    assert apply_editability_cap(declared, editability)[0] == expected
 
 
@pytest.mark.parametrize("editability", [*EDITABILITY, None])
@pytest.mark.parametrize("tier", list(TrustTier))
def test_the_cap_never_raises_a_tier(tier, editability):
    new_tier, _ = apply_editability_cap(tier, editability)
 
    assert new_tier <= tier
 
 
def test_cap_note_says_what_changed_and_why():
    _, note = apply_editability_cap(TrustTier.A, "org-wide")
 
    assert note == "A -> C: editable by org-wide"
 
 
def test_cap_note_explains_an_undeclared_editability():
    _, note = apply_editability_cap(TrustTier.A, None)
 
    assert "not declared" in note
 
 
def test_no_note_when_nothing_was_capped():
    assert apply_editability_cap(TrustTier.A, "immutable")[1] is None
    assert apply_editability_cap(TrustTier.D, "anonymous")[1] is None
 
 
def test_an_org_wide_origin_declared_A_comes_out_C(tmp_path):
    _make_origin_dir(tmp_path, "team_wiki", 2)
    rules = {"team_wiki": Rule(TrustTier.A, "reason", "org-wide")}
 
    [origin] = discover_origins(tmp_path, rules)
 
    assert origin.tier == TrustTier.C
    assert origin.editability == "org-wide"
    assert "org-wide" in origin.cap_note
 
 
def test_an_anonymous_origin_comes_out_D(tmp_path):
    _make_origin_dir(tmp_path, "forum", 2)
    rules = {"forum": Rule(TrustTier.C, "reason", "anonymous")}
 
    assert discover_origins(tmp_path, rules)[0].tier == TrustTier.D
 
 
def test_an_undeclared_editability_comes_out_D(tmp_path):
    _make_origin_dir(tmp_path, "notes", 1)
    rules = {"notes": Rule(TrustTier.A, "reason", None)}
 
    [origin] = discover_origins(tmp_path, rules)
 
    assert origin.tier == TrustTier.D
    assert "not declared" in origin.cap_note
 
 
def test_an_immutable_origin_keeps_its_declared_tier(tmp_path):
    _make_origin_dir(tmp_path, "notes", 1)
    rules = {"notes": Rule(TrustTier.A, "reason", "immutable")}
 
    [origin] = discover_origins(tmp_path, rules)
 
    assert origin.tier == TrustTier.A
    assert origin.cap_note is None
 
 
def test_a_human_override_beats_the_cap(tmp_path):
    # The cap lowers the COMPUTED tier. A reviewer's override still wins (F2).
    decisions = {"wiki": ReviewDecision("wiki", TrustTier.A, False, True, "overridden")}
 
    assert tier_for_document(tmp_path / "wiki" / "doc.txt", tmp_path, decisions) == TrustTier.A

# --- save_decisions / load_latest_decisions -------------------------------

def test_save_decisions_increments_version(tmp_path):
    decisions = [ReviewDecision("wiki", TrustTier.B, False, False, "reason")]

    first = save_decisions(decisions, tmp_path)
    second = save_decisions(decisions, tmp_path)

    assert first.name == "decisions_v1.json"
    assert second.name == "decisions_v2.json"


def test_save_decisions_creates_the_directory_if_absent(tmp_path):
    decisions_dir = tmp_path / "not_yet_created"

    out_path = save_decisions([], decisions_dir)

    assert out_path.exists()


def test_save_decisions_writes_correct_schema(tmp_path):
    decisions = [ReviewDecision("public_inbox", TrustTier.D, True, False, "excluded by reviewer")]

    payload = json.loads(save_decisions(decisions, tmp_path).read_text())
    written = payload["decisions"][0]

    assert payload["version"] == 1
    assert len(payload["decisions"]) == 1
    assert {k: written[k] for k in
            ("origin_name", "tier", "excluded", "overridden", "reason", "editability")} == {
        "origin_name": "public_inbox",
        "tier": "D",
        "excluded": True,
        "overridden": False,
        "reason": "excluded by reviewer",
        "editability": None,
    }
    # Stamped at save time when the decision carries none, so it cannot be
    # asserted literally -- only that it is present and UTC.
    assert written["reviewed_at"].endswith("+00:00")


def test_save_decisions_records_an_utc_timestamp(tmp_path):
    payload = json.loads(save_decisions([], tmp_path).read_text())

    assert payload["decided_at"].endswith("+00:00")


def test_save_decisions_ignores_non_matching_filenames(tmp_path):
    save_decisions([], tmp_path)
    (tmp_path / "decisions_v1_backup.json").write_text("{}")

    assert save_decisions([], tmp_path).name == "decisions_v2.json"


def test_versions_are_compared_numerically_not_as_text(tmp_path):
    # "v9" sorts after "v10" as text; the audit trail would silently
    # start overwriting once it reached double digits.
    for _ in range(10):
        save_decisions([], tmp_path)

    assert latest_decisions_path(tmp_path).name == "decisions_v10.json"
    assert save_decisions([], tmp_path).name == "decisions_v11.json"


def test_latest_decisions_path_is_none_when_nothing_saved(tmp_path):
    assert latest_decisions_path(tmp_path) is None


def test_load_latest_decisions_empty_when_no_files(tmp_path):
    assert load_latest_decisions(tmp_path) == {}


def test_load_latest_decisions_returns_newest_version(tmp_path):
    save_decisions([ReviewDecision("wiki", TrustTier.B, False, False, "first")], tmp_path)
    save_decisions([ReviewDecision("wiki", TrustTier.A, True, True, "second")], tmp_path)

    latest = load_latest_decisions(tmp_path)

    assert latest["wiki"].tier == TrustTier.A
    assert latest["wiki"].excluded is True
    assert latest["wiki"].overridden is True
    assert latest["wiki"].reason == "second"


def test_decisions_survive_a_save_load_round_trip(tmp_path):
    original = [
        ReviewDecision("wiki", TrustTier.B, False, False, "accepted"),
        ReviewDecision("public_inbox", TrustTier.D, True, False, "excluded"),
    ]

    save_decisions(original, tmp_path)
    loaded = load_latest_decisions(tmp_path)

    assert set(loaded) == {d.origin_name for d in original}
    for decision in original:
        got = loaded[decision.origin_name]
        assert (got.tier, got.excluded, got.overridden, got.reason) == (
            decision.tier, decision.excluded, decision.overridden, decision.reason
        )
        # save_decisions stamps a review time when the decision carries none,
        # so the round trip is not identity.
        assert got.reviewed_at is not None

def test_save_decisions_records_the_origin_strategy(tmp_path):
    payload = json.loads(save_decisions([], tmp_path).read_text())
 
    assert payload["origin_strategy"] == "folder"
 
 
def test_editability_survives_a_save_load_round_trip(tmp_path):
    save_decisions([ReviewDecision("wiki", TrustTier.C, False, False, "r", editability="org-wide")], tmp_path)
 
    assert load_latest_decisions(tmp_path)["wiki"].editability == "org-wide"
 
 
def test_save_keeps_a_review_date_the_decision_already_carries(tmp_path):
    # Decay depends on this: a reused decision must keep its OLD date, or the
    # date refreshes on every run and no origin ever goes stale.
    old = NOW - timedelta(days=40)
    save_decisions([ReviewDecision("wiki", TrustTier.B, False, False, "r", reviewed_at=old)], tmp_path)
 
    assert load_latest_decisions(tmp_path)["wiki"].reviewed_at == old
 
 
def test_an_old_decisions_file_without_the_new_fields_still_loads(tmp_path):
    (tmp_path / "decisions_v1.json").write_text(json.dumps({
        "version": 1,
        "decided_at": "2026-06-01T10:00:00+00:00",
        "decisions": [{"origin_name": "wiki", "tier": "B", "excluded": False,
                       "overridden": False, "reason": "r"}],
    }))
 
    loaded = load_latest_decisions(tmp_path)["wiki"]
 
    assert loaded.editability is None
    assert loaded.reviewed_at == datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)   # falls back to the file date
 
 
# --- decay v1: when must an origin be asked about again -------------------
 
def _origin_obj(tier=TrustTier.B, editability="authenticated-few"):
    return Origin("wiki", tier, "reason", 3, editability)
 
 
def _previous(days_ago=10, tier=TrustTier.B, editability="authenticated-few", overridden=False):
    return ReviewDecision("wiki", tier, False, overridden, "reason",
                          editability=editability, reviewed_at=NOW - timedelta(days=days_ago))
 
 
def test_a_new_origin_needs_review():
    assert needs_review(_origin_obj(), None, NOW) == "new origin"
 
 
def test_a_decision_with_no_review_date_needs_review():
    previous = _previous()
    previous.reviewed_at = None
 
    assert needs_review(_origin_obj(), previous, NOW) == "no review date on record"
 
 
@pytest.mark.parametrize(
    ("days", "stale"),
    [
        (0, False),
        (10, False),
        (REVIEW_MAX_AGE_DAYS, False),       # exactly at the limit is still fresh
        (REVIEW_MAX_AGE_DAYS + 1, True),
        (365, True),
    ],
)
def test_a_review_goes_stale_only_after_the_limit(days, stale):
    why = needs_review(_origin_obj(), _previous(days_ago=days), NOW)
 
    assert (why is not None) == stale
 
 
def test_one_second_past_the_limit_is_stale():
    previous = _previous(days_ago=REVIEW_MAX_AGE_DAYS)
    previous.reviewed_at -= timedelta(seconds=1)
 
    assert needs_review(_origin_obj(), previous, NOW) is not None
 
 
def test_the_stale_reason_says_how_old_it_is():
    why = needs_review(_origin_obj(), _previous(days_ago=120), NOW)
 
    assert "120 days" in why
 
 
def test_a_fresh_unchanged_origin_is_left_alone():
    assert needs_review(_origin_obj(), _previous(days_ago=10), NOW) is None
 
 
def test_changing_editability_in_trust_yaml_triggers_review():
    why = needs_review(_origin_obj(editability="anonymous"), _previous(editability="authenticated-few"), NOW)
 
    assert why == "editability changed in trust.yaml"
 
 
def test_changing_the_tier_in_trust_yaml_triggers_review():
    why = needs_review(_origin_obj(tier=TrustTier.A), _previous(tier=TrustTier.B), NOW)
 
    assert why == "tier changed in trust.yaml"
 
 
def test_a_reviewer_override_does_not_trigger_review_just_because_the_yaml_differs():
    # The reviewer chose A on purpose; the yaml saying B is expected, not a change.
    previous = _previous(tier=TrustTier.A, overridden=True)
 
    assert needs_review(_origin_obj(tier=TrustTier.B), previous, NOW) is None
 
 
def test_origins_needing_review_lists_only_the_ones_that_need_it():
    origins = [
        Origin("fresh", TrustTier.B, "r", 1, "authenticated-few"),
        Origin("stale", TrustTier.B, "r", 1, "authenticated-few"),
        Origin("brand_new", TrustTier.D, "r", 1, None),
    ]
    decisions = {
        "fresh": ReviewDecision("fresh", TrustTier.B, False, False, "r", "authenticated-few", NOW - timedelta(days=5)),
        "stale": ReviewDecision("stale", TrustTier.B, False, False, "r", "authenticated-few", NOW - timedelta(days=200)),
    }
 
    result = origins_needing_review(origins, decisions, NOW)
 
    assert [name for name, _ in result] == ["stale", "brand_new"]
 
 
def test_nothing_needs_review_when_everything_is_fresh():
    origins = [Origin("wiki", TrustTier.B, "r", 1, "authenticated-few")]
 
    assert origins_needing_review(origins, {"wiki": _previous(days_ago=1)}, NOW) == []

# --- origin_of / tier_for_document ----------------------------------------

def test_origin_of_returns_top_level_folder(tmp_path):
    assert origin_of(tmp_path / "wiki" / "doc.txt", tmp_path) == "wiki"


def test_origin_of_returns_top_level_folder_for_deeply_nested_file(tmp_path):
    assert origin_of(tmp_path / "wiki" / "a" / "b" / "doc.txt", tmp_path) == "wiki"


def test_origin_of_returns_none_for_loose_file(tmp_path):
    assert origin_of(tmp_path / "doc.txt", tmp_path) is None


def test_origin_of_returns_none_outside_corpus(tmp_path):
    assert origin_of(tmp_path.parent / "unrelated" / "doc.txt", tmp_path) is None


def test_tier_for_document_excluded_returns_none(tmp_path):
    decisions = {"public_inbox": ReviewDecision("public_inbox", TrustTier.D, True, False, "r")}

    assert tier_for_document(tmp_path / "public_inbox" / "doc.txt", tmp_path, decisions) is None


def test_tier_for_document_reviewed_returns_decision_tier(tmp_path):
    decisions = {"wiki": ReviewDecision("wiki", TrustTier.B, False, False, "r")}

    assert tier_for_document(tmp_path / "wiki" / "doc.txt", tmp_path, decisions) == TrustTier.B


def test_tier_for_document_unreviewed_origin_returns_default_tier(tmp_path):
    # A newly added folder enters at the lowest trust rather than vanishing.
    assert tier_for_document(tmp_path / "brand_new" / "doc.txt", tmp_path, {}) == DEFAULT_TIER


def test_tier_for_document_no_origin_folder_returns_none(tmp_path):
    assert tier_for_document(tmp_path / "doc.txt", tmp_path, {}) is None


def test_tier_for_document_honours_an_override_above_the_default(tmp_path):
    decisions = {"wiki": ReviewDecision("wiki", TrustTier.A, False, True, "overridden")}

    assert tier_for_document(tmp_path / "wiki" / "doc.txt", tmp_path, decisions) == TrustTier.A


# --- dataclasses ----------------------------------------------------------

def test_origin_carries_everything_the_review_screen_prints(tmp_path):
    origin = Origin("wiki", TrustTier.B, "internal", 4)

    assert (origin.name, origin.tier, origin.reason, origin.document_count) == (
        "wiki",
        TrustTier.B,
        "internal",
        4,
    )
