import json

import pytest

from pgrag.core.registry import (
    Rule,
    DEFAULT_TIER,
    Origin,
    ReviewDecision,
    TrustTier,
    discover_origins,
    latest_decisions_path,
    load_latest_decisions,
    load_rules,
    origin_of,
    save_decisions,
    tier_for_document,
)


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
