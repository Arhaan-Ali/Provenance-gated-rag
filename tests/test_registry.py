import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from provenance_gated_rag.registry import (
    DEFAULT_TIER,
    REVIEW_MAX_AGE_DAYS,
    DecisionsError,
    Origin,
    ReviewDecision,
    Rule,
    TrustTier,
    apply_editability_cap,
    discover_origins,
    load_latest_decisions,
    load_rules,
    needs_review,
    origin_of,
    run_review_gate,
    save_decisions,
    tier_for_document,
)
from provenance_gated_rag.signing import load_key, sign


def _make_origin_dir(tmp_path, name, file_count):
    origin_dir = tmp_path / name
    origin_dir.mkdir()
    for i in range(file_count):
        (origin_dir / f"doc_{i}.txt").write_text(f"content {i}")
    return origin_dir


# --- discover_origins -------------------------------------------------

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

    assert len(origins) == 1
    assert origins[0].tier == DEFAULT_TIER


def test_discover_origins_ignores_loose_files_at_top_level(tmp_path):
    _make_origin_dir(tmp_path, "verified_internal_wiki", 1)
    (tmp_path / "stray_file.txt").write_text("not an origin")

    origins = discover_origins(tmp_path, rules={})

    assert len(origins) == 1
    assert origins[0].name == "verified_internal_wiki"


def test_discover_origins_counts_nested_files_recursively(tmp_path):
    origin_dir = _make_origin_dir(tmp_path, "verified_internal_wiki", 2)
    nested = origin_dir / "subfolder"
    nested.mkdir()
    (nested / "extra.txt").write_text("nested content")

    origins = discover_origins(tmp_path, rules={})

    assert origins[0].document_count == 3


# --- run_review_gate ----------------------------------------------------

def test_run_review_gate_accept_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")

    decisions = run_review_gate([Origin("verified_internal_wiki", TrustTier.B, "reason", 3)])

    assert len(decisions) == 1
    assert decisions[0].excluded is False
    assert decisions[0].overridden is False
    assert decisions[0].tier == TrustTier.B


def test_run_review_gate_exclude(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "x")

    decisions = run_review_gate([Origin("public_inbox_unverified", TrustTier.D, "reason", 1)])

    assert decisions[0].excluded is True


def test_run_review_gate_override(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "a")

    decisions = run_review_gate([Origin("verified_internal_wiki", TrustTier.B, "reason", 3)])

    assert decisions[0].overridden is True
    assert decisions[0].tier == TrustTier.A


def test_run_review_gate_retries_on_invalid_input(monkeypatch):
    responses = iter(["not-a-real-choice", "z9", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    decisions = run_review_gate([Origin("verified_internal_wiki", TrustTier.B, "reason", 3)])

    assert decisions[0].excluded is False
    assert decisions[0].overridden is False


# --- save_decisions / load_latest_decisions ------------------------------

def test_save_decisions_increments_version(tmp_path):
    decisions = [ReviewDecision("verified_internal_wiki", TrustTier.B, False, False, "reason")]

    first_path = save_decisions(decisions, tmp_path)
    second_path = save_decisions(decisions, tmp_path)

    assert first_path.name == "decisions_v1.json"
    assert second_path.name == "decisions_v2.json"


def test_save_decisions_writes_correct_schema(tmp_path):
    decisions = [
        ReviewDecision("public_inbox_unverified", TrustTier.D, True, False, "excluded by reviewer"),
    ]

    out_path = save_decisions(decisions, tmp_path)
    payload = json.loads(out_path.read_text())

    assert payload["version"] == 1
    assert len(payload["decisions"]) == 1
    assert payload["decisions"][0] == {
        "origin_name": "public_inbox_unverified",
        "tier": "D",
        "excluded": True,
        "overridden": False,
        "reason": "excluded by reviewer",
        "reviewed_at": None,
        "editability": None,
        "files": None,
    }
    assert payload["previous_signature"] is None
    assert len(payload["signature"]) == 64


def test_save_decisions_ignores_non_matching_filenames(tmp_path):
    decisions = [ReviewDecision("verified_internal_wiki", TrustTier.B, False, False, "reason")]
    save_decisions(decisions, tmp_path)
    (tmp_path / "decisions_v1_backup.json").write_text("{}")

    second_path = save_decisions(decisions, tmp_path)

    assert second_path.name == "decisions_v2.json"


def test_load_latest_decisions_empty_when_no_files(tmp_path):
    assert load_latest_decisions(tmp_path) == {}


def test_load_latest_decisions_returns_newest_version(tmp_path):
    save_decisions([ReviewDecision("verified_internal_wiki", TrustTier.B, False, False, "first")], tmp_path)
    save_decisions([ReviewDecision("verified_internal_wiki", TrustTier.A, True, True, "second")], tmp_path)

    latest = load_latest_decisions(tmp_path)

    assert latest["verified_internal_wiki"].tier == TrustTier.A
    assert latest["verified_internal_wiki"].excluded is True
    assert latest["verified_internal_wiki"].reason == "second"


# --- load_rules -----------------------------------------------------------

def test_load_rules_creates_template_on_first_run(tmp_path):
    rules_path = tmp_path / "trust.yaml"

    rules = load_rules(rules_path)

    assert rules_path.exists()
    assert rules["verified_internal_wiki"].tier == TrustTier.B


def test_load_rules_reads_existing_yaml(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text(
        "origins:\n  some_partner_feed:\n    tier: C\n    reason: \"TLS-verified partner feed\"\n"
    )

    rules = load_rules(rules_path)

    assert rules["some_partner_feed"] == Rule(TrustTier.C, "TLS-verified partner feed")


def test_load_rules_rejects_unknown_tier(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  bad_origin:\n    tier: Z\n")

    with pytest.raises(ValueError):
        load_rules(rules_path)


def test_load_rules_rejects_missing_tier_key(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  bad_origin:\n    reason: \"no tier given\"\n")

    with pytest.raises(ValueError):
        load_rules(rules_path)


# --- origin_of / tier_for_document -----------------------------------------

def test_origin_of_returns_top_level_folder(tmp_path):
    doc_path = tmp_path / "verified_internal_wiki" / "doc.txt"

    assert origin_of(doc_path, tmp_path) == "verified_internal_wiki"


def test_origin_of_returns_none_for_loose_file(tmp_path):
    doc_path = tmp_path / "doc.txt"

    assert origin_of(doc_path, tmp_path) is None


def test_origin_of_returns_none_outside_corpus(tmp_path):
    outside = tmp_path.parent / "unrelated" / "doc.txt"

    assert origin_of(outside, tmp_path) is None


def test_tier_for_document_excluded_returns_none(tmp_path):
    doc_path = tmp_path / "public_inbox_unverified" / "doc.txt"
    decisions = {"public_inbox_unverified": ReviewDecision("public_inbox_unverified", TrustTier.D, True, False, "r")}

    assert tier_for_document(doc_path, tmp_path, decisions) is None


def test_tier_for_document_reviewed_returns_decision_tier(tmp_path):
    doc_path = tmp_path / "verified_internal_wiki" / "doc.txt"
    decisions = {"verified_internal_wiki": ReviewDecision("verified_internal_wiki", TrustTier.B, False, False, "r")}

    assert tier_for_document(doc_path, tmp_path, decisions) == TrustTier.B


def test_tier_for_document_unreviewed_origin_returns_default_tier(tmp_path):
    doc_path = tmp_path / "brand_new_origin" / "doc.txt"

    assert tier_for_document(doc_path, tmp_path, decisions={}) == DEFAULT_TIER


def test_tier_for_document_no_origin_folder_returns_none(tmp_path):
    doc_path = tmp_path / "doc.txt"

    assert tier_for_document(doc_path, tmp_path, decisions={}) is None


def test_tier_for_document_honours_an_override_above_the_default(tmp_path):
    doc_path = tmp_path / "verified_internal_wiki" / "doc.txt"
    decisions = {
        "verified_internal_wiki": ReviewDecision(
            "verified_internal_wiki", TrustTier.A, False, True, "overridden"
        )
    }

    assert tier_for_document(doc_path, tmp_path, decisions) == TrustTier.A


def test_origin_of_returns_top_level_folder_for_deeply_nested_file(tmp_path):
    doc_path = tmp_path / "verified_internal_wiki" / "a" / "b" / "doc.txt"

    assert origin_of(doc_path, tmp_path) == "verified_internal_wiki"


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


# --- load_rules edge cases -------------------------------------------------

def test_load_rules_creates_missing_parent_directories(tmp_path):
    rules_path = tmp_path / "nested" / "deeper" / "trust.yaml"

    load_rules(rules_path)

    assert rules_path.exists()


@pytest.mark.parametrize("written_tier", ["b", " B ", "B\n"])
def test_load_rules_normalises_tier_case_and_whitespace(tmp_path, written_tier):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text(f'origins:\n  wiki:\n    tier: "{written_tier}"\n')

    assert load_rules(rules_path)["wiki"].tier == TrustTier.B


def test_load_rules_supplies_a_default_reason_when_omitted(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  wiki:\n    tier: B\n")

    assert load_rules(rules_path)["wiki"].reason == "Set in trust.yaml"


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


# --- discover_origins edge cases -------------------------------------------

def test_discover_origins_returns_origins_sorted_by_name(tmp_path):
    for name in ["zebra", "alpha", "middle"]:
        _make_origin_dir(tmp_path, name, 1)

    origins = discover_origins(tmp_path, rules={})

    assert [o.name for o in origins] == ["alpha", "middle", "zebra"]


def test_discover_origins_empty_corpus_returns_empty_list(tmp_path):
    assert discover_origins(tmp_path, rules={}) == []


# --- decision audit trail --------------------------------------------------

def test_save_decisions_creates_the_directory_if_absent(tmp_path):
    decisions_dir = tmp_path / "not_yet_created"

    assert save_decisions([], decisions_dir).exists()


def test_save_decisions_records_an_utc_timestamp(tmp_path):
    payload = json.loads(save_decisions([], tmp_path).read_text())

    assert payload["decided_at"].endswith("+00:00")


def test_versions_are_compared_numerically_not_as_text(tmp_path):
    # "v9" sorts after "v10" as text; the audit trail would silently start
    # overwriting once it reached double digits.
    for _ in range(10):
        save_decisions([], tmp_path)

    assert save_decisions([], tmp_path).name == "decisions_v11.json"


def test_decisions_survive_a_save_load_round_trip(tmp_path):
    original = [
        ReviewDecision("verified_internal_wiki", TrustTier.B, False, False, "accepted"),
        ReviewDecision("public_inbox_unverified", TrustTier.D, True, False, "excluded"),
    ]

    save_decisions(original, tmp_path)
    loaded = load_latest_decisions(tmp_path)

    # Every field round-trips except reviewed_at, which is filled in rather
    # than preserved as None: save_decisions always writes the file's
    # decided_at, and the loader inherits it when a decision carries no date of
    # its own. That is accurate rather than invented -- the review gate wrote
    # the whole file at that moment, so that IS when these were decided.
    for d in original:
        got = loaded[d.origin_name]
        assert (got.tier, got.excluded, got.overridden, got.reason) == (
            d.tier, d.excluded, d.overridden, d.reason
        )

    # The consequence worth pinning: a decision loaded from disk is never
    # missing a review date, so needs_review's "no review date on record"
    # branch is unreachable through this path. It still guards a decision
    # constructed directly, which is what test_a_decision_with_no_timestamp_
    # is_re_reviewed covers.
    assert all(got.reviewed_at is not None for got in loaded.values())


# --- editability caps ------------------------------------------------------

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
    tier, _ = apply_editability_cap(TrustTier.A, editability)

    assert tier == expected


def test_undeclared_editability_is_treated_as_anonymous():
    # Silence is not evidence that few people can edit a source. The cap is
    # identical; only the note differs, so a reviewer can tell the two apart.
    undeclared_tier, _ = apply_editability_cap(TrustTier.A, None)
    anonymous_tier, _ = apply_editability_cap(TrustTier.A, "anonymous")

    assert undeclared_tier == anonymous_tier == TrustTier.D


def test_a_cap_never_raises_a_tier():
    # A wide-open source declared at D stays D; the cap only demotes.
    tier, note = apply_editability_cap(TrustTier.D, "org-wide")

    assert tier == TrustTier.D
    assert note is None


def test_no_note_when_nothing_was_capped():
    assert apply_editability_cap(TrustTier.B, "immutable") == (TrustTier.B, None)


def test_the_note_names_both_tiers_and_the_reason():
    _, note = apply_editability_cap(TrustTier.A, "org-wide")

    assert "A -> C" in note
    assert "org-wide" in note


def test_the_note_explains_an_undeclared_source_differently():
    _, note = apply_editability_cap(TrustTier.A, None)

    assert "not declared" in note


def test_discover_origins_applies_the_cap(tmp_path):
    _make_origin_dir(tmp_path, "wiki", 1)
    rules = {"wiki": Rule(TrustTier.B, "reason", "org-wide")}

    origin = discover_origins(tmp_path, rules)[0]

    assert origin.tier == TrustTier.C
    assert origin.editability == "org-wide"
    assert "B -> C" in origin.cap_note


def test_discover_origins_leaves_an_uncapped_origin_alone(tmp_path):
    _make_origin_dir(tmp_path, "wiki", 1)
    rules = {"wiki": Rule(TrustTier.B, "reason", "immutable")}

    origin = discover_origins(tmp_path, rules)[0]

    assert origin.tier == TrustTier.B
    assert origin.cap_note is None


def test_load_rules_reads_editability(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  wiki:\n    tier: B\n    editability: ORG-WIDE\n")

    assert load_rules(rules_path)["wiki"].editability == "org-wide"


def test_load_rules_rejects_unknown_editability(tmp_path):
    rules_path = tmp_path / "trust.yaml"
    rules_path.write_text("origins:\n  wiki:\n    tier: B\n    editability: whoever\n")

    with pytest.raises(ValueError, match="unknown editability"):
        load_rules(rules_path)


def test_the_projects_own_trust_yaml_caps_the_wiki(tmp_path):
    # The committed rubric claims B for a wiki its own reason describes as
    # staff-editable. The cap is what stops that claim standing unchallenged.
    rules = load_rules(PROJECT_TRUST_YAML)

    tier, note = apply_editability_cap(
        rules["verified_internal_wiki"].tier,
        rules["verified_internal_wiki"].editability,
    )

    assert tier == TrustTier.C
    assert note is not None


# --- drift re-review (needs_review) ----------------------------------------

PROJECT_TRUST_YAML = Path(__file__).resolve().parents[1] / "trust.yaml"

NOW = datetime(2026, 9, 29, tzinfo=UTC)


def _origin(name="verified_internal_wiki", tier=TrustTier.B):
    return Origin(name, tier, "reason", 3)


def _previous(tier=TrustTier.B, *, overridden=False, days_ago=1, reviewed=True):
    return ReviewDecision(
        "verified_internal_wiki",
        tier,
        False,
        overridden,
        "reason",
        NOW - timedelta(days=days_ago) if reviewed else None,
        files={},
    )


def test_a_settled_origin_is_not_re_reviewed():
    assert needs_review(_origin(), _previous(), NOW) is None


def test_an_origin_with_no_prior_decision_is_new():
    assert needs_review(_origin(), None, NOW) == "new origin"


def test_a_decision_with_no_timestamp_is_re_reviewed():
    # Decisions written before reviewed_at existed cannot be aged, so they
    # get asked again rather than trusted indefinitely.
    assert needs_review(_origin(), _previous(reviewed=False), NOW) == "no review date on record"


def test_a_decision_older_than_the_max_age_is_re_reviewed():
    reason = needs_review(_origin(), _previous(days_ago=91), NOW)

    assert reason is not None
    assert "91 days ago" in reason


def test_a_decision_exactly_at_the_max_age_still_stands():
    assert needs_review(_origin(), _previous(days_ago=REVIEW_MAX_AGE_DAYS), NOW) is None


def test_a_changed_computed_tier_triggers_re_review():
    reason = needs_review(_origin(tier=TrustTier.C), _previous(tier=TrustTier.B), NOW)

    assert reason == "computed tier changed B -> C"


def test_a_changed_editability_triggers_re_review():
    # The tier in trust.yaml did not move, but who can edit the source did --
    # which is exactly the kind of change a tier comparison cannot see.
    previous = _previous()
    previous.editability = "authenticated-few"
    origin = Origin("verified_internal_wiki", TrustTier.B, "reason", 3, "org-wide")

    assert needs_review(origin, previous, NOW) == (
        "editability changed authenticated-few -> org-wide"
    )


def test_newly_declared_editability_triggers_re_review():
    previous = _previous()
    origin = Origin("verified_internal_wiki", TrustTier.B, "reason", 3, "anonymous")

    assert needs_review(origin, previous, NOW) == "editability changed undeclared -> anonymous"


def test_an_override_is_not_treated_as_drift():
    # The reviewer deliberately set A against a computed B. Asking again every
    # run would be nagging about a decision they already made.
    previous = _previous(tier=TrustTier.A, overridden=True)

    assert needs_review(_origin(tier=TrustTier.B), previous, NOW) is None


def test_an_overridden_decision_still_ages_out():
    previous = _previous(tier=TrustTier.A, overridden=True, days_ago=120)

    assert needs_review(_origin(tier=TrustTier.B), previous, NOW) is not None


def test_reviewed_at_survives_a_save_load_round_trip(tmp_path):
    stamped = ReviewDecision("verified_internal_wiki", TrustTier.B, False, False, "r", NOW)

    save_decisions([stamped], tmp_path)
    loaded = load_latest_decisions(tmp_path)["verified_internal_wiki"]

    assert loaded.reviewed_at == NOW


def test_run_review_gate_stamps_a_review_time(monkeypatch):
    _answer(monkeypatch, "")

    decisions = run_review_gate([_origin()])

    assert decisions[0].reviewed_at is not None
    assert decisions[0].reviewed_at.tzinfo is not None


# --- review gate behaviour -------------------------------------------------

def _answer(monkeypatch, *responses):
    replies = iter(responses)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))


def test_exclude_keeps_the_computed_tier_on_the_record(monkeypatch):
    # The audit trail should still say what the tier was when it was excluded.
    _answer(monkeypatch, "x")

    decisions = run_review_gate([Origin("public_inbox_unverified", TrustTier.D, "reason", 1)])

    assert decisions[0].tier == TrustTier.D


def test_override_records_both_the_old_and_new_tier_in_the_reason(monkeypatch):
    _answer(monkeypatch, "a")

    decisions = run_review_gate([Origin("verified_internal_wiki", TrustTier.B, "reason", 3)])

    assert "B" in decisions[0].reason
    assert "A" in decisions[0].reason


@pytest.mark.parametrize("keystroke", ["x", "X", " x ", "a", "A", " a "])
def test_review_gate_input_is_case_and_whitespace_insensitive(monkeypatch, keystroke):
    _answer(monkeypatch, keystroke)

    decisions = run_review_gate([Origin("verified_internal_wiki", TrustTier.B, "reason", 3)])

    assert decisions[0].excluded or decisions[0].overridden


def test_each_origin_is_reviewed_independently(monkeypatch):
    _answer(monkeypatch, "", "x", "a")

    decisions = run_review_gate(
        [
            Origin("first", TrustTier.B, "reason", 1),
            Origin("second", TrustTier.B, "reason", 1),
            Origin("third", TrustTier.B, "reason", 1),
        ]
    )

    assert [d.origin_name for d in decisions] == ["first", "second", "third"]
    assert decisions[0].excluded is False and decisions[0].overridden is False
    assert decisions[1].excluded is True
    assert decisions[2].tier == TrustTier.A


def test_no_origins_means_no_decisions_and_no_prompting(monkeypatch):
    def explode(_prompt=""):
        raise AssertionError("should not prompt when there is nothing to review")

    monkeypatch.setattr("builtins.input", explode)

    assert run_review_gate([]) == []


def test_review_screen_shows_why_a_tier_was_capped(monkeypatch, capsys):
    # A reviewer pressing Enter on a demoted tier should be able to see that
    # it was demoted, and why, without reading trust.yaml.
    _answer(monkeypatch, "")
    origin = Origin(
        "verified_internal_wiki", TrustTier.C, "reason", 3, "org-wide", "B -> C: editable by org-wide"
    )

    run_review_gate([origin])

    assert "Capped: B -> C: editable by org-wide" in capsys.readouterr().out


def test_review_screen_omits_the_cap_line_when_nothing_was_capped(monkeypatch, capsys):
    _answer(monkeypatch, "")

    run_review_gate([Origin("verified_internal_wiki", TrustTier.B, "reason", 3)])

    assert "Capped" not in capsys.readouterr().out


def test_review_screen_shows_the_tier_letter_and_its_meaning(monkeypatch, capsys):
    _answer(monkeypatch, "")

    run_review_gate([Origin("verified_internal_wiki", TrustTier.B, "reason", 3)])
    printed = capsys.readouterr().out

    assert "verified_internal_wiki" in printed
    assert "B" in printed
    assert "Internal Documents" in printed
    assert "3" in printed


# --- a malformed decisions file must stop the run --------------------------
#
# This file decides which origins are trusted and which are excluded, so
# "unparseable" has to mean "stop", not "default". Our loader used to coerce:
# bool(d["excluded"]) turned null into False, so an origin a reviewer had
# EXCLUDED was indexed while the record still said excluded. Each case below
# was silently accepted before the validation was ported from the fork.

def _write_signed(tmp_path, payload):
    """Sign a hand-written payload with the test key, as save_decisions would.

    Unsigned files are no longer trusted at all (they read as "nothing
    reviewed yet"), so without a signature these tests would never reach the
    validation they exist to check.
    """
    payload = {**payload, "previous_signature": None}
    payload["signature"] = sign(payload, load_key(create=True))
    (tmp_path / "decisions_v1.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _write_decisions(tmp_path, decisions, decided_at="2026-10-12T00:00:00+00:00"):
    return _write_signed(tmp_path, {"version": 1, "decided_at": decided_at, "decisions": decisions})


_VALID = {
    "origin_name": "wiki", "tier": "B",
    "excluded": False, "overridden": False, "reason": "accepted",
}


def test_a_valid_decisions_file_still_loads(tmp_path):
    loaded = load_latest_decisions(_write_decisions(tmp_path, [_VALID]))
    assert loaded["wiki"].tier is TrustTier.B
    assert loaded["wiki"].excluded is False


@pytest.mark.parametrize(
    ("label", "entry"),
    [
        # The fail-open that mattered: null/0/"" all became False, so the
        # exclude button stopped working while the audit trail still claimed it.
        ("excluded null", {**_VALID, "excluded": None}),
        ("excluded 0", {**_VALID, "excluded": 0}),
        ("excluded empty string", {**_VALID, "excluded": ""}),
        ("overridden null", {**_VALID, "overridden": None}),
        # A tier we cannot parse must never fall back to a default, in either
        # direction: D would hide a corrupt file, A would hand out trust.
        ("unknown tier", {**_VALID, "tier": "Z"}),
        ("lowercase tier", {**_VALID, "tier": "b"}),
        ("numeric tier", {**_VALID, "tier": 3}),
        # A naive timestamp used to surface as a TypeError deep in the decay
        # comparison, far from the file that caused it.
        ("naive reviewed_at", {**_VALID, "reviewed_at": "2026-06-01T10:00:00"}),
        ("unparseable reviewed_at", {**_VALID, "reviewed_at": "yesterday"}),
    ],
)
def test_a_malformed_decision_raises_rather_than_coercing(tmp_path, label, entry):
    with pytest.raises(DecisionsError):
        load_latest_decisions(_write_decisions(tmp_path, [entry]))


def test_a_missing_tier_key_reports_the_tier_not_a_keyerror(tmp_path):
    """The handler must not index the key whose absence it is handling.

    `except KeyError: raise DecisionsError(f"... {d['tier']} ...")` raises a
    second KeyError when "tier" is the missing key, and the caller sees that
    instead of the message. The fork has this; we use d.get().
    """
    entry = {k: v for k, v in _VALID.items() if k != "tier"}

    with pytest.raises(DecisionsError, match="unknown tier"):
        load_latest_decisions(_write_decisions(tmp_path, [entry]))


def test_a_decisions_file_that_is_not_json_raises(tmp_path):
    (tmp_path / "decisions_v1.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(DecisionsError, match="not valid JSON"):
        load_latest_decisions(tmp_path)


def test_an_absent_reviewed_at_is_not_an_error(tmp_path):
    """_parse_date(None) returns None rather than raising.

    Reachable only when the file has no decided_at either, which
    save_decisions never produces -- but a hand-written or third-party file
    can, and "no date" is a legitimate state meaning "re-review this".
    """
    _write_signed(tmp_path, {"version": 1, "decisions": [_VALID]})

    loaded = load_latest_decisions(tmp_path)

    assert loaded["wiki"].reviewed_at is None
    assert needs_review(
        Origin("wiki", TrustTier.B, "r", 1), loaded["wiki"], datetime.now(UTC)
    ) == "no review date on record"


# --- the trust.yaml hash in the signed trail --------------------------------

def test_the_rules_hash_is_saved_signed_and_read_back(tmp_path):
    from provenance_gated_rag.registry import load_latest_rules_hash

    save_decisions([ReviewDecision("wiki", TrustTier.B, False, False, "r")], tmp_path, "ab" * 32)

    assert load_latest_rules_hash(tmp_path) == "ab" * 32


def test_editing_the_recorded_rules_hash_breaks_the_signature(tmp_path):
    from provenance_gated_rag.registry import load_latest_rules_hash

    path = save_decisions([ReviewDecision("wiki", TrustTier.B, False, False, "r")], tmp_path, "ab" * 32)
    payload = json.loads(path.read_text())
    payload["trust_rules_sha256"] = "cd" * 32
    path.write_text(json.dumps(payload))

    with pytest.raises(DecisionsError, match="signature"):
        load_latest_rules_hash(tmp_path)


def test_a_trail_saved_without_a_rules_hash_reports_none(tmp_path):
    from provenance_gated_rag.registry import load_latest_rules_hash

    save_decisions([ReviewDecision("wiki", TrustTier.B, False, False, "r")], tmp_path)

    assert load_latest_rules_hash(tmp_path) is None


def test_with_unchanged_rules_a_new_high_tier_origin_is_accepted_on_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    decisions = run_review_gate([Origin("wiki", TrustTier.B, "reason", 1)], rules_changed=False)

    assert decisions[0].tier == TrustTier.B and not decisions[0].overridden


def test_with_changed_rules_enter_keeps_a_new_origin_at_d(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    decisions = run_review_gate([Origin("wiki", TrustTier.B, "reason", 1)], rules_changed=True)

    assert decisions[0].tier == TrustTier.D and decisions[0].overridden


def test_with_changed_rules_a_new_d_origin_is_not_asked_twice(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    decisions = run_review_gate([Origin("inbox", TrustTier.D, "reason", 1)], rules_changed=True)

    assert decisions[0].tier == TrustTier.D and not decisions[0].overridden
