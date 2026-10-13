"""The review gate (cli.run_review_gate): keystrokes, overrides, and re-asking (decay)."""
from datetime import datetime, timedelta, timezone

import pytest

from pgrag.cli import run_review_gate
from pgrag.core.registry import REVIEW_MAX_AGE_DAYS, Origin, ReviewDecision, TrustTier

NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)


def _origin(name="wiki", tier=TrustTier.B, count=3, editability=None):
    return Origin(name, tier, "reason", count, editability)


def _answer(monkeypatch, *responses):
    """Feed canned keystrokes to the gate instead of a real keyboard."""
    replies = iter(responses)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))


def _count_questions(monkeypatch, reply=""):
    """Answer every question with `reply`, and return a list that grows per question."""
    asked = []
    monkeypatch.setattr("builtins.input", lambda _prompt="": asked.append(1) or reply)
    return asked


def _previous(name="wiki", days_ago=10, tier=TrustTier.B, editability=None):
    return ReviewDecision(name, tier, False, False, "reason",
                          editability=editability, reviewed_at=NOW - timedelta(days=days_ago),
                          files={})


# --- keystrokes -----------------------------------------------------------

def test_enter_accepts_the_computed_tier(monkeypatch):
    _answer(monkeypatch, "")

    decisions = run_review_gate([_origin()])

    assert len(decisions) == 1
    assert decisions[0].tier == TrustTier.B
    assert decisions[0].excluded is False
    assert decisions[0].overridden is False


def test_x_excludes_the_origin(monkeypatch):
    _answer(monkeypatch, "x")

    decisions = run_review_gate([_origin("public_inbox", TrustTier.D, 1)])

    assert decisions[0].excluded is True
    assert decisions[0].overridden is False


def test_exclude_keeps_the_computed_tier_on_the_record(monkeypatch):
    # The audit trail should still say what the tier *was* when it was excluded.
    _answer(monkeypatch, "x")

    decisions = run_review_gate([_origin("public_inbox", TrustTier.D, 1)])

    assert decisions[0].tier == TrustTier.D


def test_a_tier_letter_overrides(monkeypatch):
    _answer(monkeypatch, "a")

    decisions = run_review_gate([_origin()])

    assert decisions[0].tier == TrustTier.A
    assert decisions[0].overridden is True
    assert decisions[0].excluded is False


def test_override_records_both_the_old_and_new_tier_in_the_reason(monkeypatch):
    _answer(monkeypatch, "a")

    decisions = run_review_gate([_origin()])

    assert "B" in decisions[0].reason
    assert "A" in decisions[0].reason


@pytest.mark.parametrize("keystroke", ["x", "X", " x ", "a", "A", " a "])
def test_input_is_case_and_whitespace_insensitive(monkeypatch, keystroke):
    _answer(monkeypatch, keystroke)

    decisions = run_review_gate([_origin()])

    assert decisions[0].excluded or decisions[0].overridden


def test_invalid_input_reprompts_until_valid(monkeypatch):
    _answer(monkeypatch, "not-a-real-choice", "z9", "")

    decisions = run_review_gate([_origin()])

    assert decisions[0].excluded is False
    assert decisions[0].overridden is False


def test_each_origin_is_reviewed_independently(monkeypatch):
    _answer(monkeypatch, "", "x", "a")

    decisions = run_review_gate(
        [_origin("first"), _origin("second"), _origin("third")]
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


def test_review_screen_shows_the_tier_letter_and_its_meaning(monkeypatch, capsys):
    _answer(monkeypatch, "")

    run_review_gate([_origin()])
    printed = capsys.readouterr().out

    assert "wiki" in printed
    assert "B" in printed
    assert "Internal Documents" in printed
    assert "3" in printed


# --- editability and the cap ----------------------------------------------

def test_a_decision_carries_the_origins_editability_and_a_review_time(monkeypatch):
    _answer(monkeypatch, "")

    [d] = run_review_gate([_origin(editability="org-wide")], now=NOW)

    assert d.editability == "org-wide"
    assert d.reviewed_at == NOW


def test_accepting_a_capped_origin_records_the_capped_tier(monkeypatch):
    _answer(monkeypatch, "")
    capped = Origin("wiki", TrustTier.C, "reason", 3, "org-wide", "A -> C: editable by org-wide")

    [d] = run_review_gate([capped])

    assert d.tier == TrustTier.C
    assert d.overridden is False


def test_a_reviewer_can_override_the_cap(monkeypatch):
    _answer(monkeypatch, "a")
    capped = Origin("wiki", TrustTier.C, "reason", 3, "org-wide", "A -> C: editable by org-wide")

    [d] = run_review_gate([capped])

    assert d.tier == TrustTier.A
    assert d.overridden is True


# --- decay v1: re-asking ----------------------------------------------------

def test_a_stale_origin_is_asked_again_and_its_clock_restarts(monkeypatch):
    asked = _count_questions(monkeypatch)

    [d] = run_review_gate([_origin()], {"wiki": _previous(days_ago=REVIEW_MAX_AGE_DAYS + 1)}, now=NOW)

    assert len(asked) == 1
    assert d.reviewed_at == NOW


def test_a_fresh_origin_is_not_asked_and_keeps_its_old_date(monkeypatch):
    def must_not_ask(_prompt=""):
        pytest.fail("the gate asked about an origin reviewed 10 days ago")

    monkeypatch.setattr("builtins.input", must_not_ask)
    previous = _previous(days_ago=10)

    [d] = run_review_gate([_origin()], {"wiki": previous}, now=NOW)

    assert d is previous                                  # reused as-is
    assert d.reviewed_at == NOW - timedelta(days=10)      # so it can still go stale later


def test_a_new_origin_is_asked(monkeypatch):
    asked = _count_questions(monkeypatch)

    run_review_gate([_origin()], {}, now=NOW)

    assert len(asked) == 1


def test_a_change_of_editability_in_trust_yaml_is_asked_again(monkeypatch):
    asked = _count_questions(monkeypatch)

    run_review_gate([_origin(editability="anonymous")], {"wiki": _previous(editability="authenticated-few")}, now=NOW)

    assert len(asked) == 1


def test_only_the_stale_origin_in_a_mixed_list_is_asked(monkeypatch):
    asked = _count_questions(monkeypatch)
    origins = [_origin("fresh"), _origin("stale")]
    previous = {"fresh": _previous("fresh", days_ago=5), "stale": _previous("stale", days_ago=200)}

    decisions = run_review_gate(origins, previous, now=NOW)

    assert len(asked) == 1
    assert [d.origin_name for d in decisions] == ["fresh", "stale"]
    assert decisions[1].reviewed_at == NOW


def test_the_screen_says_why_an_origin_is_being_asked_again(monkeypatch, capsys):
    _answer(monkeypatch, "")

    run_review_gate([_origin()], {"wiki": _previous(days_ago=120)}, now=NOW)

    assert "120 days" in capsys.readouterr().out


# --- the reviewer sees, and approves, the exact file list -----------------

def _origin_with_files(files, name="wiki"):
    return Origin(name, TrustTier.B, "reason", len(files), None, files=files)


def test_a_decision_records_the_file_list_the_card_showed(monkeypatch):
    _answer(monkeypatch, "")
    files = {"a.txt": "1" * 64, "b.txt": "2" * 64}

    decisions = run_review_gate([_origin_with_files(files)], now=NOW)

    assert decisions[0].files == files


def test_a_new_origins_card_lists_its_files(monkeypatch, capsys):
    _answer(monkeypatch, "")

    run_review_gate([_origin_with_files({"a.txt": "1" * 64, "notes/b.md": "2" * 64})], now=NOW)

    out = capsys.readouterr().out
    assert "a.txt" in out and "notes/b.md" in out


def test_a_file_planted_in_an_approved_origin_is_asked_about_and_shown(monkeypatch, capsys):
    asked = _count_questions(monkeypatch)
    before = {"a.txt": "1" * 64}
    previous = {"wiki": ReviewDecision("wiki", TrustTier.B, False, False, "reason",
                                       reviewed_at=NOW - timedelta(days=5), files=before)}

    decisions = run_review_gate(
        [_origin_with_files({**before, "planted.txt": "9" * 64})], previous, now=NOW)

    out = capsys.readouterr().out
    assert len(asked) == 1
    assert "files changed since review: 1 added" in out
    assert "+ planted.txt" in out
    assert "planted.txt" in decisions[0].files


def test_a_long_file_list_is_cut_short_on_the_card(monkeypatch, capsys):
    _answer(monkeypatch, "")
    files = {f"doc_{i:02}.txt": "1" * 64 for i in range(40)}

    run_review_gate([_origin_with_files(files)], now=NOW)

    out = capsys.readouterr().out
    assert "doc_00.txt" in out
    assert "and 25 more" in out
