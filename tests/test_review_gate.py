import pytest

from pgrag.cli import run_review_gate
from pgrag.core.registry import Origin, TrustTier


def _origin(name="wiki", tier=TrustTier.B, count=3):
    return Origin(name, tier, "reason", count)


def _answer(monkeypatch, *responses):
    """Feed canned keystrokes to the gate instead of a real keyboard."""
    replies = iter(responses)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))


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
