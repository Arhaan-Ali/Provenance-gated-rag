"""Facts read from trusted sources, and declining when two of them disagree."""
import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from provenance_gated_rag.facts import (
    Fact,
    answer_conflicts,
    conflicts,
    describe,
    extract_facts,
    facts_from_nodes,
)
from provenance_gated_rag.grounding import answer_with_grounding
from provenance_gated_rag.registry import TrustTier
from provenance_gated_rag.sanitizer import REDACTION_MARKER


def _facts(text):
    return [(f.key, f.value) for f in extract_facts(text, "x")]


# --- reading facts -----------------------------------------------------------

@pytest.mark.parametrize(("text", "fact"), [
    ("Elena Cho is the Chief Executive Officer.", ("CEO", "Elena Cho")),
    ("Elena Cho has served as Chief Executive Officer since 2015.", ("CEO", "Elena Cho")),
    ("Sofia Marchetti, Chief Financial Officer, joined in 2019.", ("CFO", "Sofia Marchetti")),
    ("The CEO of the company is Maria Lopez, who has led it since 2019.", ("CEO", "Maria Lopez")),
    ("Raj Patel is our CTO.", ("CTO", "Raj Patel")),
    ("Ana Ruiz is the Head of Sales.", ("head of sales", "Ana Ruiz")),
    ("Our headquarters is in Berlin, Germany.", ("headquarters", "Berlin")),
    ("A manufacturer founded in 2015 and headquartered in Austin, Texas.", ("headquarters", "Austin")),
])
def test_each_kind_of_fact_is_read(text, fact):
    assert fact in _facts(text)


# Each of these matches a fact pattern; only the negation or the past tense
# keeps it from being read as a current fact.
@pytest.mark.parametrize("text", [
    "Marcus Webb, Chief Executive Officer, has not been in office since 2014.",
    "Marcus Webb, Chief Executive Officer, is no longer with the company.",
    "Marcus Webb, Chief Executive Officer until 2014, left the company.",
    "Marcus Webb, Chief Executive Officer, previously ran the Denver office.",
    "The company was headquartered in Denver until 2020.",
])
def test_a_negated_or_past_statement_is_not_a_current_fact(text):
    assert _facts(text) == []


def test_a_founding_year_told_in_the_past_is_still_read():
    assert _facts("The company was founded in 2015.") == [("founded", "2015")]


def test_a_heading_is_not_read_into_the_name_below_it():
    text = "Meridian Robotics — Leadership Team\n\nElena Cho is the Chief Executive Officer."
    assert _facts(text) == [("CEO", "Elena Cho")]


def test_a_hard_wrapped_sentence_is_still_one_sentence():
    text = "A robotics manufacturer founded in 2015 and headquartered\nin Austin, Texas."
    assert ("headquarters", "Austin") in _facts(text)


def test_chief_executive_officer_and_ceo_are_the_same_fact():
    a = extract_facts("Elena Cho is the Chief Executive Officer.", "a")
    b = extract_facts("Elena Cho is the CEO.", "b")
    assert conflicts(a + b) == {}


# --- finding a disagreement ---------------------------------------------------

def test_two_values_for_one_key_is_a_conflict_with_both_sources():
    facts = extract_facts("Elena Cho is the CEO.", "wiki/leadership.txt") + extract_facts(
        "Marcus Webb is the Chief Executive Officer.", "partner/intro.txt")

    found = conflicts(facts)

    assert list(found) == ["CEO"]
    assert describe("CEO", found["CEO"]) == (
        "Trusted sources disagree on the CEO: 'Elena Cho' in wiki/leadership.txt; "
        "'Marcus Webb' in partner/intro.txt.")


def test_the_same_value_in_different_case_is_not_a_conflict():
    facts = [Fact("headquarters", "Berlin", "a", "s"), Fact("headquarters", "berlin", "b", "s")]
    assert conflicts(facts) == {}


DISPUTED = (extract_facts("Elena Cho is the CEO.", "wiki")
            + extract_facts("Marcus Webb is the Chief Executive Officer.", "partner"))


@pytest.mark.parametrize("answer", [
    "Elena Cho leads the company [1].",
    "The CEO is unknown.",
    "The Chief Executive Officer is listed in the wiki.",
])
def test_an_answer_that_touches_the_disputed_fact_is_flagged(answer):
    assert answer_conflicts(answer, DISPUTED)


def test_an_answer_about_something_else_is_not():
    assert answer_conflicts("Raj Patel is the CTO [1].", DISPUTED) == []


# --- which sources count -----------------------------------------------------

def _node(text, tier, origin="o", source_file="f.txt"):
    return NodeWithScore(node=TextNode(text=text, metadata={
        "origin": origin, "source_file": source_file, "trust_tier": tier}), score=1.0)


def test_only_trusted_sources_give_facts():
    nodes = [_node("Elena Cho is the CEO.", "B", "wiki", "lead.txt"),
             _node("Marcus Webb is the CEO.", "D", "inbox", "press.txt")]

    assert [(f.value, f.where) for f in facts_from_nodes(nodes, TrustTier.C)] == [("Elena Cho", "wiki/lead.txt")]


def test_a_neutralised_source_gives_no_facts():
    assert facts_from_nodes([_node(REDACTION_MARKER, "B")], TrustTier.C) == []


# --- the decline -------------------------------------------------------------

class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, _prompt):
        return type("R", (), {"text": self.reply})()


LEAD = _node("Elena Cho is the Chief Executive Officer of Meridian Robotics.", "B", "wiki", "lead.txt")
PARTNER = _node("Marcus Webb is the Chief Executive Officer of Meridian Robotics.", "B", "partner", "intro.txt")


def test_two_retrieved_trusted_sources_that_disagree_decline_the_answer():
    result = answer_with_grounding("Who is the CEO?", [LEAD, PARTNER],
                                   FakeLLM("Elena Cho is the Chief Executive Officer of Meridian Robotics [1]."))

    assert result.abstained
    assert "disagree" in result.reasons[0]
    assert "'Marcus Webb' in partner/intro.txt" in result.reasons[1]


def test_a_disagreeing_source_that_was_not_retrieved_still_declines_it():
    beyond = extract_facts("Marcus Webb is the Chief Executive Officer.", "partner/intro.txt")
    result = answer_with_grounding("Who is the CEO?", [LEAD],
                                   FakeLLM("Elena Cho is the Chief Executive Officer of Meridian Robotics [1]."),
                                   facts=beyond)

    assert result.abstained and "partner/intro.txt" in result.reasons[1]


def test_an_untrusted_source_disagreeing_does_not_decline_a_grounded_answer():
    planted = _node("Marcus Webb is the Chief Executive Officer of Meridian Robotics.", "D", "inbox", "p.txt")
    result = answer_with_grounding("Who is the CEO?", [LEAD, planted],
                                   FakeLLM("Elena Cho is the Chief Executive Officer of Meridian Robotics [1]."))

    assert not result.abstained


def test_a_question_away_from_the_dispute_is_still_answered():
    cto = _node("Raj Patel is the Chief Technology Officer.", "B", "wiki", "lead.txt")
    result = answer_with_grounding("Who is the CTO?", [cto, LEAD, PARTNER],
                                   FakeLLM("Raj Patel is the Chief Technology Officer [1]."))

    assert not result.abstained


def test_ingestion_names_trusted_sources_that_disagree(tmp_path, capsys):
    from provenance_gated_rag.ingest import load_documents
    from provenance_gated_rag.registry import ReviewDecision

    for origin, name, text in (("wiki", "lead.txt", "Elena Cho is the Chief Executive Officer."),
                               ("partner", "intro.txt", "Marcus Webb is the CEO."),
                               ("inbox", "press.txt", "Viktor Hale is the CEO.")):
        (tmp_path / origin).mkdir()
        (tmp_path / origin / name).write_text(text)
    decisions = [ReviewDecision("wiki", TrustTier.B, False, False, "r"),
                 ReviewDecision("partner", TrustTier.B, False, False, "r"),
                 ReviewDecision("inbox", TrustTier.D, False, False, "r")]

    load_documents(tmp_path, decisions)

    out = capsys.readouterr().out
    assert ("Trusted sources disagree on the CEO: 'Elena Cho' in wiki/lead.txt; "
            "'Marcus Webb' in partner/intro.txt.") in out
    assert "Viktor Hale" not in out   # tier D is not a trusted source
