"""The entailment check in M5, with a fake model: what it is asked, and what it declines.

The real model's behaviour is in test_nli_model.py, which runs only where
torch and the pinned weights are installed.
"""
from llama_index.core.schema import NodeWithScore, TextNode

from provenance_gated_rag.grounding import (
    CONTRADICTED,
    GroundedQueryEngine,
    answer_with_grounding,
    check_grounding,
)
from provenance_gated_rag.nli import DEFAULT_MODEL, EntailmentJudge, Judgement

ENTAILS = Judgement(0.98, 0.01, 0.01)
NEUTRAL = Judgement(0.05, 0.90, 0.05)
CONTRADICTS = Judgement(0.0, 0.01, 0.99)


class FakeJudge:
    """Contradicts when the claim has a word the premise opposes; records every pair."""

    def __init__(self, opposites=(("opposes", "leads"), ("2016", "2015")), neutral=("roughly",)):
        self.opposites, self.neutral, self.asked = opposites, neutral, []

    def __call__(self, pairs):
        self.asked += pairs
        out = []
        for premise, claim in pairs:
            if any(a in claim and b in premise for a, b in self.opposites):
                out.append(CONTRADICTS)
            elif any(w in claim for w in self.neutral):
                out.append(NEUTRAL)
            else:
                out.append(ENTAILS)
        return out


def _node(text, tier="B"):
    return NodeWithScore(node=TextNode(text=text, metadata={"origin": "wiki", "trust_tier": tier}), score=1.0)


SOURCE = _node("Elena Cho has served as Chief Executive Officer since the company was founded in 2015 "
               "and leads its engineering strategy.")
# Shares almost every word with SOURCE and has no "not": the lexical checks pass it.
FLIPPED = ("Elena Cho has served as Chief Executive Officer since the company was founded in 2015 "
           "and opposes its engineering strategy [1].")


# --- the judge itself ------------------------------------------------------

def test_constructing_a_judge_loads_nothing():
    loads = []
    EntailmentJudge(_loader=lambda: loads.append(1))
    assert loads == []


def test_the_model_is_loaded_once_on_first_use_and_pairs_are_batched():
    loads, batches = [], []

    def loader():
        loads.append(1)
        return lambda pairs: batches.append(len(pairs)) or [ENTAILS] * len(pairs)

    judge = EntailmentJudge(batch_size=2, _loader=loader)
    assert judge([]) == [] and loads == []
    assert len(judge([("p", "c")] * 5)) == 5
    judge([("p", "c")])

    assert loads == [1]
    assert batches == [2, 2, 1, 1]


def test_the_default_model_is_pinned():
    assert EntailmentJudge().describe.startswith(DEFAULT_MODEL + "@")
    assert "unpinned" not in EntailmentJudge().describe


# --- in check_grounding ----------------------------------------------------

def test_the_lexical_checks_alone_pass_a_flipped_claim():
    # The gap this closes, kept visible.
    assert check_grounding(FLIPPED, [SOURCE]) == []


def test_a_claim_its_source_contradicts_is_declined():
    problems = check_grounding(FLIPPED, [SOURCE], judge=FakeJudge())

    assert len(problems) == 1
    assert "contradicts the source it relies on" in problems[0]


def test_a_neutral_reading_is_not_declined():
    # A fair paraphrase often reads as neutral; declining it would decline good answers.
    answer = "Elena Cho has served as Chief Executive Officer since 2015, roughly [1]."
    assert check_grounding(answer, [SOURCE], judge=FakeJudge()) == []


def test_one_passage_that_entails_the_claim_clears_it():
    # Both sentences hold every name and number in the claim, so both are
    # judged; the fake reads the first as contradicting it.
    source = _node("In 2016 Raj Patel, then an engineer, was promoted to CTO. Raj Patel became CTO in 2016.")
    judge = FakeJudge(opposites=(("became CTO", "promoted"),))

    assert check_grounding("Raj Patel became CTO in 2016 [1].", [source], judge=judge) == []
    assert len(judge.asked) == 2


def test_another_trusted_source_contradicting_the_claim_declines_it():
    other = _node("Elena Cho leads the engineering strategy, and has since 2015.")

    problems = check_grounding(FLIPPED, [SOURCE, other], judge=FakeJudge(opposites=(("opposes", "leads the"),)))

    assert any("is contradicted by trusted source [2]" in p for p in problems)


def test_the_claim_is_judged_without_its_citation():
    judge = FakeJudge()
    check_grounding(FLIPPED, [SOURCE], judge=judge)

    assert all("[1]" not in claim for _, claim in judge.asked)


def test_the_judge_is_not_asked_about_a_sentence_already_declined():
    judge = FakeJudge()
    check_grounding("Marcus Webb is the Chief Executive Officer [1].", [SOURCE], judge=judge)

    assert judge.asked == []


def test_the_judge_is_not_asked_about_talk_about_the_sources():
    judge = FakeJudge()
    check_grounding("This is consistent across sources [1].", [SOURCE], judge=judge)

    assert judge.asked == []


def test_the_threshold_is_a_high_bar():
    assert CONTRADICTED >= 0.9


# --- wired through ---------------------------------------------------------

class FakeLLM:
    def complete(self, _prompt):
        return type("R", (), {"text": FLIPPED})()


def test_answer_with_grounding_passes_the_judge_on():
    result = answer_with_grounding("What does Elena Cho do?", [SOURCE], FakeLLM(), judge=FakeJudge())

    assert result.abstained
    assert any("contradicts" in r for r in result.reasons)


def test_the_engine_passes_the_judge_on():
    index = type("I", (), {"as_retriever": lambda _self, similarity_top_k: type(
        "R", (), {"retrieve": lambda _s, _q: [SOURCE]})()})()

    result = GroundedQueryEngine(index, FakeLLM(), judge=FakeJudge()).query("What does Elena Cho do?")

    assert result.abstained
