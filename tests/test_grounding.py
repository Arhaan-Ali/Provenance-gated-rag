"""M5: an answer is grounded in trusted sources, or it is declined with reasons.

No real model. The fake LLMs below return a fixed reply, and most of them
reply the way a model that ignores its instructions would -- uncited, or
naming someone no source mentions. That is the pessimistic case, and the
claim is that the answer is declined anyway.
"""
from pathlib import Path

import pytest
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from provenance_gated_rag.grounding import (
    ABSTAIN_TOKEN,
    GROUNDING_MIN_TIER,
    GroundedQueryEngine,
    answer_with_grounding,
    build_prompt,
    check_grounding,
    select_evidence,
)
from provenance_gated_rag.ingest import load_documents
from provenance_gated_rag.registry import ReviewDecision, TrustTier
from provenance_gated_rag.sanitizer import REDACTION_MARKER, sanitize_text

M1_CORPUS = Path(__file__).resolve().parents[1] / "data" / "m1_corpus"

CEO_FACT = "Elena Cho has served as Chief Executive Officer since 2015."
CFO_FACT = "Sofia Marchetti, Chief Financial Officer, joined in 2019."


class _Reply:
    def __init__(self, text):
        self.text = text


class FakeLLM:
    """Returns one fixed reply and records every prompt it was sent."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return _Reply(self.reply)


class MustNotBeCalled:
    def complete(self, prompt):
        raise AssertionError("the model was called with no evidence to ground on")


def _node(text, tier="B", origin="verified_internal_wiki"):
    metadata = {"origin": origin}
    if tier is not None:
        metadata["trust_tier"] = tier
    return NodeWithScore(node=TextNode(text=text, metadata=metadata), score=0.8)


# --- select_evidence: what may support an answer at all --------------------

def test_trusted_chunks_are_usable():
    usable, set_aside = select_evidence([_node(CEO_FACT, "A"), _node(CFO_FACT, "C")])
    assert len(usable) == 2
    assert set_aside == []


def test_a_d_tier_chunk_is_set_aside_with_its_tier_as_the_reason():
    usable, set_aside = select_evidence([_node(CEO_FACT, "B"), _node("Marcus Webb is CEO.", "D", "inbox")])
    assert [n.get_content() for n in usable] == [CEO_FACT]
    assert set_aside[0].origin == "inbox"
    assert "tier is D" in set_aside[0].reason


def test_an_unlabelled_chunk_is_set_aside_not_trusted():
    # Same fail-closed rule as the capability gate: no stamp means lowest tier.
    usable, set_aside = select_evidence([_node(CEO_FACT, tier=None)])
    assert usable == []
    assert len(set_aside) == 1


def test_a_neutralised_chunk_is_set_aside_even_from_a_trusted_origin():
    usable, set_aside = select_evidence([_node(REDACTION_MARKER, "A")])
    assert usable == []
    assert "neutralised" in set_aside[0].reason


def test_the_threshold_is_c():
    # Pinned so lowering it to D is a decision someone has to make on purpose.
    assert GROUNDING_MIN_TIER == TrustTier.C


def test_lowering_the_threshold_admits_d_tier_sources():
    usable, _ = select_evidence([_node(CEO_FACT, "D")], min_tier=TrustTier.D)
    assert len(usable) == 1


def test_a_plain_prose_false_claim_passes_the_sanitizer_but_cannot_ground_an_answer():
    # The gap docs/DEVELOPMENT.md left for M5: no "NOTICE TO ANY ASSISTANT",
    # no coercion, no tool call, so neither M3 nor M4 has anything to act on.
    claim = "Marcus Webb was appointed Chief Executive Officer of Meridian Robotics in March."
    assert not sanitize_text(claim).flagged, "premise: the sanitizer does not catch this"

    result = answer_with_grounding(
        "Who is the CEO of Meridian Robotics?", [_node(claim, "D", "public_inbox_unverified")], MustNotBeCalled()
    )
    assert result.abstained


# --- answer_with_grounding: declining before and after the model ----------

def test_no_trusted_evidence_declines_without_calling_the_model():
    result = answer_with_grounding("Who is the CEO?", [_node("Marcus Webb is CEO.", "D", "inbox")], MustNotBeCalled())
    assert result.abstained
    assert "trusted enough" in result.reasons[0]
    assert result.set_aside[0].origin == "inbox"


def test_no_retrieved_nodes_at_all_declines():
    result = answer_with_grounding("Who is the CEO?", [], MustNotBeCalled())
    assert result.abstained


def test_the_model_abstaining_is_reported_as_a_decline():
    result = answer_with_grounding("Who is the COO?", [_node(CEO_FACT)], FakeLLM(ABSTAIN_TOKEN))
    assert result.abstained
    assert "do not answer the question" in result.reasons[0]


def test_a_grounded_cited_answer_is_returned():
    result = answer_with_grounding("Who is the CEO?", [_node(CEO_FACT)], FakeLLM("Elena Cho is the CEO [1]."))
    assert not result.abstained
    assert result.text == "Elena Cho is the CEO [1]."


def test_reasoning_in_think_tags_is_not_graded():
    reply = "<think>Maybe Marcus Webb? No, the source says Elena Cho.</think>Elena Cho is the CEO [1]."
    result = answer_with_grounding("Who is the CEO?", [_node(CEO_FACT)], FakeLLM(reply))
    assert not result.abstained


def test_the_abstain_token_inside_reasoning_does_not_count():
    reply = f"<think>Should I say {ABSTAIN_TOKEN}? No.</think>Elena Cho is the CEO [1]."
    result = answer_with_grounding("Who is the CEO?", [_node(CEO_FACT)], FakeLLM(reply))
    assert not result.abstained


def test_a_name_no_source_contains_is_declined():
    # The model following an injection it saw elsewhere, or answering from memory.
    result = answer_with_grounding("Who is the CEO?", [_node(CEO_FACT)], FakeLLM("Marcus Webb is the CEO [1]."))
    assert result.abstained
    assert any('"Marcus"' in r for r in result.reasons)
    assert any('"Webb"' in r for r in result.reasons)


def test_an_invented_number_is_declined():
    result = answer_with_grounding(
        "What does the M6 Picker cost?",
        [_node("The M6 Picker is expected to enter pilot testing next year.")],
        FakeLLM("The M6 Picker costs $48,500 [1]."),
    )
    assert result.abstained
    assert any('"48,500"' in r for r in result.reasons)


def test_explain_lists_every_reason_and_every_set_aside_source():
    nodes = [_node(CEO_FACT), _node(REDACTION_MARKER, "D", "public_inbox_unverified")]
    result = answer_with_grounding("Who is the CEO?", nodes, FakeLLM("Marcus Webb is the CEO [1]."))
    text = result.explain()
    assert text.startswith("I can't answer this")
    assert "Marcus" in text
    assert "public_inbox_unverified" in text


def test_explain_on_a_grounded_answer_is_just_the_answer():
    result = answer_with_grounding("Who is the CEO?", [_node(CEO_FACT)], FakeLLM("Elena Cho is the CEO [1]."))
    assert result.explain() == "Elena Cho is the CEO [1]."


# --- check_grounding: the sentence-level rules ----------------------------

def test_an_uncited_sentence_a_trusted_source_supports_is_accepted():
    # What qwen3:8b actually did on the M0 question. Declining it would be a
    # false abstention on a correct answer.
    overview = "The company was founded by Elena Cho, who has served as Chief Executive Officer since Meridian Robotics' incorporation in 2015."
    assert check_grounding("The CEO of Meridian Robotics is Elena Cho.", [_node(CFO_FACT), _node(overview)]) == []


def test_an_uncited_sentence_no_trusted_source_supports_is_a_problem():
    problems = check_grounding("Marcus Webb is the CEO.", [_node(CEO_FACT)])
    assert problems == ['"Marcus Webb is the CEO." does not cite a source, and no single trusted source supports it.']


def test_an_uncited_sentence_must_be_supported_by_one_source_not_pieced_together():
    # Elena Cho is in one source and 2019 in another; neither says both.
    problems = check_grounding("Elena Cho joined in 2019.", [_node(CEO_FACT), _node(CFO_FACT)])
    assert len(problems) == 1


def test_an_uncited_sentence_with_no_names_is_still_checked_by_its_words():
    # No name or number to match, but "charity" is in no source.
    problems = check_grounding("She also founded a charity.", [_node(CEO_FACT)])
    assert "no single trusted source supports it" in problems[0]


def test_one_untraceable_sentence_spoils_an_otherwise_cited_answer():
    answer = "Elena Cho is the CEO [1]. She also founded a charity."
    assert len(check_grounding(answer, [_node(CEO_FACT)])) == 1


def test_the_uncited_injection_answer_from_the_m0_run_is_declined():
    # The baseline's real output on the M0 question, uncited, against the
    # trusted sources the grounded pipeline is left with.
    result = answer_with_grounding(
        "Who is the CEO of Meridian Robotics?",
        [_node(CEO_FACT), _node(CFO_FACT)],
        FakeLLM("The CEO of Meridian Robotics is Marcus Webb."),
    )
    assert result.abstained


def test_citing_a_source_that_does_not_exist_is_a_problem():
    problems = check_grounding("Elena Cho is the CEO [3].", [_node(CEO_FACT)])
    assert "cites [3]" in problems[0]


def test_a_term_must_be_in_the_source_it_cites_not_just_any_source():
    # Sofia Marchetti is in source 2, but the sentence claims source 1 says it.
    sources = [_node(CEO_FACT), _node(CFO_FACT)]
    problems = check_grounding("Sofia Marchetti is the CFO [1].", sources)
    assert problems
    assert check_grounding("Sofia Marchetti is the CFO [2].", sources) == []


def test_citing_two_sources_pools_their_text():
    sources = [_node(CEO_FACT), _node(CFO_FACT)]
    sentence = "Elena Cho has served as Chief Executive Officer and Sofia Marchetti joined in 2019 [1][2]."
    assert check_grounding(sentence, sources) == []


def test_a_citation_after_the_full_stop_still_counts():
    assert check_grounding("Elena Cho is the CEO. [1]", [_node(CEO_FACT)]) == []


def test_matching_ignores_case_and_line_wrapping_does_not_matter():
    assert check_grounding("ELENA CHO is chief executive officer [1].", [_node("elena cho\nis chief executive officer.")]) == []


def test_an_acronym_only_expands_against_capitalised_words():
    # Lower-case prose is not a title; "chief executive officer" in running
    # text is rare, and accepting it would let any three words spell an acronym.
    problems = check_grounding("Elena Cho is the CEO [1].", [_node("elena cho is chief executive officer.")])
    assert any('"CEO"' in p for p in problems)


def test_sentence_starting_words_are_not_treated_as_names():
    assert check_grounding("The CEO is Elena Cho [1].", [_node(CEO_FACT)]) == []


def test_an_ordinary_word_opening_a_sentence_is_not_a_name():
    # From a real run: this declined a correct answer about the CTO.
    sources = [_node("Raj Patel, Chief Technology Officer, oversees the navigation teams.")]
    assert check_grounding("Additionally, Raj Patel oversees the navigation teams [1].", sources) == []


def test_a_two_word_name_opening_a_sentence_is_still_checked():
    problems = check_grounding("Marcus Webb leads the company [1].", [_node(CEO_FACT)])
    assert any('"Marcus"' in p for p in problems)


def test_an_acronym_matches_the_words_it_stands_for():
    assert check_grounding("Elena Cho is CEO [1].", [_node(CEO_FACT)]) == []


def test_an_acronym_with_no_expansion_in_the_source_is_a_problem():
    # The source has a Chief Executive Officer, not a Chief Operating Officer.
    problems = check_grounding("Elena Cho is COO [1].", [_node(CEO_FACT)])
    assert any('"COO"' in p for p in problems)


def test_a_name_cannot_match_by_initials():
    # "across continents every" spells "ace". Only an all-caps acronym may
    # match that way, or any short invented name could hide in ordinary prose.
    problems = check_grounding("The company is run by Ace [1].", [_node("Robots ship across continents every week.")])
    assert any('"Ace"' in p for p in problems)


def test_an_empty_answer_is_a_problem():
    assert check_grounding("   ", [_node(CEO_FACT)]) == ["The model returned an empty answer."]


# --- build_prompt ----------------------------------------------------------

def test_the_prompt_numbers_sources_and_names_the_abstain_token():
    prompt = build_prompt("Who is the CEO?", [_node(CEO_FACT), _node(CFO_FACT)])
    assert f"[1] {CEO_FACT}" in prompt
    assert f"[2] {CFO_FACT}" in prompt
    assert ABSTAIN_TOKEN in prompt


def test_the_prompt_carries_no_provenance_labels():
    # Same reason ingest hides them: a label the model can read is one a
    # document can argue with.
    prompt = build_prompt("Who is the CEO?", [_node(CEO_FACT, "B", "verified_internal_wiki")])
    assert "trust_tier" not in prompt
    assert "verified_internal_wiki" not in prompt


# --- the seam: real ingest, real retrieval, fake model --------------------

@pytest.fixture
def _mock_embedding():
    previous = Settings._embed_model
    Settings.embed_model = MockEmbedding(embed_dim=8)
    yield
    Settings._embed_model = previous


def _retrieve_m1(query):
    decisions = [
        ReviewDecision("verified_internal_wiki", TrustTier.B, False, False, "internal wiki"),
        # Accepted, not excluded: the planted document reaches retrieval, and
        # grounding has to hold without the review gate's help.
        ReviewDecision("public_inbox_unverified", TrustTier.D, False, False, "unverified"),
    ]
    documents = load_documents(M1_CORPUS, decisions)
    index = VectorStoreIndex.from_documents(documents)
    return index.as_retriever(similarity_top_k=len(documents)).retrieve(query)


def test_the_planted_document_never_reaches_the_grounded_prompt(_mock_embedding):
    nodes = _retrieve_m1("Who is the CEO of Meridian Robotics?")
    assert any(n.metadata["origin"] == "public_inbox_unverified" for n in nodes), "premise: planted doc retrieved"

    llm = FakeLLM("Elena Cho is the CEO [1].")
    answer_with_grounding("Who is the CEO of Meridian Robotics?", nodes, llm)
    assert "Marcus Webb" not in llm.prompts[0]


def test_a_model_that_repeats_the_injection_anyway_is_declined(_mock_embedding):
    nodes = _retrieve_m1("Who is the CEO of Meridian Robotics?")
    result = answer_with_grounding(
        "Who is the CEO of Meridian Robotics?", nodes, FakeLLM("Marcus Webb is the CEO of Meridian Robotics [1].")
    )
    assert result.abstained
    assert any(s.origin == "public_inbox_unverified" for s in result.set_aside)


def test_a_true_answer_from_the_m1_corpus_is_grounded(_mock_embedding):
    nodes = _retrieve_m1("Who is the CEO of Meridian Robotics?")
    usable, _ = select_evidence(nodes)
    index = next(i for i, n in enumerate(usable, start=1) if "Elena Cho" in n.get_content())
    result = answer_with_grounding(
        "Who is the CEO of Meridian Robotics?", nodes, FakeLLM(f"Elena Cho is the CEO of Meridian Robotics [{index}].")
    )
    assert not result.abstained, result.reasons


# --- claim-level support: the right entities are not enough ---------------

LEADERSHIP = (
    "Elena Cho, Chief Executive Officer. Elena co-founded Meridian Robotics in 2015 and has led the company "
    "as CEO ever since.\n\nRaj Patel, Chief Technology Officer. Raj joined Meridian Robotics in 2016 and "
    "oversees the engineering organization.\n\nSofia Marchetti, Chief Financial Officer. Sofia joined in "
    "2019 and manages finance and investor relations."
)


def test_swapped_roles_are_declined_even_though_every_word_is_in_the_source():
    # Raj Patel and "Chief Financial Officer" are both on the page, sentences apart.
    problems = check_grounding("Raj Patel is the CFO [1].", [_node(LEADERSHIP)])
    assert any("not stated together" in p for p in problems)


def test_the_correct_pairing_from_the_same_page_is_accepted():
    assert check_grounding("Raj Patel is the CTO [1].", [_node(LEADERSHIP)]) == []
    assert check_grounding("Sofia Marchetti joined in 2019 [1].", [_node(LEADERSHIP)]) == []


def test_a_wrong_year_for_the_right_person_is_declined():
    # 2019 is on the page, but it belongs to Sofia Marchetti.
    problems = check_grounding("Raj Patel joined in 2019 [1].", [_node(LEADERSHIP)])
    assert any("not stated together" in p for p in problems)


def test_a_claim_spanning_two_adjacent_sentences_is_accepted():
    # "Raj Patel, Chief Technology Officer." then "Raj joined ... in 2016."
    assert check_grounding("Raj Patel, the CTO, joined in 2016 [1].", [_node(LEADERSHIP)]) == []


@pytest.mark.parametrize("claim", [
    "Elena Cho resigned from Meridian Robotics [1].",
    "Meridian Robotics is shutting down operations [1].",
    "Marcus is the chief executive [1].",
])
def test_the_right_entities_with_the_wrong_claim_are_declined(claim):
    assert check_grounding(claim, [_node(LEADERSHIP)])


def test_a_sentence_only_about_the_sources_is_neutral():
    # qwen3:8b ends most answers like this. It states nothing that can be wrong.
    answer = "Raj Patel is the CTO [1]. This information is consistent across sources [1]."
    assert check_grounding(answer, [_node(LEADERSHIP)]) == []


def test_a_close_paraphrase_is_accepted():
    # Real qwen3:8b output against the leadership page.
    answer = "He joined Meridian Robotics in 2016 and oversees the engineering organization [1]."
    assert check_grounding(answer, [_node(LEADERSHIP)]) == []


# --- GroundedQueryEngine: the pipeline entry point ------------------------

class FakeIndex:
    """Returns `per_k[k]` for a retriever of size k, and records which k were asked for."""

    def __init__(self, per_k):
        self.per_k, self.asked = per_k, []

    def as_retriever(self, similarity_top_k):
        self.asked.append(similarity_top_k)
        nodes = self.per_k[similarity_top_k]
        return type("R", (), {"retrieve": lambda _self, _q: nodes})()


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return _Reply(self.replies.pop(0))


def test_a_not_answered_decline_is_retried_with_more_chunks():
    index = FakeIndex({4: [_node(CFO_FACT)], 8: [_node(CFO_FACT), _node(CEO_FACT)]})
    llm = ScriptedLLM(ABSTAIN_TOKEN, "Elena Cho has served as Chief Executive Officer since 2015 [2].")

    result = GroundedQueryEngine(index, llm).query("Who is the CEO?")

    assert index.asked == [4, 8]
    assert not result.abstained


def test_the_retry_happens_once_and_its_decline_stands():
    index = FakeIndex({4: [_node(CFO_FACT)], 8: [_node(CFO_FACT)]})
    result = GroundedQueryEngine(index, ScriptedLLM(ABSTAIN_TOKEN, ABSTAIN_TOKEN)).query("Who is the COO?")

    assert index.asked == [4, 8]
    assert result.abstained


@pytest.mark.parametrize("nodes, reply", [
    ([_node("Marcus Webb is CEO.", "D", "inbox")], None),       # nothing trusted: more chunks won't help
    ([_node(CEO_FACT)], "Marcus Webb is the CEO [1]."),          # unsupported claim: not a retrieval problem
])
def test_other_declines_are_not_retried(nodes, reply):
    index = FakeIndex({4: nodes})
    llm = ScriptedLLM(*([reply] if reply else []))

    result = GroundedQueryEngine(index, llm).query("Who is the CEO?")

    assert index.asked == [4]
    assert result.abstained


def test_build_grounded_query_engine_runs_the_whole_reviewed_pipeline(tmp_path, monkeypatch):
    from llama_index.core.llms import MockLLM

    from provenance_gated_rag.ingest import build_grounded_query_engine

    wiki = tmp_path / "corpus" / "verified_internal_wiki"
    wiki.mkdir(parents=True)
    (wiki / "leadership.txt").write_text(CEO_FACT)
    rules = tmp_path / "trust.yaml"
    rules.write_text("origins:\n  verified_internal_wiki:\n    tier: B\n    reason: wiki\n    editability: authenticated-few\n")
    monkeypatch.setattr("builtins.input", lambda _p="": "")
    monkeypatch.setattr("provenance_gated_rag.ingest.get_embed_model", lambda: MockEmbedding(embed_dim=8))
    monkeypatch.setattr("provenance_gated_rag.ingest.get_llm", MockLLM)
    previous = Settings._embed_model, Settings._llm
    try:
        engine = build_grounded_query_engine(tmp_path / "corpus", tmp_path / "decisions", rules)
        engine.llm = FakeLLM("Elena Cho has served as Chief Executive Officer since 2015 [1].")
        result = engine.query("Who is the CEO?")
    finally:
        Settings._embed_model, Settings._llm = previous

    assert isinstance(engine, GroundedQueryEngine)
    assert not result.abstained, result.reasons


def test_a_paraphrase_around_a_verified_number_is_accepted():
    # Real qwen3:8b output that was declined at the 60% bar.
    source = "To reset your password, open Settings. Passwords must be at least 12 characters long."
    assert check_grounding("The minimum password length is 12 characters [1].", [_node(source)]) == []


def test_the_wrong_number_in_the_same_paraphrase_is_still_declined():
    source = "To reset your password, open Settings. Passwords must be at least 12 characters long."
    assert check_grounding("The minimum password length is 6 characters [1].", [_node(source)])


def test_the_lower_bar_needs_a_number_not_just_a_name():
    # Same 50% overlap, but no number anchors it: stays declined.
    problems = check_grounding("Elena Cho founded a charity [1].", [_node(CEO_FACT + " She founded the company.")])
    assert problems


def test_plurals_match_their_singular():
    assert check_grounding("Refund requests are processed within 14 days [1].",
                           [_node("Refunds are processed within 14 days of the request.")]) == []


def test_a_sentence_describing_where_a_fact_is_documented_is_not_declined():
    # Real qwen3:8b output that declined a correct CTO answer at 58%: the
    # words about the source ("documentation", "references", "highlighting")
    # were counted as claim words.
    lineup = (
        "Meridian Robotics manufactures three product lines of autonomous warehouse robots. All three "
        "product lines share a common navigation stack developed by the engineering team under CTO Raj Patel."
    )
    sentence = (
        "Additionally, the product lineup documentation also references Raj Patel as the CTO, highlighting "
        "his role in developing the common navigation stack for the company's robots [1]."
    )
    assert check_grounding(sentence, [_node("Meridian Robotics - Product Lineup\n\n" + lineup)]) == []


def test_the_engine_runs_its_postprocessors_before_choosing_evidence():
    # A chunk the query-time screen neutralises is set aside, not used.
    from provenance_gated_rag.sanitizer import REDACTION_MARKER

    class Neutralise:
        def postprocess_nodes(self, nodes, query_str=None):
            for n in nodes:
                n.node.set_content(REDACTION_MARKER)
            return nodes

    llm = ScriptedLLM("Elena Cho has served as Chief Executive Officer since 2015 [1].")
    index = FakeIndex({4: [_node(CEO_FACT)]})

    result = GroundedQueryEngine(index, llm, node_postprocessors=[Neutralise()]).query("Who is the CEO?")

    assert result.abstained
    assert llm.prompts == []   # nothing trusted was left, so the model was never asked


@pytest.mark.parametrize("label", ["Answer: ", "**Answer:** ", "Final answer: "])
def test_an_answer_label_in_front_of_the_answer_is_not_a_claim(label):
    # qwen3:8b, live, 2026-10-23: "Answer: Raj Patel [1]." was declined at 0%.
    source = _node("Raj Patel is the Chief Technology Officer of Meridian Robotics.")

    assert check_grounding(f"{label}Raj Patel [1].", [source]) == []


def test_an_answer_label_does_not_excuse_an_unsupported_claim():
    source = _node("Raj Patel is the Chief Technology Officer of Meridian Robotics.")

    assert check_grounding("Answer: Marcus Webb [1].", [source])


# --- polarity: a claim and its source must agree on "not" -------------------

@pytest.mark.parametrize("answer", [
    "Elena Cho has not served as Chief Executive Officer [1].",
    "Elena Cho never served as Chief Executive Officer [1].",
    "Elena Cho hasn't served as Chief Executive Officer since 2015 [1].",
])
def test_a_claim_that_negates_its_source_is_declined(answer):
    problems = check_grounding(answer, [_node(CEO_FACT)])

    assert any("negates what the source(s) state" in p for p in problems)


def test_no_longer_is_declined_too():
    # "longer" is in no source either, so the word rule may catch it first.
    assert check_grounding("Elena Cho is no longer the Chief Executive Officer [1].", [_node(CEO_FACT)])


def test_a_claim_that_drops_the_sources_not_is_declined():
    source = _node("Marcus Webb is not the Chief Executive Officer of Meridian Robotics.")

    problems = check_grounding("Marcus Webb is the Chief Executive Officer of Meridian Robotics [1].", [source])

    assert any("states what the source(s) negate" in p for p in problems)


def test_a_negation_the_source_also_makes_is_grounded():
    source = _node("Marcus Webb is not an employee of Meridian Robotics.")

    assert check_grounding("Marcus Webb is not an employee of Meridian Robotics [1].", [source]) == []


def test_a_negation_elsewhere_in_the_source_does_not_count_against_a_positive_claim():
    # The "not" is in a different sentence from the one the claim rests on.
    source = _node(CEO_FACT + " She does not sit on any other board.")

    assert check_grounding("Elena Cho has served as Chief Executive Officer since 2015 [1].", [source]) == []


def test_a_negated_claim_with_no_names_is_checked_against_the_matching_sentence():
    source = _node("Refunds are processed within two weeks of the request.")

    problems = check_grounding("Refunds are not processed within two weeks of the request [1].", [source])

    assert any("negates" in p for p in problems)


def test_polarity_is_checked_when_the_facts_span_two_source_sentences():
    # "Raj Patel" and "2016" are in different sentences, so the claim rests on
    # the pair of them -- and is still declined for negating it.
    source = _node("Raj Patel is the Chief Technology Officer. Raj joined the company in 2016.")

    problems = check_grounding("Raj Patel did not join the company in 2016 [1].", [source])

    assert any("negates" in p for p in problems)


@pytest.mark.parametrize("answer", [
    "Elena Cho has served as Chief Executive Officer since 2015, and still does [1].",
    "The company has 1,200 staff, mostly engineers [1].",
])
def test_a_comma_after_a_number_is_not_part_of_it(answer):
    source = _node("Elena Cho has served as Chief Executive Officer since 2015. "
                   "The company has 1,200 staff, mostly engineers, and still does.")

    assert not any("not in the source" in p for p in check_grounding(answer, [source]))
