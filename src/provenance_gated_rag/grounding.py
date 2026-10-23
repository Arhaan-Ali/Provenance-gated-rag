"""M5 grounding and abstention: answer from the sources, or decline and say why.

A plain pipeline always produces an answer. When retrieval finds nothing that
answers the question, the model fills the gap from memory or from whatever
was nearest, and the result reads exactly like a grounded answer. Nothing
downstream can tell the two apart.

Two checks here, on either side of the model call, and neither asks the
model whether it is sure:

- Before generation, select_evidence decides which retrieved chunks may
  support an answer at all. A neutralised chunk has no content, and a chunk
  below GROUNDING_MIN_TIER is not evidence -- an unverified source saying X
  is a fact about the source, not about X. If nothing is left, the model is
  never called, because a model given no evidence still answers.

- After generation, check_grounding requires every name and number in each
  sentence to appear in the source it cites -- or, if it cites none, in some
  single trusted source. A model can be told to abstain, and qwen3:8b often
  does; it can also ignore that, which is the case this exists for. A claim
  that is not in the trusted text is declined, whatever the model meant by it.

The lexical check is deliberately narrow. It catches the failure that matters
most here -- an entity or figure that is in no source, such as a planted CEO
name or an invented price -- and does not try to judge paraphrase, which it
cannot do reliably. A sentence with no names or numbers passes on its
citation alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from provenance_gated_rag.capability_gate import tiers_from_nodes
from provenance_gated_rag.facts import answer_conflicts, facts_from_nodes
from provenance_gated_rag.registry import TrustTier
from provenance_gated_rag.sanitizer import REDACTION_MARKER

# Lowest tier a chunk may have and still support a stated fact. C, not B:
# external but verified is still verified. D is what the planted documents
# arrive as, and also what an unlabelled chunk counts as, so both are set
# aside by the same rule.
GROUNDING_MIN_TIER = TrustTier.C

# What the model is told to reply when the sources do not answer the
# question. A fixed token, not "say you don't know": a phrase in the model's
# own words cannot be matched reliably, and a missed abstention would then be
# graded as an answer.
ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

_NOT_ANSWERED = "do not answer the question"

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# qwen3 emits its reasoning in <think> tags before the answer. Grounding is
# checked on the answer; the reasoning is free to mention anything.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
# A label the model puts in front of its answer -- "Answer: Raj Patel [1]."
# declined a correct answer in a real run, because "answer" is in no source.
# Markdown emphasis around it ("**Answer:**") goes with it.
_ANSWER_LABEL = re.compile(r"^[\s*_#>-]*(?:final\s+)?answer[\s*_]*:[\s*_]*", re.IGNORECASE)
# Thousands separators inside a number, never a trailing comma: "since 2015,"
# was read as the term "2015," and declined for not being in the source.
_NUMBER = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?")
_CAPITALISED = re.compile(r"\b[A-Z][a-zA-Z'-]+\b")

# Capitalised only because they start a sentence or are pronouns. Anything
# else capitalised is treated as a name, which errs towards declining.
_NOT_NAMES = frozenset({
    "A", "An", "The", "This", "That", "These", "Those", "It", "Its", "They", "Their",
    "He", "She", "His", "Her", "We", "Our", "I", "You", "Your", "In", "On", "At", "As",
    "By", "For", "From", "Of", "To", "With", "And", "But", "Or", "If", "So", "Yes", "No",
    "According", "Based", "There", "Here", "Both", "Each", "All", "None", "Not", "Also",
    "However", "Since", "After", "Before", "During", "While", "When", "Although", "Today",
    "Currently", "Previously", "Prior", "Together", "Under", "Over", "One",
})


@dataclass(frozen=True)
class SetAside:
    """A retrieved chunk that was not allowed to support the answer, and why."""

    origin: str
    reason: str


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    abstained: bool
    reasons: tuple[str, ...] = ()
    sources: tuple = ()
    set_aside: tuple[SetAside, ...] = ()

    def __str__(self) -> str:
        return self.explain()

    def explain(self) -> str:
        """The answer, or the decline with its reasons, as one printable block."""
        if not self.abstained:
            return self.text
        lines = ["I can't answer this from the sources I'm allowed to rely on."]
        lines += [f"- {reason}" for reason in self.reasons]
        lines += [f"- Set aside a source from {s.origin}: {s.reason}." for s in self.set_aside]
        return "\n".join(lines)


def select_evidence(nodes, min_tier: TrustTier = GROUNDING_MIN_TIER):
    """Split retrieved nodes into (usable, set_aside).

    Tiers come from tiers_from_nodes so an unlabelled node gets the gate's
    fail-closed answer, the lowest tier, rather than one chosen here.
    """
    usable, set_aside = [], []
    for node, tier in zip(nodes, tiers_from_nodes(nodes)):
        origin = (getattr(node, "metadata", None) or {}).get("origin", "an unknown origin")
        if node.get_content().strip() == REDACTION_MARKER:
            set_aside.append(SetAside(origin, "it was neutralised as an instruction document"))
        elif tier < min_tier:
            set_aside.append(
                SetAside(origin, f"its trust tier is {tier.name}, below the {min_tier.name} needed to support an answer")
            )
        else:
            usable.append(node)
    return usable, set_aside


def build_prompt(query: str, sources) -> str:
    numbered = "\n\n".join(f"[{i}] {node.get_content()}" for i, node in enumerate(sources, start=1))
    return (
        "Answer the question using only the numbered sources below.\n"
        "Rules:\n"
        "- After every sentence, cite the source it comes from, like [1].\n"
        "- Do not use anything you know that is not in the sources.\n"
        f"- If the sources do not answer the question, reply with exactly {ABSTAIN_TOKEN} and nothing else.\n\n"
        f"Sources:\n{numbered}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def _key_terms(sentence: str) -> list[str]:
    without_citations = _CITATION.sub("", sentence)
    words = without_citations.split()
    # A sentence's first word is capitalised whatever it is -- "Additionally"
    # declined a correct answer in a real run. Openers like that are followed
    # by a comma, or are in _NOT_NAMES; anything else opening a sentence is
    # read as a name, so "Marcus is CEO" is checked like "Marcus Webb is CEO".
    skip_first = bool(words) and not words[0][-1:].isalpha()
    names = [
        w
        for i, w in enumerate(_CAPITALISED.findall(without_citations))
        if w not in _NOT_NAMES and not (i == 0 and skip_first and words and words[0].startswith(w))
    ]
    return _NUMBER.findall(without_citations) + names


def _appears_in(term: str, text: str) -> bool:
    """Whether term is in text, counting an acronym as present if its words are.

    "CEO" where the source says "Chief Executive Officer" is the commonest
    paraphrase a model makes, and declining it would decline most correct
    answers. Only an all-caps term gets this, and only against CAPITALISED
    words in the original text: "across continents every" does not spell
    "ACE", but "Chief Executive Officer" spells "CEO".
    """
    if term.lower() in text.lower():
        return True
    if not (term.isupper() and 2 <= len(term) <= 5):
        return False
    words = re.findall(r"[A-Za-z0-9]+", text)
    size = len(term)
    return any(
        all(w[:1].isupper() for w in words[i : i + size])
        and "".join(w[0] for w in words[i : i + size]).upper() == term
        for i in range(len(words) - size + 1)
    )


# Words a model uses to talk ABOUT its sources rather than to state a claim.
# A sentence made only of these ("This is consistent across sources [1] and
# [3].") says nothing that could be false, so it is not held to the overlap
# rule below -- qwen3:8b writes one in most answers.
_META_WORDS = frozenset(
    ["source", "sources", "information", "context", "provided", "consistent", "corroborated", "corroborates", "confirms", "confirmed", "states", "stated", "mentions", "mentioned", "according", "document", "documents", "indicates", "indicated", "specified", "notes", "noted", "listed", "section", "details", "also", "additionally", "furthermore", "moreover", "overview", "across", "between", "multiple", "several", "various", "throughout", "documentation", "references", "referenced", "refers", "highlighting", "highlights", "highlighted", "describes", "described", "explicitly", "reiterates", "reiterated"]
)
_STOPWORDS = frozenset(
    ["the", "and", "for", "are", "was", "were", "has", "have", "had", "been", "being", "with", "that", "this", "these", "those", "from", "into", "onto", "over", "under", "than", "then", "its", "it's", "his", "her", "hers", "their", "they", "them", "she", "he", "who", "whom", "which", "what", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "not", "only", "own", "same", "can", "will", "just", "should", "now", "also", "very", "is", "be", "as", "of", "to", "in", "on", "at", "by", "an", "or", "if", "so", "do", "does", "did", "our", "your", "you", "we", "us"]
)
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")

# Share of a sentence's content words that must appear in the source it
# relies on. Names and numbers are checked separately and strictly; this is
# what catches the claim built from the right entities and the wrong verb --
# "Elena Cho resigned", "the company is shutting down".
MIN_WORD_OVERLAP = 0.6
# Lower bar for a sentence whose number has already been found in place,
# beside its names. The number is the claim; the words around it are the
# model's phrasing of the source's ("minimum length is 12 characters" for
# "at least 12 characters long" -- a real false decline at 0.6).
MIN_WORD_OVERLAP_ANCHORED = 0.5


def _content_words(sentence: str) -> list[str]:
    """The words that carry the claim, minus names and numbers.

    Names and numbers are checked strictly on their own; left in here they
    would carry the overlap -- "Elena Cho resigned from Meridian Robotics" is
    80% names, so the one word that is false would not matter.
    """
    names = {t.lower() for t in _key_terms(sentence)}
    words = _WORD.findall(_CITATION.sub("", sentence).lower())
    return [
        w for w in words
        if len(w) >= 3 and w not in _STOPWORDS and w not in _META_WORDS and w not in names
        and not _NEGATION.fullmatch(w)   # judged by _polarity_problem, not counted here
    ]


def _windows(text: str) -> list[str]:
    """Every pair of adjacent sentences in text (or the one sentence there is).

    The unit a claim's names and numbers have to share. A whole chunk is too
    coarse -- a leadership page names every executive and every title, so
    "Raj Patel is the CFO" found all three somewhere in it -- and a single
    sentence is too fine, because sources say "Raj Patel, CTO. Raj joined in
    2016." across a full stop.
    """
    sentences = [x for x in _SENTENCE_SPLIT.split(re.sub(r"\s+", " ", text).strip()) if x]
    if len(sentences) < 2:
        return sentences
    return [f"{a} {b}" for a, b in pairwise(sentences)]


# Words that flip what a sentence asserts. The overlap rule counts words,
# so "Elena Cho does not lead Meridian Robotics" matched a source saying she
# leads it on almost every word. The sentence and the source passage it
# relies on must agree on whether they negate.
_NEGATION = re.compile(r"\b(?:not|no|never|none|neither|nor|cannot)\b|n't\b", re.IGNORECASE)


def _negated(text: str) -> bool:
    return bool(_NEGATION.search(text))


def _sentences(text: str) -> list[str]:
    return [x for x in _SENTENCE_SPLIT.split(re.sub(r"\s+", " ", text).strip()) if x]


def _passages(terms: list[str], words: list[str], texts: list[str]) -> list[str]:
    """The source sentences a claim rests on.

    One holding every name and number in it, else a two-sentence window
    that does; with no names or numbers, a sentence sharing enough of its
    words.
    """
    if terms:
        found = [x for t in texts for x in _sentences(t) if all(_appears_in(term, x) for term in terms)]
        return found or [w for t in texts for w in _windows(t) if all(_appears_in(term, w) for term in terms)]
    return [x for t in texts for x in _sentences(t) if _overlap(words, x) >= MIN_WORD_OVERLAP]


def _polarity_problem(sentence: str, terms: list[str], words: list[str], texts: list[str]) -> str | None:
    """The sentence negates where its source passage does not, or the other way round.

    The passage is the source sentence the claim rests on: one containing
    every name and number in it, or -- with none to go on -- one sharing
    enough of its words. If any such passage agrees in polarity, it passes.
    """
    claim = _CITATION.sub("", sentence)
    passages = _passages(terms, words, texts)
    if not passages or any(_negated(x) == _negated(claim) for x in passages):
        return None
    if _negated(claim):
        return f'"{sentence}" negates what the source(s) state.'
    return f'"{sentence}" states what the source(s) negate.'


def _support_problem(sentence: str, texts: list[str]) -> str | None:
    """Why none of texts supports sentence, or None if one of them does."""
    terms = _key_terms(sentence)
    words = _content_words(sentence)
    if not terms and not words:
        return None  # nothing but talk about the sources

    absent = [term for term in terms if not any(_appears_in(term, t) for t in texts)]
    if absent:
        listed = ", ".join(f'"{t}"' for t in absent)
        return f'{listed} not in the source(s) relied on for "{sentence}".'
    if terms and not any(all(_appears_in(t, w) for t in terms) for text in texts for w in _windows(text)):
        listed = ", ".join(f'"{t}"' for t in terms)
        return f"{listed} are not stated together in the source(s) relied on for \"{sentence}\"."
    if words:
        anchored = any(_NUMBER.fullmatch(t) for t in terms)
        needed = MIN_WORD_OVERLAP_ANCHORED if anchored else MIN_WORD_OVERLAP
        best = max(_overlap(words, t) for t in texts)
        if best < needed:
            return (
                f'"{sentence}" is not what the source(s) say: only {best:.0%} of its words appear '
                f"there (at least {needed:.0%} needed)."
            )
    return _polarity_problem(sentence, terms, words, texts)


def _stem(word: str) -> str:
    # Plural only. Anything cleverer starts matching words that differ in
    # meaning, and every false match is a claim let through.
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _overlap(words: list[str], text: str) -> float:
    present = {_stem(w) for w in _WORD.findall(text.lower())}
    return sum(_stem(w) in present for w in words) / len(words)


# Thresholds for the optional entailment model (nli.EntailmentJudge). A
# claim is declined for CONTRADICTION, not for a lack of entailment: the
# model rates a fair paraphrase ("two weeks at most" for "within 14 days")
# as neutral, and declining neutral would decline those. A passage that
# entails the claim at ENTAILED or above clears it, whatever another
# passage of the same source says.
_DISAGREE = ("The trusted sources disagree on what this answer states, so it is left to you "
             "to decide which is right.")

ENTAILED = 0.5
CONTRADICTED = 0.9
# How many passages of other trusted sources a claim is compared with.
_MAX_CROSS_CHECKS = 12


def _meaning_problem(sentence: str, terms: list[str], words: list[str], relied: list[str],
                     others: list[tuple[int, str]], judge) -> str | None:
    """What an entailment model says is wrong with a claim the lexical checks passed.

    relied are the texts the sentence rests on; others are the remaining
    trusted sources as (number, text). The claim is judged against the
    passages it rests on, and against passages of the other sources that
    mention any of its names or numbers -- which is where two trusted
    sources disagreeing shows up.
    """
    claim = _CITATION.sub("", sentence).strip()
    own = _passages(terms, words, relied) or [w for t in relied for w in _windows(t)]
    cross = [
        (n, x) for n, text in others for x in _sentences(text)
        if terms and any(_appears_in(term, x) for term in terms)
    ][:_MAX_CROSS_CHECKS]
    judged = judge([(x, claim) for x in own] + [(x, claim) for _, x in cross])
    on_own, on_cross = judged[:len(own)], judged[len(own):]

    if on_own and max(j.entailment for j in on_own) < ENTAILED:
        worst = max(range(len(own)), key=lambda i: on_own[i].contradiction)
        if on_own[worst].contradiction >= CONTRADICTED:
            return f'"{sentence}" contradicts the source it relies on, which says: "{own[worst]}"'
    for (n, passage), j in zip(cross, on_cross, strict=True):
        if j.contradiction >= CONTRADICTED:
            return f'"{sentence}" is contradicted by trusted source [{n}], which says: "{passage}"'
    return None


def check_grounding(answer: str, sources, judge=None) -> list[str]:
    """Every problem that stops this answer counting as grounded. Empty means grounded.

    A sentence is supported when (1) every name and number in it appears in
    what it relies on, (2) all of them appear together within two adjacent
    sentences there, and (3) most of its other content words appear there
    too. A cited sentence relies on the sources it cites; an uncited one may
    rely on any single trusted source.

    judge, if given (nli.EntailmentJudge), then reads each sentence that
    passed against what it relies on and against the other trusted sources,
    and declines one either contradicts.
    """
    problems = []
    sentences = []
    for piece in (s.strip() for s in _SENTENCE_SPLIT.split(answer.strip())):
        # "Elena Cho is the CEO. [1]" puts the citation after the full stop,
        # so the split leaves it on its own. It belongs to the sentence before.
        if sentences and piece and not _CITATION.sub("", piece).strip(" ."):
            sentences[-1] = f"{sentences[-1]} {piece}"
        elif piece:
            sentences.append(piece)
    if not sentences:
        return ["The model returned an empty answer."]

    texts = [s.get_content() for s in sources]
    for sentence in (_ANSWER_LABEL.sub("", s) for s in sentences):
        cited = [int(n) for n in _CITATION.findall(sentence)]
        missing = [n for n in cited if not 1 <= n <= len(sources)]
        if missing:
            problems.append(f'"{sentence}" cites [{missing[0]}], which is not one of the {len(sources)} sources.')
            continue
        if cited:
            relied = sorted(set(cited))
            problem = _support_problem(sentence, [" ".join(texts[n - 1] for n in cited)])
        else:
            # Models drop citations often -- qwen3:8b answered the M0 question
            # correctly with none, and declining that is a false abstention.
            # Uncited, it must still be supported by ONE trusted source on its
            # own, not pieced together from several.
            per_source = [_support_problem(sentence, [t]) for t in texts]
            relied = [i + 1 for i, p in enumerate(per_source) if p is None][:1]
            problem = None if relied else (
                f'"{sentence}" does not cite a source, and no single trusted source supports it.'
            )
        terms, words = _key_terms(sentence), _content_words(sentence)
        if problem is None and judge is not None and (terms or words):
            problem = _meaning_problem(
                sentence, terms, words,
                [texts[n - 1] for n in relied],
                [(n, texts[n - 1]) for n in range(1, len(texts) + 1) if n not in relied],
                judge,
            )
        if problem:
            problems.append(problem)
    return problems


def answer_with_grounding(
    query: str,
    nodes,
    llm,
    min_tier: TrustTier = GROUNDING_MIN_TIER,
    judge=None,
    facts=(),
) -> GroundedAnswer:
    """Answer query from the retrieved nodes, or decline with the reasons.

    llm needs only .complete(prompt).text, which every LlamaIndex LLM has.
    judge is the optional entailment model (see check_grounding). facts are
    trusted facts from beyond the retrieved nodes -- GroundedQueryEngine
    passes the whole index's -- so a disagreement between two trusted
    sources declines the answer even when only one of them was retrieved.
    """
    usable, set_aside = select_evidence(nodes, min_tier)
    if not usable:
        return GroundedAnswer(
            text="",
            abstained=True,
            reasons=(f"None of the {len(nodes)} retrieved source(s) is trusted enough to support an answer.",),
            set_aside=tuple(set_aside),
        )

    raw = llm.complete(build_prompt(query, usable)).text
    answer = _THINK.sub("", raw).strip()

    if ABSTAIN_TOKEN in answer:
        return GroundedAnswer(
            text="",
            abstained=True,
            reasons=(f"The {len(usable)} trusted source(s) retrieved {_NOT_ANSWERED}.",),
            sources=tuple(usable),
            set_aside=tuple(set_aside),
        )

    problems = check_grounding(answer, usable, judge)
    if not problems:
        disputed = answer_conflicts(answer, [*facts, *facts_from_nodes(usable, min_tier)])
        if disputed:
            return GroundedAnswer(
                text=answer,
                abstained=True,
                reasons=(_DISAGREE, *disputed),
                sources=tuple(usable),
                set_aside=tuple(set_aside),
            )
    if problems:
        return GroundedAnswer(
            text=answer,
            abstained=True,
            reasons=("The model's answer made claims its sources do not support.", *problems),
            sources=tuple(usable),
            set_aside=tuple(set_aside),
        )

    return GroundedAnswer(text=answer, abstained=False, sources=tuple(usable), set_aside=tuple(set_aside))


class GroundedQueryEngine:
    """An index's retriever plus answer_with_grounding, behind .query(query).

    Declining because the trusted sources "do not answer" may only mean the
    answer was not in the top k. So that decline is retried once with twice
    as many chunks before it stands. Only that one: a decline for untrusted
    evidence or unsupported claims would not change with more chunks, and
    retrying it would just spend a second model call.
    """

    def __init__(
        self,
        index,
        llm,
        similarity_top_k: int = 4,
        min_tier: TrustTier = GROUNDING_MIN_TIER,
        node_postprocessors=(),
        judge=None,
    ):
        self.index = index
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.min_tier = min_tier
        # Run on what was retrieved, before evidence is selected -- e.g. the
        # sanitizer's query-time screen, whose neutralised chunks
        # select_evidence then sets aside.
        self.node_postprocessors = tuple(node_postprocessors)
        self.judge = judge
        # Every trusted fact in the index, read once: a conflict with a
        # source that was not retrieved for this question still counts.
        docstore = getattr(index, "docstore", None)
        self.facts = tuple(facts_from_nodes(list(docstore.docs.values()), min_tier)) if docstore else ()

    def retrieve(self, query: str, top_k: int | None = None):
        """What the answer would be chosen from: retrieved, then screened."""
        nodes = self.index.as_retriever(similarity_top_k=top_k or self.similarity_top_k).retrieve(query)
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_str=query)
        return nodes

    def _answer(self, query: str, top_k: int) -> GroundedAnswer:
        return answer_with_grounding(
            query, self.retrieve(query, top_k), self.llm, self.min_tier, self.judge, self.facts
        )

    def query(self, query: str) -> GroundedAnswer:
        result = self._answer(query, self.similarity_top_k)
        if result.abstained and result.reasons and _NOT_ANSWERED in result.reasons[0]:
            return self._answer(query, self.similarity_top_k * 2)
        return result
