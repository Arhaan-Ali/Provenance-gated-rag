"""Instruction sanitizer: drops chunks that instruct a model how to answer.

Text that explicitly addresses an AI/assistant, or tells one to state
something "with full confidence", is evidence the surrounding text was
authored to manipulate a model -- not just the one sentence carrying the
directive. A document willing to instruct a model has no remaining claim
to being trusted for facts either, so a single trigger anywhere in it
discards all of it rather than only the triggering sentence.

That decision belongs at the document level, before chunking:
sanitize_documents() is the real defense. The postprocessor below is
defense in depth for documents that reach an index by another route --
on its own it can be evaded by padding a document past the chunk size so
the directive and the false claim land in separate chunks.

This catches the M0/adversarial-eval attack template ("NOTICE TO ANY
AUTOMATED ASSISTANT ... must state X with full confidence"), which
addresses the model directly. It does not catch a subtler injection with no
AI-facing framing at all (see M3's planted_tool_injection.txt) -- that
attack has nothing for a content sanitizer to see and is caught instead by
the capability gate refusing the resulting action, not by detecting the
text. The two milestones are complementary, not redundant.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

SANITIZER_FLAG_KEY = "sanitizer_flagged"

REDACTION_MARKER = "[REDACTED: this chunk instructed a model how to answer and was dropped by the sanitizer]"

# Detection is CO-OCCURRENCE of two independent signals, not proximity.
#
# This used to join each pair with a bounded gap -- first [^.]{0,40}/{0,80},
# then .{0,120}? after punctuation turned out to split a match. Both were
# bypassed the same way: a proximity window is a number, and the attacker
# writes the document, so they simply exceed it. 159 characters of ordinary
# filler between "must state" and "with full confidence" defeated the 120
# version while using the exact trigger vocabulary -- no homoglyph, no
# synonym. Widening to 200 just moves the number.
#
# So there is no distance term at all now. Each half below is ordinary
# business English on its own and neither flags anything by itself; the pair
# appearing in one document is what is rare. That matches the decision the
# module already makes, which is per document and not per sentence or chunk:
# if the document is arguing with a model anywhere in it, none of it is
# indexed.
#
# Co-occurrence is scoped to a PARAGRAPH, which is a structural unit rather
# than a number. That matters in both directions:
#
#   - Punctuation cannot break it. A "!" or an "I.E." splits a sentence but
#     not a paragraph, so the bypasses that defeated the sentence-scoped and
#     character-bounded versions stay closed.
#   - Padding cannot stretch it. To get the two halves into different
#     paragraphs the attacker has to insert a blank line, which visibly
#     breaks the instruction into two disconnected statements rather than
#     burying a join inside one.
#   - And it does not fire on ordinary documents. Scoping to the whole
#     document instead flags a support handbook whose escalation section
#     says "must respond" and whose board section says "with full
#     confidence" 4,000 characters later. That is a real document and a real
#     false positive; a paragraph scope does not flag it.
#
# This is still bypassable -- an attacker who splits the instruction across a
# blank line gets through. There is no scope that is both tight enough to
# avoid false positives and loose enough to catch every arrangement, which is
# precisely why the capability gate does not read content at all.
#
# Measured on this repo's corpus: 6 of 6 planted documents flagged, 0 of 20
# genuine documents and 0 of 9 benign lookalike sentences. Re-measure before
# widening either half -- tests/test_sanitizer.py pins all three numbers.

# Signal pair 1: a document that addresses a model directly.
_ADDRESS_OPENER = re.compile(r"\bnotice to any\b", re.IGNORECASE)
_ADDRESS_TARGET = re.compile(
    r"\b(assistant|ai|language model|llm|chatbot)\b", re.IGNORECASE
)

# Self-contained phrasings: subject and target are already adjacent, so there
# is nothing to join and nothing for padding to get between.
_ADDRESS_DIRECT = re.compile(
    r"\bif you are (an? )?(ai|language model|assistant|chatbot)\b"
    r"|\b(dear|attention)[:,]?\s+(any\s+)?(automated\s+)?(assistant|ai|language model|chatbot)\b"
    # Ported from the fork. The lookahead keeps "instructions to the AI team"
    # and "... AI policy" -- ordinary internal memos -- from firing.
    r"|\binstructions? to (the |any |this )?(ai|llm|language model|chatbot)\b(?! (team|policy|ethics))",
    re.IGNORECASE,
)

# Signal pair 2: a document that demands an assertion be made with certainty.
_COERCION_IMPERATIVE = re.compile(r"\bmust (state|answer|respond)\b", re.IGNORECASE)
_COERCION_CERTAINTY = re.compile(r"\bwith full confidence\b", re.IGNORECASE)


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, before any whitespace collapsing.

    Must run on the raw text: _collapse flattens blank lines away, so
    collapsing first would destroy the only boundary this scope relies on.
    """
    return [chunk for chunk in _PARAGRAPH_SPLIT.split(text) if chunk.strip()]


def _signal_spans(collapsed: str) -> list[tuple[int, int]]:
    """Spans of every signal that fired in ONE paragraph, or [] if no pair.

    A single signal is never enough: "the assistant director must respond
    within one day" is one half of each pair and is not an attack.
    """
    spans: list[tuple[int, int]] = []

    opener = [m.span() for m in _ADDRESS_OPENER.finditer(collapsed)]
    target = [m.span() for m in _ADDRESS_TARGET.finditer(collapsed)]
    if opener and target:
        spans += opener + target

    imperative = [m.span() for m in _COERCION_IMPERATIVE.finditer(collapsed)]
    certainty = [m.span() for m in _COERCION_CERTAINTY.finditer(collapsed)]
    if imperative and certainty:
        spans += imperative + certainty

    spans += [m.span() for m in _ADDRESS_DIRECT.finditer(collapsed)]
    return spans

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# Cyrillic letters that render identically to Latin ones. NFKC does not fold
# these (they are distinct letters, not compatibility forms), so "аssistant"
# with a Cyrillic а used to match nothing. Ported from the fork.
_CONFUSABLES = str.maketrans({
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0425": "X", "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x", "\u0456": "i",
})

# Unicode categories that render as nothing: Cf (format: zero-width space,
# joiners, soft hyphen, the U+E0000 "tag" block), Co (private use), Cn
# (unassigned). NFKC does not remove them, so one inside a trigger word
# ("conf<ZWSP>idence") defeated every pattern above.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Co", "Cn"})


def _collapse(text: str, invisible_as: str = "") -> str:
    """The detection copy of a text: one line, single-spaced, normalised.

    Never stored and never shown to a model. Line wrapping is not a sentence
    break, so whitespace collapses; full-width and other compatibility forms
    fold under NFKC; Cyrillic look-alikes fold to Latin.

    invisible_as is what an invisible character becomes. Both answers are
    needed because the attacker picks the role it plays: inside a word it
    must vanish ("conf<ZWSP>idence" -> "confidence"); used as the gap between
    words it must become a space ("with<ZWSP>full<ZWSP>confidence"). Removing
    it can only fix the first, so sanitize_text scans both.
    """
    text = unicodedata.normalize("NFKC", text).translate(_CONFUSABLES)
    text = "".join(
        invisible_as if unicodedata.category(ch) in _INVISIBLE_CATEGORIES else ch
        for ch in text
    )
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(_collapse(text)) if s]


def _sentence_spans(collapsed: str) -> list[tuple[int, int, str]]:
    """(start, end, sentence) over the collapsed text, for reporting matches."""
    spans, cursor = [], 0
    for sentence in _SENTENCE_SPLIT.split(collapsed):
        if not sentence:
            continue
        start = collapsed.index(sentence, cursor)
        spans.append((start, start + len(sentence), sentence))
        cursor = start + len(sentence)
    return spans


@dataclass
class SanitizeResult:
    """The outcome of scanning one chunk of text for model-directed instructions."""

    sanitized_text: str
    triggering_sentences: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.triggering_sentences)


def sanitize_text(text: str) -> SanitizeResult:
    """Return text unchanged, or replaced with a redaction marker if triggered.

    The trigger decision is made on the WHOLE collapsed text, not sentence
    by sentence. Scanning per sentence was strictly weaker, because the
    splitter breaks on [.!?] while the patterns' own gap class excluded only
    ".": one "!" between "NOTICE TO ANY AUTOMATED SYSTEM" and "ASSISTANT"
    put the two halves in different sentences and nothing matched, even
    though the pattern matches the document perfectly well. Any punctuation
    the attacker can insert should not be a way out, so nothing is allowed
    to reset a match.

    The sentence split survives only to report WHICH text fired, which is
    all triggering_sentences was ever used for -- telling a reviewer why a
    document was dropped.
    """
    for invisible_as in ("", " "):
        triggering = _scan(text, invisible_as)
        if triggering:
            return SanitizeResult(sanitized_text=REDACTION_MARKER, triggering_sentences=triggering)
    return SanitizeResult(sanitized_text=text)


def _scan(text: str, invisible_as: str) -> list[str]:
    """The sentences that fired under ONE reading of invisible characters, or [].

    The decision and the explanation use the same reading on purpose: flagged
    means "there is an explanation", so the two must never disagree.
    """
    if not any(_signal_spans(_collapse(para, invisible_as)) for para in _paragraphs(text)):
        return []

    # Re-locate the signals in the collapsed whole, purely so the reporting
    # below can name the sentences a reviewer should read.
    collapsed = _collapse(text, invisible_as)
    spans = [
        match.span()
        for pattern in (
            _ADDRESS_OPENER, _ADDRESS_TARGET, _ADDRESS_DIRECT,
            _COERCION_IMPERATIVE, _COERCION_CERTAINTY,
        )
        for match in pattern.finditer(collapsed)
    ]

    # Every pattern needs word characters at both ends and the sentences tile
    # the collapsed text, so a match always overlaps at least one of them --
    # there is no "matched but belongs to no sentence" case to handle. A
    # defensive fallback here survived mutation testing, i.e. no test could
    # reach it, so it is left out rather than kept as an unfalsifiable branch.
    return [
        sentence
        for start, end, sentence in _sentence_spans(collapsed)
        if any(start < m_end and m_start < end for m_start, m_end in spans)
    ]


# --- two-stage screening ------------------------------------------------------
#
# Ported from the fork. Stage 1 is sanitize_text() above. Stage 2 is an
# optional model that scores how much the text reads like an injection:
#
#   patterns hit        -> flagged, STOP. The model is never called.
#   patterns clean      -> ask the model, flagged if its score >= threshold.
#   no model configured -> stage 1 alone, exactly the old behaviour.
#
# Patterns go first because they are free and explainable (the reviewer is
# shown the sentence that fired); the model is slow and gives a number, not a
# reason. The model is passed IN as a plain function, text -> score in [0, 1],
# so this module never imports torch: the real one is injection_classifier.py,
# and tests use a fake. If the classifier raises, the exception propagates --
# a scanner that fails quietly would read as "nothing found".

Classifier = Callable[[str], float]
DEFAULT_THRESHOLD = 0.92


@dataclass
class ScreenResult:
    """Outcome of screening one text.

    stage names the stage that DECIDED: "patterns" when they fired, or when
    no model is configured; "classifier" whenever the model was consulted
    (flagged or not). score is the model's score, None if it never ran.
    """

    flagged: bool
    stage: str
    score: float | None = None
    reasons: list[str] = field(default_factory=list)


def screen_text(
    text: str,
    classifier: Classifier | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> ScreenResult:
    hit = sanitize_text(text)
    if hit.flagged:
        return ScreenResult(True, "patterns", None, hit.triggering_sentences)

    if classifier is None or not text.strip():
        return ScreenResult(False, "patterns")

    # The model reads the same normalised copy the patterns read, so the
    # invisible-character and look-alike tricks do not work on it either.
    score = float(classifier(_collapse(text)))
    if score >= threshold:
        return ScreenResult(
            True, "classifier", score,
            [f"classifier score {score:.2f} >= threshold {threshold:.2f}"],
        )
    return ScreenResult(False, "classifier", score)


def sanitize_documents(
    documents,
    classifier: Classifier | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list, list]:
    """Neutralise whole documents that instruct a model, before chunking.

    Returns (kept, neutralised). Every input document appears in `kept` --
    the neutralised ones with their text replaced by REDACTION_MARKER. The
    second list is the same objects again, so a caller can report what was
    neutralised without having to diff.

    Two things are deliberate here.

    The decision is per DOCUMENT, before chunking. Chunk-level redaction is
    not sufficient on its own: a document longer than the chunk size is
    split, and the attacker controls that length. Put the directive in one
    chunk and the false claim in another and redacting the first leaves the
    second intact and retrievable; worse, the triggering chunk need not be
    retrieved at all, so a postprocessor never sees the evidence that the
    document it came from was hostile.

    But the document is NEUTRALISED, not DROPPED, and that is a correction.
    Dropping it made the sanitizer loosen the capability gate. The gate's
    protection is carried by the PRESENCE of low-tier chunks -- it takes
    min() over the tiers of what was retrieved -- and the sanitizer's job was
    to delete documents. Deleting a document deleted its tier along with its
    payload, and the documents it deletes are the loud ones the gate did not
    need to stop. Measured, same corpus, only this flag changed:

        sanitizer OFF  tiers=['B','D']  min=D  allowed=[search_corpus]
        sanitizer ON   tiers=['B']      min=B  allowed=[search_corpus,
                                                        send_email, web_search]

    The quiet payload in the B-tier origin -- no AI-facing wording, invisible
    to this function by design -- was top-ranked in both runs. Enabling the
    defence was what made that attack actionable.

    Keeping the node preserves the tier, so a hostile document still drags
    the floor down even once its text is gone. The invariant to hold on to:
    NO CONTENT-DEPENDENT STAGE MAY RAISE THE CAPABILITY FLOOR. That is also
    what InstructionSanitizerPostprocessor already did, so the two now agree.
    """
    kept, neutralised = [], []
    for document in documents:
        if screen_text(document.text, classifier, threshold).flagged:
            # set_content, not a metadata flag: the text is what reaches the
            # model, and a flag the model can read is a flag a document can
            # argue with.
            document.set_content(REDACTION_MARKER)
            neutralised.append(document)
        kept.append(document)
    return kept, neutralised


class InstructionSanitizerPostprocessor(BaseNodePostprocessor):
    """Wires sanitize_text into a query engine as a node postprocessor.

    Runs after retrieval, before the LLM sees any chunk's content -- the
    same boundary M2's label-propagation test already treats as where a
    postprocessor would read node.metadata.

    classifier is the optional stage 2 (see screen_text); None means patterns
    only, as before.
    """

    classifier: Any = None
    threshold: float = DEFAULT_THRESHOLD

    @classmethod
    def class_name(cls) -> str:
        return "InstructionSanitizerPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        for node_with_score in nodes:
            node = node_with_score.node
            if screen_text(node.text, self.classifier, self.threshold).flagged:
                node.set_content(REDACTION_MARKER)
                node.metadata[SANITIZER_FLAG_KEY] = True
                # Ported from the fork: the flag is for the caller, not the
                # model. Rendered into the prompt it is text a document can
                # argue with, and it says nothing a chunk is about.
                for excluded in (node.excluded_llm_metadata_keys, node.excluded_embed_metadata_keys):
                    if SANITIZER_FLAG_KEY not in excluded:
                        excluded.append(SANITIZER_FLAG_KEY)
        return nodes
