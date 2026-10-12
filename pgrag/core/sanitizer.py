"""Instruction sanitizer: drops documents that tell a model how to answer.

The registry decides whether an *origin* is trusted. This decides whether a
particular *document* is arguing with the model, which is a different
question: reviewing a folder is not the same as having read every file in
it, so an origin a reviewer accepted can still carry a planted file.

Text that explicitly addresses an AI, or tells one to state something "with
full confidence", is evidence the surrounding text was written to
manipulate a model -- not just the one sentence carrying the directive. A
document willing to instruct a model has no remaining claim to being
trusted for its facts either, so a single trigger anywhere in it discards
all of it.

That decision has to be made on the whole document, before chunking. A
document longer than the chunk size gets split, and the attacker controls
that length: put the directive in one chunk and the false claim in another,
and redacting only the first leaves the second intact and retrievable.
Worse, the triggering chunk need not be retrieved at all, so anything
working at query time never sees the evidence that the document it came
from was hostile. sanitize_documents() is therefore the real defence;
InstructionSanitizerPostprocessor in the llamaindex adapter is defence in
depth for documents that reach an index by some other route.

Deliberately NOT caught: an injection with no AI-facing wording at all -- a
plausible internal memo that happens to redirect something. There is
nothing here for a text matcher to see, and that attack is the capability
gate's job: it refuses the resulting action rather than detecting the text.
The two are complementary, and tests/test_sanitizer.py asserts the gap is
still there so nobody widens these patterns by accident.

No llama_index import on purpose, same as the rest of pgrag/core.
"""
from __future__ import annotations

import re
import unicodedata

from dataclasses import dataclass, field

REDACTION_MARKER = (
    "[REDACTED: this chunk instructed a model how to answer and was dropped by the sanitizer]"
)

# Detection is CO-OCCURRENCE of two independent signals, not proximity.
#
# These used to join each pair with a bounded gap -- .{0,120}? here, after
# [^.]{0,40}/{0,80} before it. Both are bypassed the same way: a proximity
# window is a number, and the attacker writes the document, so they exceed
# it. 159 characters of ordinary filler between "must state" and "with full
# confidence" defeats the 120 version while using the exact trigger
# vocabulary -- no homoglyph, no synonym. Widening to 200 moves the number.
#
# So there is no distance term. Each half below is ordinary business English
# alone and flags nothing by itself; the pair appearing together is what is
# rare. That matches the decision this module already makes, which is per
# document: if it is arguing with a model anywhere, none of it is indexed.
#
# Scoped to a PARAGRAPH, which is structural rather than numeric:
#   - Punctuation cannot break it. "!" or "I.E." splits a sentence, not a
#     paragraph, so the punctuation bypasses stay closed.
#   - Padding cannot stretch it. Separating the halves needs a blank line,
#     which visibly breaks the instruction into two statements.
#   - It does not fire on ordinary documents. Scoping to the whole document
#     instead flags a support handbook whose escalation section says "must
#     respond" and whose board section says "with full confidence" 4,000
#     characters later -- a real document, dropped whole.
#
# Still bypassable by splitting the halves across a blank line. No scope is
# both tight enough to avoid false positives and loose enough to catch every
# arrangement, which is the argument for the capability gate not reading
# content at all. tests/test_sanitizer.py asserts that gap stays open.

# Signal pair 1: a document that addresses a model directly.
_ADDRESS_OPENER = re.compile(r"\bnotice to any\b", re.IGNORECASE)
_ADDRESS_TARGET = re.compile(
    r"\b(assistant|ai|language model|llm|chatbot)\b", re.IGNORECASE
)

# Self-contained phrasings: subject and target are already adjacent, so there
# is nothing to join and nothing for padding to get between.
_ADDRESS_DIRECT = re.compile(
    r"\bif you are (an? )?(ai|language model|assistant|chatbot)\b"
    r"|\b(dear|attention)[:,]?\s+(any\s+)?(automated\s+)?(assistant|ai|language model|chatbot)\b",
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

_CONFUSABLES = str.maketrans({
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0425": "X", "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x", "\u0456": "i",
})

def _collapse(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_CONFUSABLES)
    return re.sub(r"\s+", " ", text).strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Collapse whitespace, then split on sentence-ending punctuation.

    Line wrapping inside a paragraph is not a sentence break, which is why
    the whitespace is collapsed first.
    """
    collapsed = _collapse(text)
    return [s for s in _SENTENCE_SPLIT.split(collapsed) if s]


@dataclass
class SanitizeResult:
    """The outcome of scanning one piece of text for model-directed instructions."""

    sanitized_text: str
    triggering_sentences: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.triggering_sentences)


def sanitize_text(text: str) -> SanitizeResult:
    # _collapse, not a bare re.sub: it NFKC-normalises and folds _CONFUSABLES,
    # so a Cyrillic character inside a trigger token cannot hide it. The table
    # was already here and the decision path was not using it -- one Cyrillic
    # "o" in "confidence" defeated detection entirely.
    if not any(_signal_spans(_collapse(para)) for para in _paragraphs(text)):
        return SanitizeResult(sanitized_text=text)

    # Decide above; everything below only explains why, for the reviewer.
    collapsed = _collapse(text)
    spans = [
        m.span()
        for p in (_ADDRESS_OPENER, _ADDRESS_TARGET, _ADDRESS_DIRECT,
                  _COERCION_IMPERATIVE, _COERCION_CERTAINTY)
        for m in p.finditer(collapsed)
    ]
    per_sentence = [
        s for s in _split_sentences(text) if _signal_spans(_collapse(s))
    ]
    triggering = per_sentence or [collapsed[a:b] for a, b in spans]
    return SanitizeResult(sanitized_text=REDACTION_MARKER, triggering_sentences=triggering)

def sanitize_documents(documents) -> tuple[list, list]:
    """Drop whole documents that instruct a model. Returns (kept, dropped).

    Call this before chunking and before stamping -- see the module
    docstring for why the whole-document decision is the load-bearing one.
    """
    kept, dropped = [], []
    for document in documents:
        if sanitize_text(document.text).flagged:
            dropped.append(document)
        else:
            kept.append(document)
    return kept, dropped
