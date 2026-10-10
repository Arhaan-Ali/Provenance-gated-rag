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

# Two patterns, both rare in genuine business writing. Kept narrow on
# purpose: "supersedes" and "must answer" on their own appear in ordinary
# policy and SLA text, so neither is a trigger by itself.
_AI_DIRECTED_ADDRESS = re.compile(
    r"\bnotice to any\b.{0,120}?\b(assistant|ai|language model|llm|chatbot)\b"
    r"|\bif you are (an? )?(ai|language model|assistant|chatbot)\b"
    r"|\b(dear|attention)[:,]?\s+(any\s+)?(automated\s+)?(assistant|ai|language model|chatbot)\b",
    re.IGNORECASE | re.DOTALL,
)

_ANSWER_COERCION = re.compile(
    r"\bmust (state|answer|respond)\b.{0,120}?\bwith full confidence\b"
    r"|\bwith full confidence\b.{0,120}?\bmust (state|answer|respond)\b",
    re.IGNORECASE | re.DOTALL,
)

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
    collapsed = re.sub(r"\s+", " ", text).strip()
    spans = [m.span() for p in (_AI_DIRECTED_ADDRESS, _ANSWER_COERCION)
             for m in p.finditer(collapsed)]
    if not spans:
        return SanitizeResult(sanitized_text=text)

    # Decide on the whole text. Sentences are only used to explain why.
    per_sentence = [s for s in _split_sentences(text)
                    if _AI_DIRECTED_ADDRESS.search(s) or _ANSWER_COERCION.search(s)]
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
