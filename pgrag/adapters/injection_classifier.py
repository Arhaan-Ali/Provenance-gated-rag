"""Stage 2 of the sanitizer: a local model that scores how injection-like a text is.

pgrag.core.sanitizer.screen_text() takes any function `text -> score in [0, 1]`.
This is the real one. It lives in adapters/ because it needs torch and
transformers, which pgrag/core must never import. This module imports neither
until the model is first used: importing it, constructing an
InjectionClassifier, and running the tests (which use a fake pipeline) all
cost nothing.

It runs ON YOUR MACHINE. There is no API call and no text leaves the computer.
The first construction downloads the weights from the Hugging Face Hub into
the local cache (~/.cache/huggingface) -- that is the only network use. After
that, set the environment variable HF_HUB_OFFLINE=1 to forbid even that.

Models (verified against their model cards, October 2026):
  protectai/deberta-v3-base-prompt-injection-v2   the default
      Apache-2.0, ungated, ~0.2B parameters. The card says it is English-only
      and "does not detect jailbreak attacks or handle non-English prompts",
      and it is marked archived. A Hindi injection will not be reliably
      caught by it.
  meta-llama/Llama-Prompt-Guard-2-86M             the multilingual option
      mDeBERTa-based, 86M parameters, advertised as multilingual across 8
      languages (the list is not on the pages we read -- check the card for
      Hindi before relying on it). GATED under the Llama 4 Community License:
      accept the terms on the model page and log in with `hf auth login`
      first. Labels are generic LABEL_0 / LABEL_1, where LABEL_1 = malicious.
      Its 22M sibling is English-only.
  Pass either as `model=`. Nothing here has been run against the real
  weights in this repository's tests; the tests use a fake pipeline.

Long text: these models read 512 tokens at most and silently cut the rest.
An attacker controls how long a document is, so a single pass would let a
directive sit past the cut. The text is therefore scored in overlapping
windows and the HIGHEST window score is returned: one bad window is enough.
"""
from __future__ import annotations

from typing import Any, Callable

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

# Pinned so the weights cannot change underneath us. Taken from llm-guard's
# own pin for the default model; other models run unpinned unless you pass one.
_PINNED_REVISIONS = {
    DEFAULT_MODEL: "89b085cd330414d3e7d9dd787870f315957e1e9f",
}

# Labels that mean "injection", compared upper-case. protectai says INJECTION,
# the gated Prompt Guard repos publish LABEL_1 (malicious), some copies say
# MALICIOUS. Anything else is read as the benign class.
POSITIVE_LABELS = frozenset({"INJECTION", "LABEL_1", "MALICIOUS"})


def windows(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Overlapping slices that together cover every character of text.

    Overlap so a directive straddling a boundary is whole in at least one
    window. 512 characters stays well inside the 512-token limit for English
    and typical non-English text; it is a starting point, not a tuned value.
    """
    if size <= overlap:
        raise ValueError("window size must be larger than the overlap")
    if len(text) <= size:
        return [text]
    step = size - overlap
    out = [text[start:start + size] for start in range(0, len(text) - overlap, step)]
    return out


class InjectionClassifier:
    """Callable: text -> highest injection score over its windows, in [0, 1].

    Constructing one is free: the model is NOT loaded until the first time it
    is actually asked to score something. Because screen_text() consults the
    classifier only for text the patterns did not flag, a run in which every
    text trips a pattern never loads the model at all, and one in which
    nothing does pays the load cost once, on the first query.
    """

    def __init__(
            self,
            model: str = DEFAULT_MODEL,
            revision: str | None = None,
            *,
            window_chars: int = 512,
            overlap_chars: int = 64,
            batch_size: int = 16,
            positive_labels: frozenset[str] = POSITIVE_LABELS,
            _pipeline: Any = None,
            _loader: Callable[[], Any] | None = None,
    ) -> None:
        self.model = model
        self.revision = revision or _PINNED_REVISIONS.get(model)
        self._window = window_chars
        self._overlap = overlap_chars
        self._batch = batch_size
        self._positive = frozenset(label.upper() for label in positive_labels)
        self._pipe = _pipeline
        # _pipeline / _loader exist so tests can supply a fake; real use passes neither.
        self._loader = _loader or (lambda: _load_pipeline(model, revision))

    @property
    def describe(self) -> str:
        """Which weights this is, e.g. for recording what screened an index."""
        return f"{self.model}@{self.revision or 'unpinned'}"

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def _pipeline_ready(self) -> Any:
        if self._pipe is None:
            self._pipe = self._loader()
        return self._pipe

    def __call__(self, text: str) -> float:
        if not text.strip():
            return 0.0
        results = self._pipeline_ready()(
            windows(text, self._window, self._overlap), batch_size=self._batch)
        return max(self._injection_score(r) for r in results)

    def _injection_score(self, result: dict) -> float:
        # The pipeline reports the top label and its probability. If the top
        # label is the injection class that probability is the score; if it
        # is the benign class, the injection probability is what is left.
        if str(result["label"]).upper() in self._positive:
            return float(result["score"])
        return 1.0 - float(result["score"])


def _load_pipeline(model: str, revision: str | None) -> Any:
    from transformers import pipeline   # heavy; imported only when really used

    return pipeline(
        task="text-classification",
        model=model,
        revision=revision or _PINNED_REVISIONS.get(model),
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
    )