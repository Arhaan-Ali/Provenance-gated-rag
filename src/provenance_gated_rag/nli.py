"""A local entailment model: does a source passage support, or contradict, a claim?

M5's other checks are lexical. They hold a sentence to the names, numbers
and words of its source, and to its "not" -- but "Elena Cho left Meridian
Robotics" against "Elena Cho leads Meridian Robotics" shares almost every
word and has no "not" to catch. An NLI (natural language inference) model
reads the pair and answers entailment, neutral or contradiction; that one
scores 1.00 contradiction.

Optional, like the sanitizer's classifier: it needs torch and transformers
(requirements/nli.txt), which nothing else imports. Importing this module
and constructing an EntailmentJudge cost nothing; the model is loaded the
first time it is asked about a pair. The tests use a fake.

It runs ON YOUR MACHINE. The first load downloads the weights (~370 MB)
into ~/.cache/huggingface; set HF_HUB_OFFLINE=1 afterwards to forbid even
that.

Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, MIT licence, trained
on MNLI, FEVER and ANLI -- the last is adversarial NLI, written to break
models, which is the job here. Pinned to a revision so the weights cannot
change underneath the thresholds in grounding.py.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_PINNED_REVISIONS = {
    DEFAULT_MODEL: "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
}


@dataclass(frozen=True)
class Judgement:
    """Probabilities for one (premise, claim) pair; they sum to 1."""

    entailment: float
    neutral: float
    contradiction: float


class EntailmentJudge:
    """Callable: [(premise, claim), ...] -> [Judgement, ...], in the same order.

    The model is not loaded until the first call, so a run whose answers
    the lexical checks already decline never loads it.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        revision: str | None = None,
        *,
        batch_size: int = 16,
        _loader: Callable[[], Any] | None = None,
    ) -> None:
        self.model = model
        self.revision = revision or _PINNED_REVISIONS.get(model)
        self._batch = batch_size
        self._scorer = None
        # _loader exists so tests can supply a fake; real use passes none.
        self._loader = _loader or (lambda: _load_scorer(self.model, self.revision))

    @property
    def describe(self) -> str:
        return f"{self.model}@{self.revision or 'unpinned'}"

    @property
    def loaded(self) -> bool:
        return self._scorer is not None

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> list[Judgement]:
        if not pairs:
            return []
        if self._scorer is None:
            self._scorer = self._loader()
        out: list[Judgement] = []
        for start in range(0, len(pairs), self._batch):
            out += self._scorer(list(pairs[start:start + self._batch]))
        return out


def _load_scorer(model: str, revision: str | None) -> Callable[[list[tuple[str, str]]], list[Judgement]]:
    import torch  # heavy; imported only when really used
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    network = AutoModelForSequenceClassification.from_pretrained(model, revision=revision).eval()
    # Read the label order from the model, never assume it: models disagree.
    index = {label.lower(): i for i, label in network.config.id2label.items()}
    missing = {"entailment", "neutral", "contradiction"} - set(index)
    if missing:
        raise ValueError(f"{model} has no {', '.join(sorted(missing))} label; it is not an NLI model")

    def score(pairs: list[tuple[str, str]]) -> list[Judgement]:
        encoded = tokenizer(
            [premise for premise, _ in pairs],
            [claim for _, claim in pairs],
            return_tensors="pt",
            padding=True,
            truncation="only_first",   # cut the premise, never the claim being judged
            max_length=512,
        )
        with torch.no_grad():
            probs = network(**encoded).logits.softmax(-1).tolist()
        return [
            Judgement(p[index["entailment"]], p[index["neutral"]], p[index["contradiction"]])
            for p in probs
        ]

    return score
