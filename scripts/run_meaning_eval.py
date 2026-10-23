"""Does M5 catch claims that contradict their source, without declining good answers?

Runs data/meaning_eval.json through M5's grounding check twice -- with the
lexical checks alone, then with the entailment model (nli.py) added -- and
reports, for each set:

  supported     must stand. Real qwen3:8b answers plus paraphrases. A decline is a false one.
  contradicted  must be declined. Antonyms, flipped numbers, swapped facts.
  conflict      two trusted sources disagree; the claim agrees with one. Must be declined.

No Ollama: the claims are fixed text. The entailment model needs
requirements/nli.txt; without it only the lexical column is filled in.

    python scripts/run_meaning_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core.schema import NodeWithScore, TextNode

from provenance_gated_rag.attack_suite import BENIGN
from provenance_gated_rag.grounding import answer_with_grounding, check_grounding

ROOT = Path(__file__).resolve().parents[1]


def source_text(name: str) -> str:
    if name.startswith("suite/"):
        return BENIGN[name.removeprefix("suite/")]
    return (ROOT / "data" / name).read_text(encoding="utf-8")


def node(text: str, where: str) -> NodeWithScore:
    origin, _, file = where.rpartition("/")
    return NodeWithScore(node=TextNode(text=text, metadata={
        "origin": origin or "eval", "source_file": file, "trust_tier": "B"}), score=1.0)


class Says:
    """An LLM that answers with the claim under test."""

    def __init__(self, text):
        self.text = text

    def complete(self, _prompt):
        return type("Completion", (), {"text": self.text})()


def declined(claim: str, sources, judge) -> tuple[bool, str]:
    problems = check_grounding(claim, sources, judge)
    return bool(problems), (problems[0] if problems else "")


def run(judge) -> dict:
    data = json.loads((ROOT / "data" / "meaning_eval.json").read_text(encoding="utf-8"))
    out = {}
    for kind in ("supported", "contradicted"):
        rows = []
        for where, claim in data[kind]:
            hit, why = declined(claim, [node(source_text(where), where)], judge)
            rows.append({"source": where, "claim": claim, "declined": hit, "why": why})
        out[kind] = rows
    rows = []
    for where, other_text, claim in data["conflict"]:
        sources = [node(source_text(where), where), node(other_text, "eval/disagreeing_source.txt")]
        result = answer_with_grounding("q", sources, Says(claim), judge=judge)
        rows.append({"source": where, "claim": claim, "declined": result.abstained,
                     "why": " ".join(result.reasons[1:2])})
    out["conflict"] = rows
    return out


def summary(results: dict) -> dict:
    return {kind: f"{sum(r['declined'] for r in rows)}/{len(rows)} declined" for kind, rows in results.items()}


def main() -> None:
    columns = {"words only": run(None)}
    try:
        from provenance_gated_rag.nli import EntailmentJudge

        judge = EntailmentJudge()
        columns["words + NLI"] = run(judge)
        columns["words + NLI"]["_model"] = judge.describe
    except ImportError as e:
        print(f"Entailment model not installed ({e}); install requirements/nli.txt for the second column.")

    print(f"\n{'':14}" + "".join(f"{name:>22}" for name in columns))
    for kind, should in (("supported", "should be 0"), ("contradicted", "should be all"),
                         ("conflict", "should be all")):
        cells = "".join(f"{summary({kind: c[kind]})[kind]:>22}" for c in columns.values())
        print(f"{kind:14}{cells}   ({should})")

    last = list(columns.values())[-1]
    for kind, wanted in (("supported", False), ("contradicted", True), ("conflict", True)):
        misses = [r for r in last[kind] if r["declined"] != wanted]
        for r in misses:
            print(f"\n  {'FALSE DECLINE' if not wanted else 'MISSED'} ({kind}): {r['claim']}\n    {r['why']}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "meaning_eval.json").write_text(
        json.dumps(columns, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
