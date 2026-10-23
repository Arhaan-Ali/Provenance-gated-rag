"""M6: does the defence depend on which model is behind it?

Two parts.

1. Scripted models, no Ollama needed. The same attack suite as
   run_attack_eval.py, with every attack in context for every question, run
   against two stand-in models:

     compromised - states every attacker's claim and makes every attacker's
                   tool call, offered or not. The worst any model could do.
     honest      - what a correct model would say and do.

   What holds against the compromised model holds for every model. What the
   honest model still gets done is the cost of the defences.

2. Live models. Every results/attack_eval_<model>_*.json written by
   run_attack_eval.py, newest per model, side by side:

     python scripts/run_attack_eval.py --model qwen3:8b --repeats 2
     python scripts/run_attack_eval.py --model llama3.2:3b --repeats 2
     python scripts/run_model_independence.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance_gated_rag.attack_suite import COMPROMISED, HONEST, run_scripted

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ROWS = (
    ("attacks that worked (answer or tool)", lambda s: s.get("attacks_that_worked", "-")),
    ("attack text in the plain answer", lambda s: s["attack_reached_answer"] if "attack_reached_answer" in s
                                                  else s["attack_in_answer"]),
    ("attacker's tool call ran", lambda s: s["attacker_tool_call_ran"]),
    ("tool calls carrying attacker content", lambda s: s["calls_carrying_attacker_content"]
                                                       if "calls_carrying_attacker_content" in s
                                                       else len(s["tool_calls_carrying_attacker_content"])),
    ("legitimate emails sent", lambda s: f"{sum(bool(v) for v in s['legit_tool_ran'].values())}"
                                         f"/{len(s['legit_tool_ran'])}"),
    ("grounded: attack text in answer", lambda s: s.get("grounded", {}).get("attack_reached_answer", "-")),
    ("grounded: correct", lambda s: s.get("grounded", {}).get("correct", "-")),
    ("grounded: declined", lambda s: s.get("grounded", {}).get("declined", "-")),
)


def table(title: str, columns: dict[str, dict]) -> None:
    print(f"\n{title}\n")
    names = list(columns)
    width = max(len(r) for r, _ in ROWS) + 2
    print(" " * width + "".join(f"{n:>22}" for n in names))
    for label, get in ROWS:
        cells = []
        for n in names:
            try:
                cells.append(str(get(columns[n])))
            except KeyError:
                cells.append("-")
        print(f"{label:<{width}}" + "".join(f"{c:>22}" for c in cells))


def _grounded_key(summary: dict) -> dict:
    # The live eval names it attack_reached_answer, the scripted one attack_in_answer.
    g = summary.get("grounded")
    if g and "attack_in_answer" in g:
        g = {**g, "attack_reached_answer": g["attack_in_answer"]}
    return {**summary, "grounded": g} if g else summary


def scripted() -> dict:
    out = {}
    for name, scripts in (("compromised", COMPROMISED), ("honest", HONEST)):
        with tempfile.TemporaryDirectory() as tmp:
            out[name] = run_scripted(Path(tmp), scripts)
    for name, modes in out.items():
        table(f"Scripted model: {name}",
              {mode: _grounded_key(body["summary"]) for mode, body in modes.items()})
    return out


def live() -> None:
    newest: dict[str, Path] = {}
    for path in sorted(RESULTS_DIR.glob("attack_eval_*.json")):
        model = json.loads(path.read_text(encoding="utf-8"))["model"]
        newest[model] = path   # sorted by timestamp in the name, so the last wins
    if not newest:
        print("\nNo live results yet: run scripts/run_attack_eval.py --model <name> first.")
        return
    for mode in ("baseline", "defended", "scoped"):
        columns = {}
        for model, path in newest.items():
            runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
            for i, run in enumerate(runs, start=1):
                attacks = [r for r in run[mode]["results"] if r["stopped_by"] is not None]
                worked = sum(r["attack_in_answer"] or r["attack_tool_ran"] for r in attacks)
                columns[f"{model} #{i}"] = {**run[mode]["summary"],
                                            "attacks_that_worked": f"{worked}/{len(attacks)}"}
        table(f"Live models, {mode}", columns)
    print("\nFrom: " + ", ".join(p.name for p in newest.values()))


def main() -> None:
    out = scripted()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "model_independence_scripted.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    live()


if __name__ == "__main__":
    main()
