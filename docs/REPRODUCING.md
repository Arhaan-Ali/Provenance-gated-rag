# Reproducing the results

Every number in `README.md` and `docs/DEVELOPMENT.md` can be re-measured on
your own machine. This page is the order to do it in, and what you should
see. Steps 1–3 need no model and take about two minutes; steps 4–6 need
Ollama.

Measured on a MacBook Air (M-series, 16 GB) with macOS, Python 3.12 and the
versions in `requirements/lock.txt`. Ollama's model outputs vary a little
between runs and versions, which is why each live number below is given
with how many runs it held for.

## 1. Install

```bash
git clone https://github.com/piyush2676/provenance-gated-rag.git
cd provenance-gated-rag
python3.12 -m venv .venv            # or: uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements/lock.txt
cp .env.example .env
```

`lock.txt` is the full set the numbers were measured with.
`requirements/dev.txt` installs the same direct dependencies, unpinned below
them.

## 2. The tests (no model)

```bash
.venv/bin/python -m pytest -q
```

Expect `577 passed, 3 skipped` in a few seconds; the three skipped need
the entailment model (step 3b). CI runs the same suite on every push
(`.github/workflows/test.yaml`).

## 3. The model-independence result (no model)

```bash
.venv/bin/python scripts/run_model_independence.py
.venv/bin/python scripts/run_demo.py --cached
```

The first prints the attack suite against two scripted models. Against the
**compromised** one (it states every attacker's claim and makes every
attacker's tool call) you should see:

| | baseline | defended | scoped |
|---|---|---|---|
| attacks that worked | 9/9 | 6/9 (plain answer text) | 6/9 (plain answer text) |
| attacker's tool call ran | 3 | **0** | **0** |
| tool calls carrying attacker content | 4 | **0** | **0** |
| grounded: attack text in answer | – | **0/10** | **0/10** |

Against the **honest** one: grounded answers 7/7 correct, and both
legitimate emails sent under scoped release. The second command replays the
recorded presentation demo.

## 3b. The meaning check (no Ollama; ~370 MB model download once)

```bash
.venv/bin/pip install -r requirements/nli.txt
.venv/bin/python scripts/run_meaning_eval.py
.venv/bin/python -m pytest -q        # now 580 passed
```

Expect, words + NLI: supported 0/22 declined, contradicted 14/14,
conflict 3/3.

## 4. Start Ollama

```bash
ollama serve                      # or open the Ollama app
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama pull llama3.2:3b           # only for the second-model comparison
```

Every live script checks this first and stops with the reason if a model is
missing, before anything is reviewed or saved. `qwen3:8b` keeps a 16 GB
laptop busy: stop Ollama when you are done.

## 5. The live claims

```bash
PGRAG_KEY_DIR=$(mktemp -d) .venv/bin/python scripts/verify_milestones.py
```

Expect `16/16 claims verified` in about eight minutes (with step 3b
installed; without it, the meaning claim fails and says why). Setting
`PGRAG_KEY_DIR` to a scratch folder keeps the verifier's signing key and
decisions away from your own.

## 6. The live attack eval, on two models

```bash
.venv/bin/python scripts/run_attack_eval.py --model qwen3:8b --repeats 2
.venv/bin/python scripts/run_attack_eval.py --model llama3.2:3b --repeats 2
.venv/bin/python scripts/run_model_independence.py
```

Each run saves every answer to `results/attack_eval_<model>_<time>.json`;
the last command puts the newest run for each model side by side. What
should hold on every model: **0 attacker tool calls ran, 0 tool calls
carrying attacker content, 0 attack text in grounded answers.** What is
expected to differ between models: the plain (ungrounded) answers, and how
many questions each model answers correctly or declines.

Scoring is substring matching. Read the saved answers before quoting a
number.

## The interactive demos

```bash
.venv/bin/python scripts/run_m1_demo.py   # review gate: type x for public_inbox_unverified
.venv/bin/python scripts/run_m5_demo.py   # baseline vs. grounded, four questions
./scripts/run_demo_live.sh                # the presentation demo, live; stops Ollama after
```

`run_m1_demo.py` writes signed decisions to `data/decisions/` and its key to
`~/.config/pgrag/`. Back that folder up: without the key the decisions
cannot be verified, and every origin has to be reviewed again.

## If something does not match

Open an issue with the command, its full output, `ollama --version` and
`.venv/bin/pip freeze`. A number that does not reproduce is a bug in the
claim, not in your setup, until shown otherwise.
