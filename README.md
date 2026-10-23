# provenance-gated-rag

A retrieval layer that keeps working when some of its documents are hostile — by
tracking where every chunk came from, and limiting what the system may do when
untrusted content is in context. Built on top of [LlamaIndex](https://github.com/run-llama/llama_index).

This project raises the cost of document-borne prompt injection and fails safe
when it cannot tell. It does not claim to prevent prompt injection outright.

## Setup

```bash
pyenv local 3.12.6
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
pip install -r requirements/classifier.txt  # optional: the sanitizer's local classifier stage (torch)
cp .env.example .env  # LLM_PROVIDER=ollama, model qwen3:8b by default
ollama pull qwen3:8b  # requires Ollama (https://ollama.com) installed and running
ollama pull nomic-embed-text  # the embedding model
```

Dependencies are pinned exactly. The claims below are specific numbers
measured against these versions, so a dependency moving underneath them
risks the results no longer being reproducible rather than merely a broken
install. To reproduce the exact environment the numbers were measured in,
install `requirements/lock.txt` instead -- it carries the full transitive
set. `requirements/base.txt` and `dev.txt` are the maintained ones.

```bash
python scripts/run_demo.py --cached   # the three-act demo, no model needed
python scripts/run_demo.py            # the same, live against Ollama
./scripts/run_demo_live.sh            # live, starting Ollama first and stopping it after
python scripts/run_m5_demo.py         # grounding: baseline vs. grounded answers
python scripts/run_attack_eval.py     # 9 attacks: baseline vs. defended vs. scoped
python scripts/run_model_independence.py  # M6: the result against a model that obeys every attack
python scripts/run_meaning_eval.py    # M5: contradictions caught by meaning (needs requirements/nli.txt)
python scripts/verify_milestones.py   # re-check all 16 claims (~8 min)
pytest                                # 580 tests; 3 need requirements/nli.txt (CI runs the rest)
```

`docs/REPRODUCING.md` walks through reproducing every number, in order.

The live scripts load `qwen3:8b`, which keeps a 16 GB laptop busy. Stop
Ollama after a live run (`run_demo_live.sh` does this for you).

**First run after upgrading:** decisions written before signing existed are
not trusted, so every origin is reviewed once more. Back up
`~/.config/pgrag/review.key`: without it the signed decisions cannot be
verified and every origin has to be reviewed again.

## Use it as a library

Install it into your own project. The core needs no model weights; the
extras add the optional models.

```bash
pip install git+https://github.com/piyush2676/provenance-gated-rag.git
pip install "provenance-gated-rag[nli] @ git+https://github.com/piyush2676/provenance-gated-rag.git"
```

(`nli` adds the meaning check, `classifier` the sanitizer's model stage;
both pull torch.) Then, from your project folder:

```text
my-project/
  trust.yaml          which source folder gets which trust tier (written for you on first run)
  corpus/
    wiki/             one subfolder per source: .txt, .md, .json, .pdf
    inbox/
```

```python
from provenance_gated_rag.ingest import build_query_engine

engine = build_query_engine()     # or build_query_engine("corpus", "decisions", "trust.yaml")
answer = engine.query("Who is the CEO?")
print(answer)                      # the answer with citations, or why it was declined
```

The first run shows each source folder on the review screen and asks you
to approve it; your signed decisions go to `decisions/`, and the signing
key to `~/.config/pgrag/` (back it up). Ollama must be running with the
models in `.env` (`qwen3:8b` and `nomic-embed-text` by default). For the
meaning check, pass `judge=EntailmentJudge()` from `provenance_gated_rag.nli`.

The package's dependencies are compatible ranges; `requirements/lock.txt`
is the exact set the numbers below were measured with.

## Status

M0 done: `python scripts/run_m0_demo.py` reproduces a plain RAG pipeline
confidently answering a planted-document prompt injection.

M1 done: `python scripts/run_m1_demo.py` shows the origin registry
computing a trust tier per origin (a 4-tier A-D provenance scale --
A = user authored, D = external/unverified/default -- configured via
`trust.yaml` at the project root) and a review gate that lets a human
confirm, override, or exclude each origin before any embedding happens.
Excluding the low-tier planted-document origin fixes M0's wrong answer
entirely, since the poisoned content is never indexed.

Each origin also declares an `editability` in `trust.yaml` -- who is able
to change the source. `org-wide` caps its tier at C, `anonymous` at D, and
an origin that declares nothing is treated as `anonymous`, since silence is
not evidence that few people can edit it. The cap only demotes. It fires on
this project's own rubric: `verified_internal_wiki` claims B while its own
reason describes it as staff-editable, so it runs at C and the review
screen shows `Capped: B -> C: editable by org-wide`.

Drift re-review (design doc Feature 2) is now implemented: `needs_review`
re-triggers the gate for a new origin, a decision older than 90 days, a
computed tier that moved in `trust.yaml`, or a change to an origin's
declared editability, and carries every settled decision forward untouched. A reviewer's override is deliberate, so a
computed tier that disagrees with it is not treated as drift. Running the
M1 demo twice shows the second run reusing both decisions without
prompting at all.

M2 done: `pytest` (`tests/test_label_propagation.py`) asserts that every
node retrieved through M1's pipeline still carries the correct `origin`
and `trust_tier` metadata, and asserts that M0's stock pipeline has none
of it at all -- a permanent, always-passing regression test for both.

M3 done: `python scripts/run_m3_demo.py` computes the allowed tool set
(`send_email`, `lookup_fact`) from the minimum trust tier present in
retrieved context, independent of any LLM. With the gate off, the model
is persuaded by a planted document to email real internal details to an
attacker-controlled address; with the gate on, `send_email` is never
offered to the model for that turn (one `D`-tier chunk in context is
enough to withhold it), so the identical persuasion attempt achieves
nothing. This reimplements the milestone that was previously reverted --
the original version referenced the old 6-tier scale (`TrustTier.F`),
which no longer exists after the collaborator's A-D refactor above; the
corpus and gate logic are otherwise unchanged from that earlier attempt.

Adversarial eval (`python scripts/run_adversarial_eval.py`) tests the
stock M0 baseline across four different attack scenarios -- executive
identity, funding amount, HQ location, and a product safety recall,
each a different fictional company -- to check whether the vulnerability
generalizes rather than being one cherry-picked example. Result: the
stock baseline was successfully attacked in 3 of 4 scenarios (75%). The
one resisted case (a fabricated safety recall) suggests the model
applies more scrutiny to safety/harm-adjacent claims than to neutral
facts like a name, a dollar figure, or a city -- a real, honest limit on
how far the vulnerability generalizes, not a universal failure.

M4 done: `src/provenance_gated_rag/sanitizer.py` detects chunks that
explicitly address a model ("NOTICE TO ANY AUTOMATED ASSISTANT...") or
coerce a specific answer ("must state X with full confidence") and
neutralises the entire document, not just the triggering sentence: its text
is replaced with a redaction marker, and it stays in context with its tier,
so an untrusted document still lowers what the capability gate allows. Wired in as a
`node_postprocessor` (`scripts/run_m4_demo.py`, and as a toggle in
`run_adversarial_eval.py`), it takes the same adversarial eval from
**3/4 (75%) attacks succeeding down to 0/4 (0%)** -- the CEO-identity,
funding, and HQ attacks all flip from wrong to correct, and the recall
scenario (already resistant at baseline) stays correct. False-positive
cost measured directly against every genuine document already in the
repo (17 files) plus 9 hand-crafted benign lookalikes designed to share
surface words with the trigger patterns ("AI-powered assistant",
"supersedes the 2019 version", "must answer... with full confidence" in
an unrelated SLA sentence) -- zero false positives across all 26.
Known, deliberate gap: a subtler injection with no AI-facing framing at
all (M3's `planted_tool_injection.txt`) has nothing for a text sanitizer
to key on and is not caught here -- that attack is stopped by M3's
capability gate refusing the resulting action instead, so the two
milestones are complementary rather than redundant.

Hardening ported from the collaborator's fork (all with tests ported or
added alongside):

- **A review approves files, not folders.** Each decision records the
  sha256 of every file the reviewer was shown. A file added to, edited in,
  or removed from an approved origin re-triggers the review, and
  `load_documents` indexes only approved files whose contents still match
  -- so a document planted after the review is never indexed on the
  strength of an earlier approval.
- **Decisions are signed.** Every `decisions_vN.json` carries an
  HMAC-SHA256 signature and the previous file's signature, with the key
  and the newest signature kept outside the project (`~/.config/pgrag/`,
  or `$PGRAG_KEY_DIR`). A forged, edited, deleted or rolled-back decisions
  file stops the run. Unsigned files from before this are not trusted:
  every origin is reviewed once more.
- **Sanitizer stage 2 and Unicode normalisation.** Look-alike Cyrillic
  letters and invisible characters no longer hide a trigger. An optional
  local classifier (`protectai/deberta-v3-base-prompt-injection-v2`, pinned)
  scores whatever the patterns pass; pass `classifier=InjectionClassifier()`
  to `build_query_engine`. The sanitizer's flag no longer reaches the prompt.
- **The capability gate is enforced, not just computed.**
  `gated_turn.run_gated_turn` offers only the allowed tools and checks every
  tool call again before running it, so a model that asks for a tool it was
  never offered is refused.
- **CI.** `.github/workflows/test.yaml` runs the suite on every push and PR.
  The tests that used to need Ollama now use a mock embedding.

Further fixes made while porting (each with a test):

- **The reviewer sees what they approve.** The review screen previews the
  files in each origin, `[f]` lists every file, and when the computed tier
  has *risen* since the last review, Enter keeps the old tier: the higher
  letter has to be typed.
- **No race between approval and indexing.** Approved files are read once,
  hashed, and the copy that matched is what gets indexed, so a file swapped
  in between the check and the read is never indexed.
- **The decisions folder can move.** Each trail carries a `trail_id`, so the
  newest signature is still found after the project is moved or renamed. A
  missing key now gives an error that says how to recover.
- **Scoped release.** `run_gated_turn(..., scoped=recipient_allowlist(...))`
  lets `send_email` run with an untrusted source in context, but only to
  listed addresses, and only if no name, number, address or link in the
  call appears solely in an untrusted source. The second rule was added
  after `qwen3:8b` sent an allowed recipient an email whose body was the
  planted code "DEV-7731"; that email is now refused.
- **Only `.txt`, `.md`, `.json` and `.pdf` are read.** Other files are
  listed on the review screen as skipped and never indexed.
- **The run says what it caught.** Each neutralised file is named at
  ingestion, and both query engines screen retrieved chunks again.
- **Live scripts check Ollama first**, and stop before the review screen
  with the command that fixes a missing server or model.
- **Network tools can never run with untrusted content in context.** Only
  tools declared local and read-only may be allowed at tier D; a tier table
  that breaks this stops the program at import.
- **A person can approve each scoped send.** `run_gated_turn(...,
  approve=ask_on_terminal)` shows a released call exactly as it would run
  and runs it only on `y`.
- **`trust.yaml` is recorded with each signed review.** If it changed since
  the last one, a *new* origin it gives a tier above D needs that tier
  typed; Enter keeps it at D.

M5 done: `src/provenance_gated_rag/grounding.py` lets an answer stand only
if trusted sources support it, and otherwise declines and says why
(`python scripts/run_m5_demo.py`). Two deterministic checks sit on either
side of the model call. Before it, chunks below tier C and neutralised
chunks are set aside, so an unverified source cannot support a stated fact
even when it carries no instruction for M4 to catch. If nothing trusted is
left, the model is never called. After it, every sentence is checked
against the trusted source it cites (or, if it cites none, against a single
trusted source): its names and numbers must appear together within two
sentences of that source, and most of its other words must appear there too.
So an answer that repeats an injection, invents a figure, swaps two
people's roles or says something the source does not ("Elena Cho resigned")
is declined, however the model phrased it. So is a sentence that negates
what its source states, or states what the source negates ("Elena Cho does
not lead ..."). `build_query_engine` is now this grounded engine by default
(`build_plain_query_engine` is the ungrounded one, kept for the milestone
comparisons): when the model says the sources do not answer, it retries
once with twice as many chunks before declining.

Measured with `qwen3:8b`, with the planted origin *accepted* rather than
excluded: the M0 baseline answered "Marcus Webb" for the CEO; the grounded
pipeline answered "Elena Cho" with citations, declined the two questions the
corpus cannot answer (a COO, a product price) with a reason, and answered
the control question (the CTO). Across the 9-attack eval
(`scripts/run_attack_eval.py`, 2 repeats):

| | Attack text in answer | Attacker's tool call ran | Legit emails sent |
|---|---|---|---|
| Baseline | 3/9 | 1 | -- |
| Defended (M1-M4) | 3/9 | 0 | 0/2 |
| Defended + grounded (M5) | **0/10** (7/7 correct, 2 declined) | 0 | -- |
| Scoped release, 2026-10-22 | -- | 0 | 2/2 (one body carried planted text) |
| Scoped release, with the body check (2026-10-23) | -- | 0 | 1/2 (the planted one refused) |

**Meaning and disagreeing sources.** Two further checks cover what words
cannot. An optional local entailment model (`nli.py`,
`requirements/nli.txt`) declines a sentence its source contradicts ("Elena
Cho **left**" against "Elena Cho **leads**"), or that another trusted source
contradicts. A fact table (`facts.py`) reads who holds which role, the
headquarters and the founding year from every trusted document, reports
any two that disagree at ingestion, and declines an answer touching a
disputed fact, showing both sources. On `data/meaning_eval.json`: words
alone caught 5/14 contradicting claims, words + the model 14/14, with 0/22
correct answers declined (13 of them real `qwen3:8b` answers), and 3/3
conflicts between trusted sources. Live with the model on, both models
still grounded 7/7 correct with 0/10 attack text.

`python scripts/verify_milestones.py` checks 16 claims: the 15 live ones
passed **15/15** in one run against `qwen3:8b` (2026-10-23, about 8
minutes), and the meaning claim passes with `requirements/nli.txt`.

Honest notes: on the two unanswerable questions the baseline did not invent
anything either -- `qwen3:8b` said "not mentioned" by itself -- so the
hallucination M5 prevents here is the injected one. The check is strict on
purpose and can decline a correct answer: one unsupported sentence declines
the whole answer. Each false decline seen against the real model (a missing
citation, "Additionally" read as a name, "CEO" vs. "Chief Executive
Officer", a paraphrase, a sentence about where a fact is documented) was
fixed and kept as a regression test.

M6 done: the security result does not depend on the model
(`python scripts/run_model_independence.py`, `src/provenance_gated_rag/attack_suite.py`).
The same nine attacks were run against a scripted **compromised** model,
which states every attacker's claim and makes every attacker's tool call
whether or not it was offered, and against two real models of different
families, `qwen3:8b` and `llama3.2:3b` (2 runs each, identical both times):

| | compromised | qwen3:8b | llama3.2:3b |
|---|---|---|---|
| Attacks that worked, no defences | 9/9 | 4/9 | 2/9 |
| Attacker's tool call ran, defended | **0** | **0** | **0** |
| Tool calls carrying attacker content, defended | **0** | **0** | **0** |
| Attack text in grounded answers | **0/10** | **0/10** | **0/10** |
| Grounded answers correct | 0/7 (all declined) | 7/7 | 7/7 |
| Attack text in plain (ungrounded) answers, defended | 6/9 | 3/9 | 1/9 |

What holds for the compromised model holds for any model: the tool gate,
the scoped-release checks and grounding do not ask the model to resist
anything. What differs between models is everything the model decides for
itself: how many attacks land without the defences, and how much attack
text reaches an answer that is not grounded. So the answer a user should
rely on is the grounded one. Scoped release sent 1/2 legitimate emails with
`qwen3:8b` (the other carried "DEV-7731" and was refused) and 0/2 with
`llama3.2:3b`, which addressed both to the attacker's address and was
refused. One `llama3.2:3b` call that did run carried an attacker URL into
`search_corpus`, a local search allowed at every tier: nothing left the
machine, but the model was steered. The compromised-model result is a CI
test (`tests/test_model_independence.py`), so it is re-checked on every push.

M7 in progress: a clean-room install of exactly the committed files, from
`requirements/lock.txt`, passes the suite and reproduces the cached demo
and the M6 tables. `docs/REPRODUCING.md` gives the steps and the expected
output. What M7 asks for that has not happened yet: someone else doing it
on their own machine.

Test coverage was audited and closed before proceeding further: prior to
this, `run_review_gate`'s override path and invalid-input retry loop,
`save_decisions`'s versioning, `discover_origins`, and `load_documents`'s
exclusion path had only ever been exercised manually via demo runs, never
by an automated test. `tests/test_registry.py` and `tests/test_ingest.py`
closed all of it, running near-instantly since none of them need an LLM
or embedding model.

A collaborator (Arhaan Ali) contributed a refactor of the trust-tier
system -- from the original 6-tier Admiralty scale (A-F) down to a
simpler 4-tier provenance-type scale (A-D), plus moving the origin rubric
out of hardcoded Python and into a committed `trust.yaml` config file.
This was reviewed, integrated (fixing the resulting breaks in `ingest.py`
and all three test files), and extended with new tests for the two
genuinely new capabilities it added (`load_latest_decisions`,
`tier_for_document`) before merging -- see `docs/DEVELOPMENT.md` for the
full integration notes. 29 tests passed as of this integration; 580 pass
now, at 97% line coverage, with no model needed.

## Milestones

- [x] M0 — A baseline you can break: plain RAG giving a confident wrong answer
      from one planted document
- [x] M1 — Origin registry + review gate before indexing runs
- [x] M2 — Label propagation: a passing assertion that labels reach the
      generator intact (fails on a stock pipeline)
- [x] M3 — Capability gate: the M0 attack still convincing the model, and
      still achieving nothing
- [x] M4 — Instruction sanitizer: attack rate down, false-positive cost measured
- [x] M5 — Grounding and abstention: baseline hallucinating where this
      declines and explains why
- [x] M6 — Evaluation: the model-independence result
- [ ] M7 — Release: someone else installing it and reproducing the numbers

## Licence

MIT. See `LICENSE`.
