# Provenance-gated-rag

A retrieval layer that keeps working when some of its documents are
hostile -- by tracking where every chunk came from, and limiting what a
system is allowed to do when untrusted content is in context.

This does not claim to prevent prompt injection outright. It raises the
cost of document-borne prompt injection and fails safe when it cannot
tell.

## Status

The registry and the demo are wired together. `demo/run.py` reviews every
origin before indexing, stamps the approved tier onto each document, and
drops the ones that must not be indexed.

| Piece | State |
|---|---|
| `pgrag/core/registry.py` | Trust tiers, editability caps, review gate, review decay, versioned audit trail |
| `pgrag/cli.py` | The review gate's human-facing screen |
| `pgrag/adapters/llamaindex.py` | Stamps tiers onto documents, hides them from the prompt, re-stamps after transforms |
| `pgrag/core/capability_gate.py` | Derives the allowed tool set from the weakest tier in context |
| `pgrag/core/sanitizer.py` | Drops whole documents that instruct the model, before chunking |
| `demo/run.py` | Interactive RAG demo, running the full review-and-stamp pipeline |
| `tests/` | 234 tests across the registry, review gate, adapter, capability gate and sanitizer |

## Project structure

```
pgrag/
  core/
    registry.py        -- tiers, editability caps, review gate, decay, audit trail
    capability_gate.py -- allowed tools, derived from the weakest tier present
    sanitizer.py       -- drops documents that tell the model how to answer
  adapters/
    llamaindex.py      -- stamp_documents, restamp_nodes, to_nodes,
                          InstructionSanitizerPostprocessor
  cli.py               -- the review gate screen
demo/
  run.py               -- interactive RAG demo (Ollama + llama-index)
  tools.py             -- mock tools for the capability gate
tests/                 -- 234 tests
Requirements.txt       -- pinned exactly, see the note in the file
```

## Trust tiers

Four tiers, `A` (highest trust) down to `D` (lowest, and the default for
anything unlisted):

| Tier | Meaning |
|---|---|
| A | User authored -- you wrote it yourself |
| B | Internal documents -- from an internal system, others can edit it |
| C | External verified -- from outside, but signature/TLS/domain checked |
| D | External un-verified -- everything else |

The rubric lives in `trust.yaml`, which `load_rules()` creates from a
commented starter template on first run. Any origin not listed defaults to
`D`, so a new or unknown source is never trusted by accident.

### Editability caps the tier

A tier says how much you trust what a source says today. It says nothing
about how easily that can change tomorrow without anyone re-reviewing it,
so each origin also declares who can edit it:

| `editability` | Cap applied |
|---|---|
| `immutable` | none |
| `authenticated-few` | none |
| `org-wide` | tier capped at `C` |
| `anonymous` | tier capped at `D` |
| not declared | treated as `anonymous`, capped at `D` |

Silence is not evidence that few people can edit a source, which is why an
undeclared origin is capped as hard as an anonymous one. The cap only ever
demotes, and it records a note the review screen shows -- for example
`B -> C: editable by org-wide`.

### Decisions expire

`needs_review(origin, previous, now)` returns why an origin must be shown
to a human again, or `None` to reuse the settled decision. It re-triggers
on a new origin, a decision with no timestamp, a decision older than
`REVIEW_MAX_AGE_DAYS` (90), a changed `editability`, or a changed computed
tier.

A reviewer's override is deliberate, so a computed tier that disagrees with
one is not treated as drift -- otherwise every run would ask again about a
decision already made. An overridden decision does still age out, so an
override is not permanent either.

## Key functions

**`pgrag/core/registry.py`**

- `load_rules(path)` -- reads `trust.yaml`, writing a starter template on
  first run; validates both tier and `editability`.
- `discover_origins(corpus_dir, rules)` -- scans for per-origin subfolders
  and computes each tier, applying the editability cap.
- `apply_editability_cap(tier, editability)` -- returns the capped tier and
  a note explaining the demotion, or `None` if nothing was capped.
- `needs_review(origin, previous, now)` -- why to re-review, or `None`.
- `save_decisions(...)` / `load_latest_decisions(...)` /
  `latest_decisions_path(...)` -- an immutable, versioned JSON audit trail.
  Versions are compared numerically, so `v10` sorts after `v9`.
- `origin_of(...)` / `tier_for_document(...)` -- per-document lookups.
  `None` means the document must not be ingested.

**`pgrag/adapters/llamaindex.py`**

- `stamp_documents(documents, corpus_dir, decisions)` -- returns
  `(kept, dropped, provenance)`. Attaches the approved tier to each kept
  document and drops anything excluded or outside an origin folder.
- `restamp_nodes(nodes, provenance)` -- re-applies the tier from the
  provenance map after all transforms have run, failing closed to `D` for
  any node it cannot trace. Nothing between chunking and indexing can
  quietly change a node's tier.
- `to_nodes(documents, provenance, transformations)` -- runs the
  transforms, then re-stamps as the last step.

Two details in the adapter are easy to miss and both matter:

- The reserved `_prov_tier` and `_prov_origin` keys are stripped from
  incoming metadata before stamping, so a document cannot declare its own
  tier in its own front matter.
- Both keys go into `excluded_llm_metadata_keys` and
  `excluded_embed_metadata_keys`, so the tier rides on the object and is
  never rendered into the prompt. A tier written as text is something a
  model can be argued out of.

**`pgrag/core/capability_gate.py`**

`compute_allowed_tools(tiers)` returns the tools allowed for a turn, from
the weakest tier present in retrieved context. One low-tier chunk withholds
every tool above it. Nothing here inspects content, so it still holds when
injection detection fails, and an empty context counts as the lowest tier.

**`pgrag/core/sanitizer.py`**

- `sanitize_text(text)` -- returns a `SanitizeResult`: the text unchanged,
  or replaced by a redaction marker, plus the sentences that triggered.
- `sanitize_documents(documents)` -- returns `(kept, dropped)`. The
  load-bearing call, wired into `build_index()` before stamping.

The registry decides whether an *origin* is trusted. This decides whether a
particular *document* is arguing with the model, which is a different
question: reviewing a folder is not the same as having read every file in
it, so an origin a reviewer accepted can still carry a planted file.

A single trigger anywhere in a document discards all of it. A document
willing to instruct a model has no remaining claim to being trusted for its
facts either.

The whole-document decision is the one that matters, and it has to happen
before chunking. A document longer than the chunk size gets split, and the
attacker controls that length: put the directive in one chunk and the false
claim in another, and redacting only the first leaves the second intact and
retrievable. Worse, the triggering chunk need not be retrieved at all, so
anything working at query time never sees the evidence that the document it
came from was hostile. `InstructionSanitizerPostprocessor` in the adapter
runs the same check between retrieval and the LLM, but only as defence in
depth for an index built by some other route.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt

ollama pull qwen3:30b-a3b
ollama pull nomic-embed-text
```

Dependencies are pinned exactly. `Requirements.txt` explains why: a version
bump can silently change how LlamaIndex carries metadata, which is the
thing the adapter depends on. Bump one package at a time and re-run the
tests.

`demo/run.py` uses `qwen3:30b-a3b` for generation and `nomic-embed-text`
through Ollama's own embedding endpoint -- both fully local, no API keys.
Note that `qwen3:30b-a3b` is around 18 GB and needs headroom to match; a
machine with 16 GB of RAM will swap or time out, and `qwen3:8b` is a
workable substitute.

## Running the demo

```bash
mkdir -p data/<origin-folder>
# put .txt documents in data/<origin-folder>/ -- one folder per origin
python demo/run.py
```

Each top-level folder under `data/` is one origin. The review gate shows
every origin's computed tier, reason, any editability cap, and document
count, then asks you to accept, override or exclude it before anything is
embedded.

Ask a question at the prompt; type `save <name>` to write the last
question, answer and sources to `results/<name>.txt`. The vector index is
persisted to `storage/` after the first run, so later runs load it back
instead of re-embedding.

## Tests

```bash
pytest
```

234 tests covering the registry, review gate, adapter, capability gate and
sanitizer. They need no LLM and no embedding model.

Mutation-checked rather than merely green: removing the spoof-stripping
`pop` fails one test, removing the metadata exclusion fails four, changing
`min(tiers)` to `max(tiers)` fails two, and sorting decision versions as
text rather than numbers fails four. All nine sanitizer mutations are
caught, including the one that matters most: making the sanitizer work per
chunk instead of per document.

## Known gaps

- **`restamp_nodes` is not wired into the demo.** The safeguard exists and
  is correct, but `demo/run.py` calls `stamp_documents` directly and never
  re-stamps after chunking, so nothing currently benefits from it.
- **Injection with no AI-facing wording is not detected.** The sanitizer
  catches text that addresses a model or demands an answer "with full
  confidence". A plausible internal memo that quietly redirects something
  has nothing in it for a text matcher to see. That attack is the
  capability gate's job -- it refuses the resulting action rather than
  detecting the text -- and `tests/test_sanitizer.py` asserts the gap is
  still open so nobody widens the patterns by accident.
- **Nothing fact-checks.** A low-tier document stating a plain false claim,
  carrying no instruction and no tool call, passes everything here.
