# Provenance-gated-rag

A retrieval layer that keeps working when some of its documents are
hostile -- by tracking where every chunk came from, and limiting what a
system is allowed to do when untrusted content is in context.

This does not claim to prevent prompt injection outright. It raises the
cost of document-borne prompt injection and fails safe when it cannot
tell.

## Status

Two pieces exist right now, **not yet wired together**:

- **`pgrag/core/registry.py`** -- a trust-tier system for document
  origins: computes a tier per origin, lets a human review/override/
  exclude before indexing, and persists the decisions as an audit trail.
  Fully implemented.
- **`demo/run.py`** -- a standalone, interactive RAG demo. Currently a
  *plain* pipeline: it indexes whatever is in `data/` with no trust
  tracking at all. It does not yet call anything in `pgrag/core/registry.py`.

`pgrag/cli.py` is an empty stub. `tests/` has no tests yet.

## Project structure

```
pgrag/
  core/
    registry.py   -- trust tiers, review gate, decision audit trail
  cli.py           -- empty stub, not implemented
demo/
  run.py           -- standalone interactive RAG demo (Ollama + llama-index)
tests/             -- empty
```

## Trust tier registry (`pgrag/core/registry.py`)

Four tiers, `A` (highest trust) down to `D` (lowest, and the default for
anything unlisted):

| Tier | Meaning |
|---|---|
| A | User authored -- you wrote it yourself |
| B | Internal documents -- from an internal system, others can edit it |
| C | External verified -- from outside, but signature/TLS/domain checked |
| D | External un-verified -- everything else |

The rubric lives in a `trust.yaml` file (not committed -- `load_rules()`
creates one with a commented starter template the first time it's run, if
none exists yet). Any origin not listed in it defaults to `D`
automatically, so a new/unknown source is never trusted by accident.

Key functions:
- `discover_origins(corpus_dir, rules)` -- scans a corpus for per-origin
  subfolders and computes each one's tier.
- `run_review_gate(origins)` -- interactively shows each origin's
  computed tier and reason, and lets a human accept, override to a
  specific tier, or exclude it entirely.
- `save_decisions(...)` / `load_latest_decisions(...)` -- an immutable,
  versioned JSON audit trail of every review decision made.
- `origin_of(doc_path, corpus_dir)` / `tier_for_document(...)` --
  per-document lookups: which origin a specific file belongs to, and
  what tier applies to it (or `None` if it must not be ingested at all).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-ollama pyyaml

ollama pull qwen3:8b
ollama pull nomic-embed-text
```

`demo/run.py` uses `qwen3:8b` for generation and `nomic-embed-text` (via
Ollama's own embedding endpoint) for embeddings -- both fully local, no
API keys needed.

## Running the demo

```bash
mkdir -p data
# put some .txt/.md/etc documents in data/
python demo/run.py
```

Ask a question at the prompt; type `save <name>` to write the last
question, answer, and sources to `results/<name>.txt`. The vector index
is persisted to `storage/` after the first run, so later runs load it
back instead of re-embedding everything.

## Not yet done

- The registry (`pgrag/core/registry.py`) and the demo (`demo/run.py`)
  are not connected -- the demo doesn't do any trust tiering yet.
- No tests.
- `pgrag/cli.py` is empty.
