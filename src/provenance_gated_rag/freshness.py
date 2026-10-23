"""Is the saved index still valid? Pure standard library, no llama_index.

An index is only reusable if ALL of these still hold:
  1. it was built from the newest decisions file,
  2. the files under data/ are byte-for-byte what they were,
  3. no origin is due for review (new, older than 90 days, tier or editability changed).
Anything else and the caller must rebuild (which re-runs the review gate).
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from provenance_gated_rag.registry import (
    Origin,
    ReviewDecision,
    latest_decisions_path,
    origins_needing_review,
)

# Ported from the collaborator's fork at 1ab46dd, deliberately unchanged, so the
# two repositories can still be compared. Four findings from the 12 Oct review
# are therefore present here too and are NOT yet fixed:
#
#   1. stale_reason compares the decisions FILENAME, not its contents, so an
#      in-place edit of the newest decisions file is invisible -- including a
#      reviewer setting "excluded": true, which then never takes effect.
#   2. The manifest lives inside the directory it attests to and is unsigned,
#      so anyone who can write persist_dir can forge "fresh" and skip the
#      review gate entirely.
#   3. data_fingerprint frames records as path + b"\0" + content + b"\0".
#      Content can contain NUL, so the framing is ambiguous and two different
#      corpora can hash identically. Needs length-prefixed framing.
#   4. "documents changed" triggers a REBUILD but never a RE-REVIEW, so a
#      rewritten file in an approved origin is re-indexed under the old
#      approval. Fixing it properly needs a per-origin content digest on
#      ReviewDecision.
#
# Also: data_fingerprint opens every file with no exception handling, so one
# unreadable file raises PermissionError through the whole staleness check.

MANIFEST_NAME = ".index_manifest.json"


def data_fingerprint(corpus_dir: Path) -> str:
    """One hash over every file's relative path and contents under corpus_dir.

    Contents, not modified-time: copying a file must not cause a rebuild, and a
    document edited with its timestamp preserved must not slip through.
    """
    files = sorted(
        (p for p in corpus_dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(corpus_dir).as_posix(),
    )
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(corpus_dir).as_posix().encode("utf-8"))
        h.update(b"\0")
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def write_manifest(persist_dir: Path, decisions_file: str, corpus_dir: Path) -> Path:
    """Record what this index was built from. Call right after persisting the index."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = persist_dir / MANIFEST_NAME
    path.write_text(json.dumps({
        "decisions_file": decisions_file,
        "data_fingerprint": data_fingerprint(corpus_dir),
        "built_at": datetime.now(UTC).isoformat(),
    }, indent=2), encoding="utf-8")
    return path


def read_manifest(persist_dir: Path) -> dict | None:
    """The manifest, or None if it is missing or unreadable (treated as stale)."""
    try:
        data = json.loads((persist_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def stale_reason(
        persist_dir: Path,
        decisions_dir: Path,
        corpus_dir: Path,
        origins: list[Origin],
        decisions: dict[str, ReviewDecision],
        now: datetime | None = None,
) -> str | None:
    """Why the saved index must be rebuilt, or None if it can be reused."""
    now = now or datetime.now(UTC)

    manifest = read_manifest(persist_dir)
    if manifest is None:
        return "no index manifest (index missing or built by an older version)"

    latest = latest_decisions_path(decisions_dir)
    if latest is None or manifest.get("decisions_file") != latest.name:
        return "decisions changed since the index was built"

    if manifest.get("data_fingerprint") != data_fingerprint(corpus_dir):
        return "documents in data/ changed since the index was built"

    pending = origins_needing_review(origins, decisions, now)
    if pending:
        names = ", ".join(f"{name} ({why})" for name, why in pending)
        return f"review due: {names}"

    return None
