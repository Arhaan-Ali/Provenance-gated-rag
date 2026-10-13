"""Origin registry: trust tiers computed per origin, never per document."""
from __future__ import annotations

import re
import yaml
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import IntEnum
from pathlib import Path

from pgrag.core.signing import (
    SIGNATURE_KEY,
    key_dir,
    load_key,
    read_head,
    sign,
    verify,
    write_head,
)


class TrustTier(IntEnum):
    """A-D Document tiers, A with the highest trust and D with the lowest trust """

    D = 1
    C = 2
    B = 3
    A = 4

    def __str__(self) -> str:
        return self.name

    def __format__(self, spec: str) -> str:
        return format(self.name, spec)

    @property
    def description(self) -> str:
        return {
            TrustTier.A: "User Authored",
            TrustTier.B: "Internal Documents",
            TrustTier.C: "External Verified",
            TrustTier.D: "External Un-verified",
        }[self]

DEFAULT_TIER = TrustTier.D #all unknown origin document are automatically D tiers.

EDITABILITY = ("immutable", "authenticated-few", "org-wide", "anonymous")
WIDE_EDITABILITY = ("org-wide", "anonymous")  # "a large or unauthenticated population"
EDITABILITY_CAP: dict[str | None, TrustTier] = {
    "org-wide": TrustTier.C,
    "anonymous": TrustTier.D,
    None: TrustTier.D,
}                 # "capped at tier C for org and D for anonymous"

# Decay v1: a review older than this many days gets asked again.
REVIEW_MAX_AGE_DAYS = 90

class DecisionsError(ValueError):
    """The decisions file is malformed. Fix or delete it; nothing was indexed."""

def _parse_date(text: str | None, where: str) -> datetime | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise DecisionsError(f"{where}: {text!r} is not a valid ISO date") from None
    if dt.tzinfo is None:
        raise DecisionsError(
            f"{where}: {text!r} has no timezone (expected e.g. 2026-10-08T00:00:00+00:00)")
    return dt

@dataclass
class Rule:
    """ One origin's entry in trust.yaml"""
    tier: TrustTier
    reason: str
    editability: str | None


@dataclass
class Origin:
    """A single origin discovered during ingest, with its computed trust tier."""

    name: str
    tier: TrustTier
    reason: str
    document_count: int
    editability: str | None = None
    cap_note: str | None = None
    # relative path inside the origin folder -> sha256 of its contents. This is
    # exactly what the reviewer is shown, and what a decision approves.
    files: dict[str, str] = field(default_factory=dict)

@dataclass
class ReviewDecision:
    """The human-confirmed outcome for one origin, after the review gate."""

    origin_name: str
    tier: TrustTier
    excluded: bool
    overridden: bool
    reason: str
    editability: str | None = None
    reviewed_at: datetime | None = None
    # The file list (path -> sha256) the reviewer saw. A file not in here, or
    # whose contents no longer match, was never approved. None = recorded
    # before file lists existed, so it has to be asked about again.
    files: dict[str, str] | None = None


_TEMPLATE = """\
# trust.yaml - which origin folder gets which trust tier.
#
# Tiers, highest trust to lowest:
#   A  User authored         you wrote it yourself
#   B  Internal documents    from an internal system, other people can edit it
#   C  External verified     from outside, but signature / TLS / domain checked
#   D  External un-verified  everything else
#
# Any origin NOT listed below is treated as D. Indentation: two spaces for
# the name, four for its fields.
# editability - who can change this source?
#   immutable | authenticated-few | org-wide | anonymous
# org-wide or anonymous caps the tier at C. Not written = treated the same way.
#

origins:
  verified_internal_wiki:
    tier: B
    reason: "Internal wiki content; editable by staff, not externally verified"
    editability: org-wide
"""


def load_rules(path: Path) -> dict[str, Rule]:
    """Read trust.yaml, writing a commented starter file on the first run. """

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_TEMPLATE, encoding="utf-8") # writes the above written template into trust.yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {} #converts yaml content into python dict
    origins = raw.get("origins") or {}
    if not isinstance(origins, dict):
        raise ValueError(f"{path}: 'origins' must be a mapping of name -> {{tier, reason}}")

    rules: dict[str, Rule] = {}

    for name, body in origins.items():
        if not isinstance(body, dict) or "tier" not in body:
            raise ValueError(f"{path}: origin {name!r} needs a 'tier' key")

        tier_name = str(body["tier"]).strip().upper()

        if tier_name not in TrustTier.__members__:
            valid = ", ".join(t.name for t in sorted(TrustTier, reverse=True))
            raise ValueError(
                f"{path}: origin {name!r} has unknown tier {body['tier']!r} "
                f"(valid tiers: {valid})"
            ) # if trust tier is not from A-D it will show an error

        editability = body.get("editability")
        if editability is not None:
            editability = str(editability).strip().lower()
            if editability not in EDITABILITY:
                raise ValueError(
                    f"{path}: origin {name!r} has unknown editability {body['editability']!r} "
                    f"(valid: {', '.join(EDITABILITY)})"
                )  # same idea as the tier check above

        rules[str(name)] = Rule(
            tier=TrustTier[tier_name],
            reason=str(body.get("reason", "Set in trust.yaml")),
            editability=editability,
        )
    return rules

def apply_editability_cap(tier, editability):
    """Capped at C tier for org-wide access documents and D tier for anonymous documents"""

    cap = EDITABILITY_CAP.get(editability)
    if cap is None or tier <= cap:
        return tier, None
    why = f"editable by {editability}" if editability else "editability not declared in trust.yaml"
    return cap, f"{tier.name} -> {cap.name}: {why}"

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def origin_files(folder: Path) -> dict[str, str]:
    """Every file under one origin folder, as relative path -> sha256."""
    return {
        f.relative_to(folder).as_posix(): file_sha256(f)
        for f in sorted(folder.rglob("*"))
        if f.is_file()
    }


def file_changes(old: dict[str, str], new: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """(added, changed, removed) going from the old file list to the new one."""
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])
    return added, changed, removed


def discover_origins(corpus_dir: Path, rules: dict[str, Rule]) -> list[Origin]:
    """Scan corpus_dir for per-origin subfolders and compute each one's tier."""

    origins: list[Origin] = []

    for entry in sorted(corpus_dir.iterdir()):
        if not entry.is_dir():
            continue
        files = origin_files(entry)
        rule = rules.get(entry.name)
        if rule is None:
            origins.append(Origin(
                entry.name, DEFAULT_TIER,
                "External or Unknown Origin, not listed in trust.yaml -> defaulting to lowest tier",
                len(files), files=files,
            ))
            continue
        tier, cap_note = apply_editability_cap(rule.tier, rule.editability)
        origins.append(Origin(entry.name, tier, rule.reason, len(files), rule.editability, cap_note, files))
    return origins

def needs_review(origin, previous, now) -> str | None:
    """Why this origin must be asked about again, or None if the old decision still holds."""
    if previous is None:
        return "new origin"
    if previous.reviewed_at is None:
        return "no review date on record"
    age = now - previous.reviewed_at
    if age > timedelta(days=REVIEW_MAX_AGE_DAYS):
        return f"last reviewed {age.days} days ago"
    if previous.editability != origin.editability:
        return "editability changed in trust.yaml"
    if not previous.overridden and previous.tier != origin.tier:
        return "tier changed in trust.yaml"
    # Approving a folder approves the files the reviewer saw, not whatever is
    # put in it later. Any addition, edit or removal is asked about again.
    if previous.files is None:
        return "no file list on record"
    added, changed, removed = file_changes(previous.files, origin.files)
    if added or changed or removed:
        parts = [f"{len(xs)} {what}" for xs, what in
                 ((added, "added"), (changed, "changed"), (removed, "removed")) if xs]
        return "files changed since review: " + ", ".join(parts)
    return None


def origins_needing_review(
        origins: list[Origin],
        decisions: dict[str, ReviewDecision],
        now: datetime,
) -> list[tuple[str, str]]:
    """(origin name, why) for every origin the review gate has to ask about."""
    out = []
    for origin in origins:
        why = needs_review(origin, decisions.get(origin.name), now)
        if why:
            out.append((origin.name, why))
    return out

_VERSION_RE = re.compile(r"^decisions_v(\d+)$") #used for exact matches and names of version with similar names gets blocked through this


def _versions(decisions_dir: Path) -> list[tuple[int, Path]]:
    """Every decisions_vN.json found, sorted by N. Odd filenames are ignored."""

    found = []
    for p in decisions_dir.glob("decisions_v*.json"):
        m = _VERSION_RE.match(p.stem)
        if m:  # skip decisions_v1_backup.json and friends instead of crashing
            found.append((int(m.group(1)), p))
    return sorted(found)


def save_decisions(decisions: list[ReviewDecision], decisions_dir: Path) -> Path:
    """Persist the decision set as a new, signed, versioned audit file that
    chains to the previous one (see pgrag/core/signing.py)."""

    decisions_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to extend a trail that has been tampered with. Checked before the
    # key is created, so a missing key cannot be papered over by a fresh one.
    trusted = _trusted_latest(decisions_dir)
    key = load_key(create=True)
    existing = _versions(decisions_dir)
    next_version = (existing[-1][0] if existing else 0) + 1
    out_path = decisions_dir / f"decisions_v{next_version}.json"

    now = datetime.now(timezone.utc)

    payload = {
        "version": next_version,
        "decided_at": now.isoformat(),
        "origin_strategy" : "folder",
        "decisions": [
            {
                "origin_name": d.origin_name,
                "tier": d.tier.name,
                "excluded": d.excluded,
                "overridden": d.overridden,
                "reason": d.reason,
                "editability": d.editability,
                "reviewed_at": (d.reviewed_at or now).isoformat(),
                "files": d.files,
            }
            for d in decisions
        ],
        # None starts a new chain: first run, or only unsigned files before.
        "previous_signature": trusted[1][SIGNATURE_KEY] if trusted else None,
    }
    payload[SIGNATURE_KEY] = sign(payload, key)
    out_path.write_text(json.dumps(payload, indent=2), encoding= "utf-8")
    write_head(decisions_dir, next_version, payload[SIGNATURE_KEY])
    return out_path

def latest_decisions_path(decisions_dir: Path) -> Path | None:
    """Newest decisions_vN.json, compared by N as a number (not as text). None if none yet."""

    existing = _versions(decisions_dir)
    return existing[-1][1] if existing else None


def _read_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise DecisionsError(f"{path.name}: not valid JSON ({e})") from None
    if not isinstance(payload, dict):
        raise DecisionsError(f"{path.name}: expected a JSON object")
    return payload


def _trusted_latest(decisions_dir: Path) -> tuple[Path, dict] | None:
    """The newest decisions file, once its whole signed chain has checked out.

    None if there is nothing to trust yet: no files, or only unsigned ones
    written before signing existed (those are never trusted, so every origin
    is reviewed again). Raises DecisionsError on anything that looks like
    tampering - nothing is indexed until a human looks.
    """
    existing = _versions(decisions_dir)
    head = read_head(decisions_dir)

    if not existing:
        if head:
            raise DecisionsError(
                f"{decisions_dir} is empty, but v{head['version']} was signed - "
                "decision files were deleted")
        return None

    latest_path = existing[-1][1]
    latest = _read_payload(latest_path)
    if SIGNATURE_KEY not in latest:
        if head is None:
            return None
        raise DecisionsError(
            f"{latest_path.name} is not signed, but v{head['version']} was - "
            "it was not written by the review gate")

    key = load_key()
    if key is None:
        raise DecisionsError(
            f"{latest_path.name} is signed, but there is no key in {key_dir()} to check it")

    # Walk back from the newest file to the start of its chain.
    by_version = dict(existing)
    version, path, payload = existing[-1][0], latest_path, latest
    signatures: dict[int, str] = {}
    while True:
        if not verify(payload, key):
            raise DecisionsError(
                f"{path.name}: signature does not match - edited, or not written by the review gate")
        if payload.get("version") != version:
            raise DecisionsError(f"{path.name}: says version {payload.get('version')!r}, file name says {version}")
        signatures[version] = payload[SIGNATURE_KEY]
        previous = payload.get("previous_signature")
        if previous is None:
            break
        version -= 1
        path = by_version.get(version)
        if path is None:
            raise DecisionsError(f"decisions_v{version}.json is missing from the signed chain")
        payload = _read_payload(path)
        if payload.get(SIGNATURE_KEY) != previous:
            raise DecisionsError(f"{path.name} is not the version v{version + 1} was signed after")

    # The head is what was last signed here. The chain must still contain it,
    # so deleting the newest file (a rollback) cannot go unnoticed. A chain
    # that runs past the head is fine: only the key can sign, and that covers
    # a crash between writing a file and writing the head.
    if head is not None and signatures.get(head.get("version")) != head.get("signature"):
        raise DecisionsError(
            f"v{head.get('version')} was the newest signed decisions file, and it is gone or "
            f"changed (newest now: {latest_path.name}) - an older decision was put back")
    if head is None or head.get("version") != existing[-1][0]:
        write_head(decisions_dir, existing[-1][0], latest[SIGNATURE_KEY])
    return latest_path, latest


def load_latest_decisions(decisions_dir: Path) -> dict[str, ReviewDecision]:
    """Load the newest decision file, keyed by origin name. Empty if there is
    nothing trusted yet. Raises DecisionsError if the signed trail was tampered with."""

    trusted = _trusted_latest(decisions_dir)
    if trusted is None:
        return {}
    existing, payload = trusted
    file_date = payload.get("decided_at")

    result: dict[str, ReviewDecision] = {}
    for d in payload.get("decisions", []):
        name = d["origin_name"]

        for flag in ("excluded", "overridden"):
            if not isinstance(d[flag], bool):
                raise DecisionsError(
                    f"{existing.name}: '{flag}' for {name!r} must be true or false, got {d[flag]!r}")

        try:
            tier = TrustTier[d["tier"]]
        except KeyError:
            raise DecisionsError(f"{existing.name}: unknown tier {d['tier']!r} for {name!r}") from None

        files = d.get("files")
        if files is not None and not (
                isinstance(files, dict)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())):
            raise DecisionsError(
                f"{existing.name}: 'files' for {name!r} must map file paths to sha256 hashes")

        result[name] = ReviewDecision(
            origin_name=name,
            tier=tier,
            excluded=d["excluded"],
            overridden=d["overridden"],
            reason=d.get("reason", ""),
            editability=d.get("editability"),
            reviewed_at=_parse_date(d.get("reviewed_at") or file_date,
                                    f"{existing.name}: reviewed_at for {name!r}"),
            files=files,
        )
    return result

def _parse_date(text: str | None, where: str) -> datetime | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise DecisionsError(f"{where}: {text!r} is not a valid ISO date") from None
    if dt.tzinfo is None:
        raise DecisionsError(
            f"{where}: {text!r} has no timezone (expected e.g. 2026-10-08T00:00:00+00:00)")
    return dt

def origin_of(doc_path: Path, corpus_dir: Path) -> str | None:
    """A document's origin is its top-level folder under corpus_dir.
       Returns None for a file sitting loose in corpus_dir with no origin folder."""
    try:
        rel = doc_path.resolve().relative_to(corpus_dir.resolve())
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None

def tier_for_document(
        doc_path: Path,
        corpus_dir: Path,
        decisions: dict[str, ReviewDecision],
) -> TrustTier | None:
    """Approved tier for one document, or None if it must not be ingested.

    When the decision carries a file list, the document must be in it with the
    same contents as when it was reviewed. Checked here, at indexing time, so a
    file added or swapped after the review card was shown is dropped.
    """

    name = origin_of(doc_path, corpus_dir)
    if name is None:
        return None
    decision = decisions.get(name)
    if decision is None:
        return DEFAULT_TIER
    if decision.excluded:
        return None
    if decision.files is not None:
        resolved = doc_path.resolve()
        key = resolved.relative_to((corpus_dir / name).resolve()).as_posix()
        try:
            if decision.files.get(key) != file_sha256(resolved):
                return None
        except OSError:
            return None
    return decision.tier