"""Origin registry: trust tiers computed per origin, never per document."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path

import yaml

from provenance_gated_rag.signing import (
    SIGNATURE_KEY,
    key_dir,
    load_key,
    new_trail_id,
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

    @property
    def description(self) -> str:
        return {
            TrustTier.A: "User Authored",
            TrustTier.B: "Internal Documents",
            TrustTier.C: "External Verified",
            TrustTier.D: "External Un-verified",
        }[self]

DEFAULT_TIER = TrustTier.D #all unknown origin document are automatically D tiers.

# The file types an origin's files may have. Anything else is left out of
# the index and listed on the review screen as skipped -- before this, every
# file was read as plain text whatever it was, so a .docx or a .png got
# indexed as bytes, and a reviewer had no way to know it would be. .md and
# .json are read as text, one Document per file; .pdf by FILE_EXTRACTOR in
# ingest.py, also one Document per file.
SUPPORTED_SUFFIXES = (".txt", ".md", ".json", ".pdf")

# A decision older than this gets asked again rather than reused forever.
REVIEW_MAX_AGE_DAYS = 90

EDITABILITY = ("immutable", "authenticated-few", "org-wide", "anonymous")

# How much trust a source can hold given who is able to change it. A tier is
# a claim about the document you read; editability is about the document you
# will read next. Undeclared counts as anonymous: not saying who can edit a
# source is not evidence that few people can.
EDITABILITY_CAP: dict[str | None, TrustTier] = {
    "org-wide": TrustTier.C,
    "anonymous": TrustTier.D,
    None: TrustTier.D,
}


class DecisionsError(ValueError):
    """The decisions file is malformed. Fix or delete it; nothing was indexed.

    Ported from the fork. Our loader previously coerced whatever it found:
    bool(d["excluded"]) turned null into False, so an origin a reviewer had
    EXCLUDED got indexed while the record still said it was excluded; a
    timestamp with no offset parsed naive and then raised TypeError deep in the
    decay comparison; and an unknown tier raised a bare KeyError. For a
    security decision, "unparseable" has to mean "stop", not "default".
    """


def _parse_date(text: str | None, where: str) -> datetime | None:
    """ISO date, or None. Raises if it is unparseable or has no timezone.

    A naive datetime is rejected here rather than later: needs_review subtracts
    it from an aware now(), which is a TypeError at the decay check -- far from
    the file that caused it.
    """
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise DecisionsError(f"{where}: {text!r} is not a valid ISO date") from None
    if dt.tzinfo is None:
        raise DecisionsError(
            f"{where}: {text!r} has no timezone "
            "(expected e.g. 2026-10-08T00:00:00+00:00)"
        )
    return dt


@dataclass
class Rule:
    """One origin's entry in trust.yaml."""

    tier: TrustTier
    reason: str
    editability: str | None = None


@dataclass
class Origin:
    """A single origin discovered during ingest, with its computed trust tier.

    `tier` is the tier after any editability cap; `cap_note` records the
    demotion so the review screen can show the reviewer why.
    """

    name: str
    tier: TrustTier
    reason: str
    document_count: int
    editability: str | None = None
    cap_note: str | None = None
    # Relative path inside the origin folder -> sha256 of its contents. This
    # is exactly what the reviewer is shown, and what a decision approves.
    files: dict[str, str] = field(default_factory=dict)
    # The origin folder, so the review screen can preview files. None for an
    # Origin built by hand (tests), which then shows names only.
    path: Path | None = None
    # Files in the folder that are not a supported type: never indexed, and
    # shown to the reviewer so a skipped file is not a surprise.
    skipped: list[str] = field(default_factory=list)

@dataclass
class ReviewDecision:
    """The human-confirmed outcome for one origin, after the review gate."""

    origin_name: str
    tier: TrustTier
    excluded: bool
    overridden: bool
    reason: str
    reviewed_at: datetime | None = None
    editability: str | None = None
    # The file list (path -> sha256) the reviewer saw. A file not in here, or
    # whose contents no longer match, was never approved. None means the
    # decision predates file lists, so needs_review asks about it again.
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

origins:
  verified_internal_wiki:
    tier: B
    reason: "Internal wiki content; editable by staff, not externally verified"
"""


def load_rules(path: Path) -> dict[str, Rule]:
    """Read trust.yaml, writing a commented starter file on the first run. """

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_TEMPLATE, encoding="utf-8") # write the above written template into trust.yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    origins = raw.get("origins") or {}
    if not isinstance(origins, dict):
        raise ValueError(f"{path}: 'origins' must be a mapping of name -> {{tier, reason}}")  # noqa: TRY004

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
            )

        editability = body.get("editability")
        if editability is not None:
            editability = str(editability).strip().lower()
            if editability not in EDITABILITY:
                raise ValueError(
                    f"{path}: origin {name!r} has unknown editability {body['editability']!r} "
                    f"(valid: {', '.join(EDITABILITY)})"
                )

        rules[str(name)] = Rule(
            tier=TrustTier[tier_name],
            reason=str(body.get("reason", "Set in trust.yaml")),
            editability=editability,
        )
    return rules


def apply_editability_cap(
    tier: TrustTier,
    editability: str | None,
) -> tuple[TrustTier, str | None]:
    """Lower a tier to what its editability can support. Returns (tier, note).

    A tier says how much you trust what the source says today. Editability
    says how easily that can change tomorrow without anyone re-reviewing it,
    so a source a large or unauthenticated population can edit cannot hold a
    high tier no matter what the rubric claims.
    """
    cap = EDITABILITY_CAP.get(editability)
    if cap is None or tier <= cap:
        return tier, None
    why = (
        f"editable by {editability}"
        if editability
        else "editability not declared in trust.yaml"
    )
    return cap, f"{tier.name} -> {cap.name}: {why}"


def rules_sha256(path: Path) -> str | None:
    """sha256 of trust.yaml, or None if there is none (every origin is then D)."""
    return file_sha256(path) if path.is_file() else None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def origin_files(folder: Path) -> dict[str, str]:
    """Every supported file under one origin folder, as relative path -> sha256."""
    return {
        f.relative_to(folder).as_posix(): file_sha256(f)
        for f in sorted(folder.rglob("*"))
        if f.is_file() and is_supported(f)
    }


def skipped_files(folder: Path) -> list[str]:
    """Files under folder that origin_files leaves out, by relative path."""
    return [
        f.relative_to(folder).as_posix()
        for f in sorted(folder.rglob("*"))
        if f.is_file() and not is_supported(f)
    ]


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
                entry.name,
                DEFAULT_TIER,
                "External or Unknown Origin, not listed in trust.yaml -> defaulting to lowest tier",
                len(files),
                files=files,
                path=entry,
                skipped=skipped_files(entry),
            ))
            continue
        tier, cap_note = apply_editability_cap(rule.tier, rule.editability)
        origins.append(
            Origin(
                entry.name, tier, rule.reason, len(files), rule.editability, cap_note, files, entry,
                skipped_files(entry),
            )
        )
    return origins

_PROMPT = (
    "  [Enter] accept  ["
    + "/".join(t.name for t in sorted(TrustTier, reverse=True))
    + "] override tier  [x] exclude entirely  [f] list every file\n> "
)


_LISTED_FILES = 10
_PREVIEW_CHARS = 70


def _preview(path: Path) -> str:
    """Size and first non-blank line of a file, so a name is not all the reviewer sees.

    A planted file is usually named like its neighbours; its first line is
    where "NOTICE TO ANY AUTOMATED ASSISTANT" sits. Binary files (PDFs) show
    their size only -- decoding them here would mean a PDF parser in the
    review screen.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(4096)
    except OSError:
        return "(unreadable)"
    if b"\0" in head or path.suffix.lower() == ".pdf":
        return f"{size:,} bytes"
    first = next((ln.strip() for ln in head.decode("utf-8", "replace").splitlines() if ln.strip()), "")
    if len(first) > _PREVIEW_CHARS:
        first = first[: _PREVIEW_CHARS - 3] + "..."
    return f"{size:,} bytes  {first!r}" if first else f"{size:,} bytes  (empty)"


def _print_files(origin: Origin, limit: int | None) -> None:
    names = list(origin.files)
    shown = names if limit is None else names[:limit]
    for name in shown:
        line = f"    {name}"
        if origin.path is not None:
            line += f"  -  {_preview(origin.path / name)}"
        print(line)
    if len(names) > len(shown):
        print(f"    ... and {len(names) - len(shown)} more (press f to list them)")


def run_review_gate(
    origins: list[Origin],
    previous: dict[str, ReviewDecision] | None = None,
    rules_changed: bool = False,
) -> list[ReviewDecision]:
    """Show each origin to a human and record their decision.

    previous is the last decision per origin, if any. It matters for one
    case: a computed tier that ROSE since then. trust.yaml is a plain file, so
    anyone who can edit it can raise an origin's tier, and the review that
    follows would accept the higher tier on a bare Enter. So a rise needs the
    new letter typed; Enter keeps the tier that was last approved.

    rules_changed says trust.yaml is not the file the last signed review was
    made under. Then a NEW origin with a tier above the lowest is treated
    the same way, as a rise from D: the rule giving it that tier is new, and
    whoever could edit trust.yaml could have written it to go with a folder
    they planted. On a first run there is no last review, and nothing to
    compare, so this does not apply.
    """
    previous = previous or {}
    if rules_changed and origins:
        print("\n! trust.yaml has changed since the last signed review.")
    decisions = []
    reviewed_at = datetime.now(UTC)
    for origin in origins:
        print(f"\nOrigin: {origin.name}")
        print(f"  Computed tier: {origin.tier.name} - {origin.tier.description}")
        print(f"  Reason: {origin.reason}")
        if origin.cap_note:
            print(f"  Capped: {origin.cap_note}")
        print(f"  Documents: {origin.document_count}")
        # The reviewer approves these files, not the folder: anything added
        # or edited after this screen is asked about again (needs_review).
        _print_files(origin, limit=_LISTED_FILES)
        if origin.skipped:
            print(
                f"  Skipped, not indexed ({len(origin.skipped)}; only "
                f"{', '.join(SUPPORTED_SUFFIXES)} are read): {', '.join(origin.skipped)}"
            )

        prev = previous.get(origin.name)
        raised_from = prev.tier if prev and not prev.excluded and origin.tier > prev.tier else None
        if prev is None and rules_changed and origin.tier > DEFAULT_TIER:
            raised_from = DEFAULT_TIER
            print(
                f"  ! New origin, given {origin.tier.name} by a trust.yaml that changed since the "
                f"last review. Enter keeps {DEFAULT_TIER.name}; type {origin.tier.name} to accept it."
            )
        elif raised_from is not None:
            print(
                f"  ! Computed tier ROSE from {raised_from.name} to {origin.tier.name} since the last "
                f"review (trust.yaml changed). Enter keeps {raised_from.name}; type "
                f"{origin.tier.name} to accept the higher tier."
            )

        while True:
            choice = input(_PROMPT).strip().upper()
            if choice == "F":
                _print_files(origin, limit=None)
                continue
            if choice in ("", "X") or choice in TrustTier.__members__:
                break
            print(f"  Not understood: {choice!r}. Try again.")

        if choice == "X":
            decisions.append(
                ReviewDecision(
                    origin.name, origin.tier, True, False, origin.reason,
                    reviewed_at, origin.editability, dict(origin.files),
                )
            )
        elif raised_from is not None and choice == origin.tier.name:
            # Typed on purpose: the rise is accepted as the computed tier.
            decisions.append(
                ReviewDecision(
                    origin.name, origin.tier, False, False, origin.reason,
                    reviewed_at, origin.editability, dict(origin.files),
                )
            )
        elif raised_from is not None and choice == "":
            reason = (
                f"Kept at {raised_from.name}: computed tier rose to {origin.tier.name} "
                "and the reviewer did not confirm it"
            )
            # Recorded as an override, so the higher computed tier is not
            # treated as drift and asked about on every later run.
            decisions.append(
                ReviewDecision(
                    origin.name, raised_from, False, True, reason,
                    reviewed_at, origin.editability, dict(origin.files),
                )
            )
        elif choice in TrustTier.__members__:
            new_tier = TrustTier[choice]
            reason = f"Overridden by reviewer from {origin.tier.name} to {new_tier.name}"
            decisions.append(
                ReviewDecision(
                    origin.name, new_tier, False, True, reason,
                    reviewed_at, origin.editability, dict(origin.files),
                )
            )
        else:
            decisions.append(
                ReviewDecision(
                    origin.name, origin.tier, False, False, origin.reason,
                    reviewed_at, origin.editability, dict(origin.files),
                )
            )

    return decisions

_VERSION_RE = re.compile(r"^decisions_v(\d+)$")


def _versions(decisions_dir: Path) -> list[tuple[int, Path]]:
    """Every decisions_vN.json found, sorted by N. Odd filenames are ignored."""
    found = []
    for p in decisions_dir.glob("decisions_v*.json"):
        m = _VERSION_RE.match(p.stem)
        if m:  # skip decisions_v1_backup.json and friends instead of crashing
            found.append((int(m.group(1)), p))
    return sorted(found)


def latest_decisions_path(decisions_dir: Path) -> Path | None:
    """Newest decisions_vN.json, compared by N as a number. None if there are none yet.

    Ported from the fork, where freshness.py needs the path itself rather than
    the parsed decisions -- it records which file an index was built from.
    """
    existing = _versions(decisions_dir)
    return existing[-1][1] if existing else None


def origins_needing_review(
        origins: list[Origin],
        decisions: dict[str, ReviewDecision],
        now: datetime,
) -> list[tuple[str, str]]:
    """(origin name, why) for every origin the review gate has to ask about.

    needs_review answers for one origin; this answers for a corpus, which is
    what a staleness check wants. Keeping the reason strings means a caller can
    tell the reviewer what moved instead of just that something did.
    """
    out = []
    for origin in origins:
        why = needs_review(origin, decisions.get(origin.name), now)
        if why:
            out.append((origin.name, why))
    return out


def save_decisions(
    decisions: list[ReviewDecision],
    decisions_dir: Path,
    trust_rules_sha256: str | None = None,
) -> Path:
    """Persist the decision set as a new, signed, versioned audit file.

    Each file is HMAC-signed and records the previous file's signature, so
    the versions form a chain (see signing.py). Ported from the fork: before
    this, anyone who could write to data/decisions/ could hand the pipeline a
    decision no reviewer made.
    """
    decisions_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to extend a trail that has been tampered with. Checked before the
    # key is created, so a missing key cannot be papered over by a fresh one.
    trusted = _trusted_latest(decisions_dir)
    key = load_key(create=True)
    existing = _versions(decisions_dir)
    next_version = (existing[-1][0] if existing else 0) + 1
    out_path = decisions_dir / f"decisions_v{next_version}.json"

    payload = {
        "version": next_version,
        "decided_at": datetime.now(UTC).isoformat(),
        "decisions": [
            {
                "origin_name": d.origin_name,
                "tier": d.tier.name,
                "excluded": d.excluded,
                "overridden": d.overridden,
                "reason": d.reason,
                "reviewed_at": d.reviewed_at.isoformat() if d.reviewed_at else None,
                "editability": d.editability,
                "files": d.files,
            }
            for d in decisions
        ],
        # None starts a new chain: first run, or only unsigned files before.
        "previous_signature": trusted[1][SIGNATURE_KEY] if trusted else None,
        # Carried along the chain, so the head can be found after a move.
        "trail_id": (trusted[1].get("trail_id") if trusted else None) or new_trail_id(),
        # The trust.yaml this review was made under, signed with it, so the
        # next review can tell whether the rubric moved underneath it.
        "trust_rules_sha256": trust_rules_sha256,
    }
    payload[SIGNATURE_KEY] = sign(payload, key)
    out_path.write_text(json.dumps(payload, indent=2), encoding= "utf-8")
    write_head(decisions_dir, next_version, payload[SIGNATURE_KEY], payload["trail_id"])
    return out_path


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
    written before signing existed. Those are never trusted, so every origin
    is reviewed again. Raises DecisionsError on anything that looks like
    tampering -- nothing is indexed until a human looks.
    """
    existing = _versions(decisions_dir)
    head = read_head(decisions_dir)

    if not existing:
        if head and load_key() is None:
            # The key is gone too, and that head can only be checked with it.
            # This is the recovery the missing-key error below tells the user
            # to do (move the folder aside, re-run), so it starts a new trail:
            # nothing is trusted, and every origin is reviewed again.
            return None
        if head:
            raise DecisionsError(
                f"{decisions_dir} is empty, but v{head['version']} was signed - "
                "decision files were deleted"
            )
        return None

    latest_path = existing[-1][1]
    latest = _read_payload(latest_path)
    if head is None:
        # No head under this path: the folder may have moved. The trail id is
        # read from an unverified file here, but that is safe -- it only picks
        # which head to compare against, and the chain is still verified.
        head = read_head(decisions_dir, latest.get("trail_id"))
    if SIGNATURE_KEY not in latest:
        if head is None:
            return None
        raise DecisionsError(
            f"{latest_path.name} is not signed, but v{head['version']} was - "
            "it was not written by the review gate"
        )

    key = load_key()
    if key is None:
        raise DecisionsError(
            f"{latest_path.name} is signed, but there is no key in {key_dir()} to check it. "
            f"If the key was moved, point $PGRAG_KEY_DIR at it. If it is lost, the trail "
            f"cannot be verified any more: move {decisions_dir} aside (keep it as the old "
            "audit record) and re-run, which reviews every origin again under a new key."
        )

    # Walk back from the newest file to the start of its chain.
    by_version = dict(existing)
    version, path, payload = existing[-1][0], latest_path, latest
    signatures: dict[int, str] = {}
    while True:
        if not verify(payload, key):
            raise DecisionsError(
                f"{path.name}: signature does not match - edited, or not written by the review gate"
            )
        if payload.get("version") != version:
            raise DecisionsError(
                f"{path.name}: says version {payload.get('version')!r}, file name says {version}"
            )
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
            f"changed (newest now: {latest_path.name}) - an older decision was put back"
        )
    if head is None or head.get("version") != existing[-1][0] or read_head(decisions_dir) is None:
        write_head(decisions_dir, existing[-1][0], latest[SIGNATURE_KEY], latest.get("trail_id"))
    return latest_path, latest


def load_latest_rules_hash(decisions_dir: Path) -> str | None:
    """The trust.yaml hash the newest signed review recorded. None if it recorded none."""
    trusted = _trusted_latest(decisions_dir)
    value = trusted[1].get("trust_rules_sha256") if trusted else None
    return value if isinstance(value, str) else None


def load_latest_decisions(decisions_dir: Path) -> dict[str, ReviewDecision]:
    """Load the newest decision file, keyed by origin name. Empty if none yet.

    Validates rather than coerces. This file decides which origins are trusted
    and which are excluded, so a malformed one must stop the run instead of
    being silently interpreted -- see DecisionsError.
    """
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
                    f"{existing.name}: {flag!r} for {name!r} must be true or false, "
                    f"got {d[flag]!r}"
                )

        try:
            tier = TrustTier[d["tier"]]
        except KeyError:
            # d.get, not d[...]: when "tier" is the missing key, indexing it
            # inside the handler raises a second KeyError and the caller sees
            # that instead of this message. The fork has that bug; we do not.
            raise DecisionsError(
                f"{existing.name}: unknown tier {d.get('tier')!r} for {name!r}"
            ) from None

        files = d.get("files")
        if files is not None and not (
            isinstance(files, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
        ):
            raise DecisionsError(
                f"{existing.name}: 'files' for {name!r} must map file paths to sha256 hashes"
            )

        result[name] = ReviewDecision(
            origin_name=name,
            tier=tier,
            excluded=d["excluded"],
            overridden=d["overridden"],
            reason=d.get("reason", ""),
            reviewed_at=_parse_date(
                d.get("reviewed_at") or file_date,
                f"{existing.name}: reviewed_at for {name!r}",
            ),
            editability=d.get("editability"),
            files=files,
        )
    return result


def needs_review(
    origin: Origin,
    previous: ReviewDecision | None,
    now: datetime,
) -> str | None:
    """Why this origin must be shown to a human again, or None to reuse the old decision.

    Without this, the first reviewer's answer stands forever: a new origin
    folder is silently accepted at the default tier, and a rubric change in
    trust.yaml never gets seen by anyone. Reusing a decision is only safe
    while the thing it was made about has not moved.

    A reviewer's override is deliberate, so a computed tier that disagrees
    with it is not drift and does not re-trigger -- otherwise every run
    would ask about the same origin the reviewer already settled.
    """
    if previous is None:
        return "new origin"
    if previous.reviewed_at is None:
        return "no review date on record"

    age = now - previous.reviewed_at
    if age > timedelta(days=REVIEW_MAX_AGE_DAYS):
        return f"last reviewed {age.days} days ago"
    if previous.editability != origin.editability:
        return (
            f"editability changed {previous.editability or 'undeclared'} "
            f"-> {origin.editability or 'undeclared'}"
        )
    if not previous.overridden and previous.tier != origin.tier:
        return f"computed tier changed {previous.tier.name} -> {origin.tier.name}"
    # Approving a folder approves the files the reviewer saw, not whatever is
    # put in it later. Any addition, edit or removal is asked about again.
    if previous.files is None:
        return "no file list on record"
    added, changed, removed = file_changes(previous.files, origin.files)
    if added or changed or removed:
        parts = [
            f"{len(xs)} {what}"
            for xs, what in ((added, "added"), (changed, "changed"), (removed, "removed"))
            if xs
        ]
        return "files changed since review: " + ", ".join(parts)
    return None


def origin_of(doc_path: Path, corpus_dir: Path) -> str | None:
    """A document's origin is its top-level folder under corpus_dir.

    Returns None for a file sitting loose in corpus_dir with no origin folder.
    """
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

    None means the reviewer excluded this origin, or the file has no origin
    folder at all. Callers must drop those documents - that is the fail-closed
    path and skipping it defeats the exclude button.

    An origin that exists but was never reviewed gets DEFAULT_TIER rather than
    None: it enters the index at the lowest trust, where the capability gate
    handles it, and the review gate surfaces it on the next run. Refusing to
    ingest it instead would make a newly added folder vanish silently.
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
        # Checked at indexing time too, so a file added or swapped after the
        # review screen was shown is dropped even if nothing re-asked.
        resolved = doc_path.resolve()
        key = resolved.relative_to((corpus_dir / name).resolve()).as_posix()
        try:
            if decision.files.get(key) != file_sha256(resolved):
                return None
        except OSError:
            return None
    return decision.tier
