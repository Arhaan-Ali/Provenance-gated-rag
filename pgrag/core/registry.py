"""Origin registry: trust tiers computed per origin, never per document."""
from __future__ import annotations

import re
import yaml
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import IntEnum
from pathlib import Path



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
EDITABILITY_CAP = dict[str | None, TrustTier] = {
    "org-wide": TrustTier.C,
    "anonymous": TrustTier.D,
    None: TrustTier.D,
}                 # "capped at tier C for org and D for anonymous"

# Decay v1: a review older than this many days gets asked again.
REVIEW_MAX_AGE_DAYS = 90


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
    reason: "Internal wiki content; editable by staff, not externally verified
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
    """Cap the tier at C when many people (or nobody knows who) can edit the source.
       Returns (tier after cap, note for the review screen or None)."""
    cap = EDITABILITY_CAP.get(editability)
    if cap is None or tier <= cap:
        return tier, None
    why = f"editable by {editability}" if editability else "editability not declared in trust.yaml"
    return cap, f"{tier.name} -> {cap.name}: {why}"

def discover_origins(corpus_dir: Path, rules: dict[str, Rule]) -> list[Origin]:
    """Scan corpus_dir for per-origin subfolders and compute each one's tier."""

    origins: list[Origin] = []

    for entry in sorted(corpus_dir.iterdir()):
        if not entry.is_dir():
            continue
        document_count = sum(1 for f in entry.rglob("*") if f.is_file())
        rule = rules.get(entry.name)
        if rule is None:
            origins.append(Origin(
                entry.name, DEFAULT_TIER,
                "External or Unknown Origin, not listed in trust.yaml -> defaulting to lowest tier",
                document_count,
            ))
            continue
        tier, cap_note = apply_editability_cap(rule.tier, rule.editability)
        origins.append(Origin(entry.name, tier, rule.reason, document_count, rule.editability, cap_note))
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
    """Persist the decision set as a new, immutable, versioned audit file."""
    decisions_dir.mkdir(parents=True, exist_ok=True)
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
            }
            for d in decisions
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding= "utf-8")
    return out_path

def latest_decisions_path(decisions_dir: Path) -> Path | None:
    """Newest decisions_vN.json, compared by N as a number (not as text). None if none yet."""
    existing = _versions(decisions_dir)
    return existing[-1][1] if existing else None


def load_latest_decisions(decisions_dir: Path) -> dict[str, ReviewDecision]:
    """Load the newest decision file, keyed by origin name. Empty if none yet."""
    existing = latest_decisions_path(decisions_dir)
    if existing is None:
        return {}
    payload = json.loads(existing.read_text(encoding="utf-8"))
    file_date = payload.get("decided_at")
    return {
        d["origin_name"]: ReviewDecision(
            origin_name=d["origin_name"],
            tier=TrustTier[d["tier"]],
            excluded=bool(d["excluded"]),
            overridden=bool(d["overridden"]),
            reason=d.get("reason", ""),
            editability=d.get("editability"),
            reviewed_at=_parse_date(d.get("reviewed_at") or file_date),
        )
        for d in payload.get("decisions", [])
    }

def _parse_date(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None

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
    """Approved tier for one document, or None if it must not be ingested."""
    name = origin_of(doc_path, corpus_dir)
    if name is None:
        return None
    decision = decisions.get(name)
    if decision is None:
        return DEFAULT_TIER
    if decision.excluded:
        return None
    return decision.tier
