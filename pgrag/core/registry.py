"""Origin registry: trust tiers computed per origin, never per document."""
from __future__ import annotations

import re
import yaml
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass
class Origin:
    """A single origin discovered during ingest, with its computed trust tier."""

    name: str
    tier: TrustTier
    reason: str
    document_count: int

@dataclass
class ReviewDecision:
    """The human-confirmed outcome for one origin, after the review gate."""

    origin_name: str
    tier: TrustTier
    excluded: bool
    overridden: bool
    reason: str


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


def load_rules(path: Path) -> dict[str, tuple[TrustTier, str]]:
    """Read trust.yaml, writing a commented starter file on the first run. """

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_TEMPLATE, encoding="utf-8") # writes the above written template into trust.yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {} #converts yaml content into python dict
    origins = raw.get("origins") or {}
    if not isinstance(origins, dict):
        raise ValueError(f"{path}: 'origins' must be a mapping of name -> {{tier, reason}}")

    rules: dict[str, tuple[TrustTier, str]] = {}

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

        rules[str(name)] = (TrustTier[tier_name], str(body.get("reason", "Set in trust.yaml")))
    return rules


def discover_origins(corpus_dir: Path, rules: dict[str, tuple[TrustTier, str]]) -> list[Origin]:
    """Scan corpus_dir for per-origin subfolders and compute each one's tier."""

    origins: list[Origin] = []

    for entry in sorted(corpus_dir.iterdir()):
        if not entry.is_dir():
            continue
        document_count = sum(1 for f in entry.rglob("*") if f.is_file())
        if entry.name in rules:
            tier, reason = rules[entry.name]
        else:
            tier, reason = DEFAULT_TIER, "External or Unknown Origin, not listed in trust.yaml -> defaulting to lowest tier"
        origins.append(Origin(entry.name, tier, reason, document_count))
    return origins


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

    payload = {
        "version": next_version,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "decisions": [
            {
                "origin_name": d.origin_name,
                "tier": d.tier.name,
                "excluded": d.excluded,
                "overridden": d.overridden,
                "reason": d.reason,
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
    return {
        d["origin_name"]: ReviewDecision(
            origin_name=d["origin_name"],
            tier=TrustTier[d["tier"]],
            excluded=bool(d["excluded"]),
            overridden=bool(d["overridden"]),
            reason=d.get("reason", ""),
        )
        for d in payload.get("decisions", [])
    }


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