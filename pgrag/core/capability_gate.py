"""Capability gate: what a turn may do, derived from its weakest source."""
from __future__ import annotations

from pgrag.core.registry import DEFAULT_TIER, TrustTier


# Minimum tier every retrieved chunk must meet for the tool to be offered.
TOOL_REQUIRED_TIER = {
    "lookup_fact": TrustTier.D,
    "send_email": TrustTier.B,
}


def compute_allowed_tools(tiers: list[TrustTier]) -> list[str]:
    """Tools allowed this turn. One low-tier chunk withholds every tool above it.

    Nothing here inspects content, so it still holds when injection detection fails.
    Empty context counts as the lowest tier.
    """
    minimum = min(tiers, default=DEFAULT_TIER)
    return [tool for tool, required in TOOL_REQUIRED_TIER.items() if minimum >= required]
