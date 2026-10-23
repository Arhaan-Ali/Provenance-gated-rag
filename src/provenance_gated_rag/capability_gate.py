"""M3 capability gate: what a turn may do, derived from its weakest source.

Permissions come from the lowest trust tier present in retrieved context, so
one untrusted chunk withholds a consequential tool for that turn no matter
how convincing the rest of the context is. Nothing here inspects content --
this is the defense that works even when detection fails.
"""
from provenance_gated_rag.registry import DEFAULT_TIER, TrustTier

# Minimum tier every retrieved chunk must meet for the tool to be offered.
# Aligned with the fork so both repositories gate the same names identically.
TOOL_REQUIRED_TIER = {
    "search_corpus": TrustTier.D,
    "send_email": TrustTier.B,
    "write_database": TrustTier.A,
    "http_request": TrustTier.A,
    "web_search": TrustTier.B,
}


# The only tools that may be offered with a tier-D chunk in context: ones
# whose whole effect stays on this machine and changes nothing. Measured
# (M6): llama3.2:3b passed an attacker's exfiltration URL to search_corpus
# as its query. Harmless for a local search, and exactly the exfiltration if
# the tool had reached the network -- so a tool at tier D must be listed
# here, and listing one is a claim that it cannot send anything anywhere.
LOCAL_READ_ONLY_TOOLS = frozenset({"search_corpus"})


def check_tool_table(table: dict[str, TrustTier]) -> None:
    """Refuse a tier table that lets an untrusted source reach a consequential tool."""
    unsafe = sorted(t for t, tier in table.items() if tier <= DEFAULT_TIER and t not in LOCAL_READ_ONLY_TOOLS)
    if unsafe:
        raise ValueError(
            f"{', '.join(unsafe)} would be offered with untrusted ({DEFAULT_TIER.name}) content in "
            f"context. Only local, read-only tools may be: {', '.join(sorted(LOCAL_READ_ONLY_TOOLS))}."
        )


# At import, so a misconfigured table stops everything instead of running.
check_tool_table(TOOL_REQUIRED_TIER)


# The metadata key ingest stamps each document with. Here rather than in the
# caller because reading it is part of the gate's job -- see tiers_from_nodes.
TIER_METADATA_KEY = "trust_tier"


def compute_allowed_tools(tiers: list[TrustTier]) -> list[str]:
    minimum = min(tiers, default=DEFAULT_TIER)
    return [tool for tool, required_tier in TOOL_REQUIRED_TIER.items() if minimum >= required_tier]


def tiers_from_nodes(nodes) -> list[TrustTier]:
    """Tier of every retrieved node. An unlabelled node counts as the lowest.

    This exists because the fail-closed rule used to live in the caller. Three
    scripts each wrote

        [TrustTier[n.node.metadata["trust_tier"]] for n in nodes]

    which is an unguarded index: a node with no stamp raises KeyError, and the
    obvious "fix" of .get(key, ...) invites a default chosen at the call site.
    Both plausible call sites give opposite answers -- filtering unlabelled
    nodes out offers every tool, defaulting them to D offers one -- so the
    answer belongs here, once, where it can be tested.

    A node arrives unlabelled when something upstream re-identified it: a
    transform that rebuilt nodes, a merged index, an index persisted by an
    older version. "I cannot tell what this is" has to mean the lowest tier,
    never the highest.

    Accepts NodeWithScore (what a retriever returns) or a bare node.
    """
    tiers = []
    for item in nodes:
        # item.metadata, not item.node.metadata: NodeWithScore proxies
        # .metadata through to its node, so this reads correctly for a
        # retriever's output AND for a bare node. Unwrapping .node first
        # also works but only for the wrapped shape, and a getattr dance to
        # handle both is unfalsifiable -- no test can tell the versions apart.
        name = (getattr(item, "metadata", None) or {}).get(TIER_METADATA_KEY)
        try:
            tiers.append(TrustTier[name])
        except KeyError:
            # Missing, misspelled, lowercase, or a tier letter that no longer
            # exists -- all the same answer.
            tiers.append(DEFAULT_TIER)
    return tiers


def allowed_tools_for_nodes(nodes) -> list[str]:
    """What this turn may do, given what was retrieved.

    The entry point a caller should use. compute_allowed_tools stays public
    because it is the pure function worth testing exhaustively, but a caller
    that assembles the tier list itself is a caller that can get the
    fail-closed default wrong.
    """
    return compute_allowed_tools(tiers_from_nodes(nodes))
