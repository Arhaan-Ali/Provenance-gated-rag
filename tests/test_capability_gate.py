import pytest

from provenance_gated_rag.capability_gate import (
    TOOL_REQUIRED_TIER,
    compute_allowed_tools,
)
from provenance_gated_rag.registry import TrustTier


def test_high_trust_context_allows_all_tools():
    allowed = compute_allowed_tools([TrustTier.A, TrustTier.B])
    assert "search_corpus" in allowed
    assert "send_email" in allowed


def test_one_low_tier_chunk_blocks_send_email():
    allowed = compute_allowed_tools([TrustTier.A, TrustTier.D])
    assert "search_corpus" in allowed
    assert "send_email" not in allowed


def test_empty_context_defaults_to_lowest_tier():
    allowed = compute_allowed_tools([])
    assert allowed == ["search_corpus"]


def test_tier_exactly_at_threshold_is_allowed():
    allowed = compute_allowed_tools([TrustTier.B])
    assert "send_email" in allowed


def test_tier_just_below_threshold_is_blocked():
    allowed = compute_allowed_tools([TrustTier.C])
    assert "send_email" not in allowed


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        # The whole table, pinned per tier. Writing the lists out rather than
        # deriving them from TOOL_REQUIRED_TIER is the point: a self-referential
        # assertion cannot catch a wrong threshold or a dangerous tool added at
        # the wrong tier, and this parametrization is what does.
        (TrustTier.A, ["http_request", "search_corpus", "send_email",
                       "web_search", "write_database"]),
        (TrustTier.B, ["search_corpus", "send_email", "web_search"]),
        (TrustTier.C, ["search_corpus"]),
        (TrustTier.D, ["search_corpus"]),
    ],
)
def test_every_single_tier_context(tier, expected):
    assert sorted(compute_allowed_tools([tier])) == sorted(expected)


def test_order_of_chunks_does_not_matter_only_the_weakest():
    assert compute_allowed_tools([TrustTier.D, TrustTier.A]) == compute_allowed_tools(
        [TrustTier.A, TrustTier.D]
    )


def test_one_untrusted_chunk_outvotes_many_trusted_ones():
    many_trusted = [TrustTier.A] * 20

    assert "send_email" not in compute_allowed_tools([*many_trusted, TrustTier.D])


def test_duplicate_tiers_behave_like_a_single_one():
    assert compute_allowed_tools([TrustTier.B, TrustTier.B]) == compute_allowed_tools(
        [TrustTier.B]
    )


def test_every_registered_tool_is_reachable_at_its_own_threshold():
    # A tool nobody can ever call is a config bug, not a safe default.
    for tool, required_tier in TOOL_REQUIRED_TIER.items():
        assert tool in compute_allowed_tools([required_tier])


# --- what may be offered with untrusted content in context -----------------

def test_only_local_read_only_tools_are_allowed_at_the_lowest_tier():
    from provenance_gated_rag.capability_gate import LOCAL_READ_ONLY_TOOLS

    assert set(compute_allowed_tools([TrustTier.D])) <= LOCAL_READ_ONLY_TOOLS


@pytest.mark.parametrize("tool", ["send_email", "http_request", "web_search", "write_database"])
def test_a_table_that_puts_a_consequential_tool_at_tier_d_is_refused(tool):
    from provenance_gated_rag.capability_gate import check_tool_table

    with pytest.raises(ValueError, match=tool):
        check_tool_table({**TOOL_REQUIRED_TIER, tool: TrustTier.D})


def test_a_new_tool_added_at_tier_d_is_refused_until_declared_local():
    from provenance_gated_rag.capability_gate import check_tool_table

    with pytest.raises(ValueError, match="fetch_url"):
        check_tool_table({**TOOL_REQUIRED_TIER, "fetch_url": TrustTier.D})


def test_the_shipped_table_passes():
    from provenance_gated_rag.capability_gate import check_tool_table

    check_tool_table(TOOL_REQUIRED_TIER)
