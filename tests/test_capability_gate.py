import pytest

from pgrag.core.capability_gate import TOOL_REQUIRED_TIER, compute_allowed_tools
from pgrag.core.registry import TrustTier


def test_high_trust_context_allows_all_tools():
    allowed = compute_allowed_tools([TrustTier.A])

    assert sorted(allowed) == sorted(TOOL_REQUIRED_TIER)

def test_one_b_chunk_lowers_the_whole_turn():
   allowed = compute_allowed_tools([TrustTier.A, TrustTier.B])
   assert "send_email" in allowed          # B is enough for this
   assert "write_database" not in allowed  # needs A
   assert "http_request" not in allowed    # needs A


def test_one_low_tier_chunk_blocks_send_email():
    allowed = compute_allowed_tools([TrustTier.A, TrustTier.D])

    assert "search_corpus" in allowed
    assert "send_email" not in allowed


def test_empty_context_defaults_to_lowest_tier():
    # Failing closed: no context must not read as "nothing untrusted here".
    assert compute_allowed_tools([]) == ["search_corpus"]


def test_tier_exactly_at_threshold_is_allowed():
    assert "send_email" in compute_allowed_tools([TrustTier.B])


def test_tier_just_below_threshold_is_blocked():
    assert "send_email" not in compute_allowed_tools([TrustTier.C])


@pytest.mark.parametrize(
    ("tiers", "expected"),
    [
        ([TrustTier.A], ["search_corpus", "send_email", "web_search", "write_database", "http_request"]),
        ([TrustTier.B], ["search_corpus", "send_email", "web_search"]),
        ([TrustTier.C], ["search_corpus"]),
        ([TrustTier.D], ["search_corpus"]),
    ],
)
def test_every_single_tier_context(tiers, expected):
    assert sorted(compute_allowed_tools(tiers)) == sorted(expected)


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
    for tool, required in TOOL_REQUIRED_TIER.items():
        assert tool in compute_allowed_tools([required])
