from __future__ import annotations
"""Review gate: the human-facing screen for confirming origin tiers."""


from pgrag.core.registry import Origin, ReviewDecision, TrustTier


_TIER_NAMES = "/".join(t.name for t in sorted(TrustTier, reverse=True)) #Because int enum gives list in declaration order we need to sort to get A-D

def run_review_gate(origins: list[Origin]) -> list[ReviewDecision]:
    """Show each origin to a human and record their decision."""


    decisions = []
    for origin in origins:
        print(f"\nOrigin: {origin.name}")
        print(f"  Computed tier: {origin.tier.name} - {origin.tier.description}")
        print(f"  Reason: {origin.reason}")
        print(f"  Documents: {origin.document_count}")

        while True:
            choice = input(f"  [Enter] accept  [{_TIER_NAMES}] override tier  [x] exclude entirely\n> ").strip().upper() #remove _prompt and instead of the add the input text normally
            if choice in ("", "X") or choice in TrustTier.__members__:
                break
            print(f"  Not understood: {choice!r}. Try again.")

        if choice == "X":
            decisions.append(ReviewDecision(origin.name, origin.tier, True, False, origin.reason))
        elif choice in TrustTier.__members__:
            new_tier = TrustTier[choice]
            reason = f"Overridden by reviewer from {origin.tier.name} to {new_tier.name}"
            decisions.append(ReviewDecision(origin.name, new_tier, False, True, reason))
        else:
            decisions.append(ReviewDecision(origin.name, origin.tier, False, False, origin.reason))

    return decisions