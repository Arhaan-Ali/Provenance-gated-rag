"""The terminal face of pgrag: logo, review gate cards, and summary."""
from __future__ import annotations

from datetime import datetime, timezone

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pgrag.core.registry import Origin, ReviewDecision, TrustTier, needs_review

# Rich turns colour off by itself when output is piped or NO_COLOR is set.
console = Console(highlight=False)

GOLD = "#e6b93f"
GOLD_DIM = "#8c6f1f"

# One colour per tier, used everywhere (review, summary, gate output).
TIER_STYLE = {
    TrustTier.A: "green",
    TrustTier.B: "dodger_blue1",
    TrustTier.C: "yellow",
    TrustTier.D: "red",
}

_TIER_NAMES = "/".join(t.name for t in sorted(TrustTier, reverse=True)) #Because int enum gives list in declaration order we need to sort to get A-D

# Solid blocks are bright, the box-drawing shadow is dim. Hard-coded so the
# logo needs no extra library.
LOGO = r"""
██████╗  ██████╗ ██████╗  █████╗  ██████╗
██╔══██╗██╔════╝ ██╔══██╗██╔══██╗██╔════╝
██████╔╝██║  ███╗██████╔╝███████║██║  ███╗
██╔═══╝ ██║   ██║██╔══██╗██╔══██║██║   ██║
██║     ╚██████╔╝██║  ██║██║  ██║╚██████╔╝
╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
""".strip("\n")

COMMANDS = (
    ("review", "confirm each origin's tier before anything is indexed"),
    ("index", "embed only what the review approved"),
    ("query", "ask a question"),
    ("explain", "why does this document carry this tier?"),
    ("doctor", "check trust.yaml, review dates and editability"),
)


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


def badge(tier: TrustTier) -> Text:
    """A coloured block with the tier letter, e.g. ' C '."""
    return Text(f" {tier.name} ", style=f"bold black on {TIER_STYLE[tier]}")


def print_logo() -> None:
    """The big logo. Only for the home screen, --help and first run, never on
    every command, and never when output is piped to a file."""
    if not console.is_terminal:
        return
    for line in LOGO.split("\n"):
        text = Text()
        for ch in line:
            text.append(ch, style=f"bold {GOLD}" if ch == "█" else GOLD_DIM)
        console.print(text)


def print_header(command: str, detail: str = "") -> None:
    """One-line header for every normal command: PGRAG · review"""
    line = Text.assemble(("PGRAG", f"bold {GOLD}"), (" · ", "grey50"), (command, "bold white"))
    if detail:
        line.append(f"   {detail}", style="grey62")
    console.print(line)
    console.print()


def show_home(origins: int | None = None, documents: int | None = None,
              last_reviewed: str | None = None, version: str = "0.1.0") -> None:
    """The screen you get from plain `pgrag`: logo, status line, command list."""
    print_logo()
    console.print(Text(f"provenance-gated retrieval  ·  v{version}", style="grey62"))
    if origins is not None:
        bits = [f"{origins} origins"]
        if documents is not None:
            bits.append(f"{documents} documents")
        if last_reviewed:
            bits.append(f"reviewed {last_reviewed}")
        console.print(Text("  ·  ".join(bits), style="grey62"))
    console.print()
    grid = Table.grid(padding=(0, 3))
    for name, what in COMMANDS:
        grid.add_row(Text(name, style="bold white"), Text(what, style="grey62"))
    console.print(grid)
    console.print()


# ---------------------------------------------------------------------------
# Review gate
# ---------------------------------------------------------------------------


def _origin_card(origin: Origin, why: str) -> Panel:
    """One origin as a card. Built from Text objects, so a '[' in a folder
    name or reason can never be read as Rich markup."""
    rows = Table.grid(padding=(0, 2))
    rows.add_column(style="grey62")
    rows.add_row("TIER", Text.assemble(badge(origin.tier), "  ", (origin.tier.description, "white")))
    rows.add_row("REASON", Text(origin.reason))
    edit = Text(origin.editability or "not declared")
    if origin.cap_note:
        edit.append(f"   capped  {origin.cap_note}", style="yellow")
    rows.add_row("EDITABILITY", edit)
    rows.add_row("DOCUMENTS", Text(f"{origin.document_count:,}"))
    title = Text.assemble((origin.name + "/", "bold"), ("  ", ""), (f"[{why}]", "grey62"))
    return Panel(rows, title=title, title_align="left",
                 border_style=TIER_STYLE[origin.tier], box=box.ROUNDED, padding=(0, 1))


def _print_summary(decisions: list[ReviewDecision]) -> None:
    counts = {tier: 0 for tier in sorted(TrustTier, reverse=True)}
    rejected = 0
    for d in decisions:
        if d.excluded:
            rejected += 1
        else:
            counts[d.tier] += 1
    line = Text("Decided  ", style="bold")
    for tier, n in counts.items():
        if n:
            line.append_text(badge(tier))
            line.append(f" {n}  ")
    if rejected:
        line.append(f"rejected {rejected}", style="red")
    console.print()
    console.print(line)


def run_review_gate(
        origins: list[Origin],
        previous: dict[str, ReviewDecision] | None = None,
        now: datetime | None = None,
) -> list[ReviewDecision]:
    """Show each origin to a human. Ask again only if it's new, stale, or changed."""
    previous = previous or {}
    now = now or datetime.now(timezone.utc)

    print_header("review", f"{len(origins)} origins")

    decisions = []
    for origin in origins:
        prev = previous.get(origin.name)
        why = needs_review(origin, prev, now)

        if why is None:
            # Still shown, not asked. Keep the OLD date so it can go stale later.
            console.print(Text.assemble(
                ("✓ ", "green"), (origin.name + "/", "bold"),
                (f"   tier {prev.tier.name}, reviewed {prev.reviewed_at:%Y-%m-%d} - keeping earlier decision", "grey62"),
            ))
            decisions.append(prev)
            continue

        console.print(_origin_card(origin, why))
        console.print(Text(f"  [Enter] stamp   [{_TIER_NAMES}] override tier   [x] reject", style="black on grey70"))

        while True:
            choice = input("> ").strip().upper()
            if choice in ("", "X") or choice in TrustTier.__members__:
                break
            console.print(Text(f"  Not understood: {choice!r}. Try again.", style="red"))

        if choice == "X":
            tier, excluded, overridden, reason = origin.tier, True, False, origin.reason
            console.print(Text("  ✗ rejected, will not be indexed", style="red"))
        elif choice in TrustTier.__members__:
            tier, excluded, overridden = TrustTier[choice], False, True
            reason = f"Overridden by reviewer from {origin.tier.name} to {tier.name}"
            console.print(Text.assemble(("  ↻ overridden to ", "yellow"), badge(tier)))
        else:
            tier, excluded, overridden, reason = origin.tier, False, False, origin.reason
            console.print(Text.assemble(("  ✓ stamped ", "green"), badge(tier)))
        console.print()

        decisions.append(ReviewDecision(
            origin.name, tier, excluded, overridden, reason,
            editability=origin.editability,
            reviewed_at=now,
        ))

    _print_summary(decisions)
    return decisions