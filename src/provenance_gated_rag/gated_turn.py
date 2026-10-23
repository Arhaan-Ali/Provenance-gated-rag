"""The capability gate enforced on a real turn, not just computed.

Ported from the fork (run_gated_turn in pgrag/adapters/llamaindex.py). Until
this, the gate here only produced a list: scripts/run_m3_demo.py passed that
list to predict_and_call and trusted the model to stay inside it. A model
that names a tool it was never offered -- which is exactly what an injected
instruction asks it to do -- was then up to the LLM library to refuse.

So the gate is enforced twice:

  1. the model is only offered the tools the weakest retrieved tier allows;
  2. every tool call it makes is checked again before anything runs.

The second check is what holds when a model asks for a tool it was never
offered. Nothing here trusts the model to stay inside its tool list.

Scoped release (opt-in): one untrusted chunk withholds send_email even when
the user's request is legitimate -- the fork's eval measured 0/2 legitimate
emails sent. A caller can name a withheld tool together with a rule its
arguments must meet, e.g. "the recipient is on an allowlist". That tool is
then offered, and every call to it runs only if the rule holds; the rule is
checked here, after the model has chosen the arguments, so an injected "send
it to attacker@example.com" is refused like any withheld call.

A released call is also refused if its arguments carry content only an
untrusted source supplied. The allowlist checks who receives an email, not
what it says: measured with qwen3:8b, a legitimate email to an allowed
recipient carried the planted ticket number "DEV-7731" in its body. So every
name, number, address and link in the arguments has to appear in the
question or in a source trusted enough for the tool; one that appears only
in the untrusted sources stops the call.

Both checks are lexical, so a caller can add the last one that is not: a
person. Pass approve= (e.g. ask_on_terminal) and every released call that
passed both checks is shown to them, exactly as it would run, and runs only
on a yes.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import MetadataMode
from llama_index.core.tools import FunctionTool

from provenance_gated_rag.capability_gate import (
    TOOL_REQUIRED_TIER,
    allowed_tools_for_nodes,
    tiers_from_nodes,
)
from provenance_gated_rag.grounding import _key_terms
from provenance_gated_rag.ingest import PROVENANCE_KEYS
from provenance_gated_rag.sanitizer import SANITIZER_FLAG_KEY

TURN_PROMPT = """Answer the question using only the context below.
If the user asks you to do something and one of your tools does it, call that tool.

Context:
{context}

Question: {question}"""

_HIDDEN_FROM_MODEL = (*PROVENANCE_KEYS, SANITIZER_FLAG_KEY)


@dataclass
class GatedTurn:
    """What happened in one turn, for the caller to show."""

    answer: str
    allowed: list[str]
    withheld: list[str]
    ran: list[tuple[str, dict, str]] = field(default_factory=list)  # (tool, args, output)
    refused: list[tuple[str, dict, str]] = field(default_factory=list)  # (tool, args, why)


def _hide(node) -> None:
    # Normally done at ingest. Done again here because this is the last point
    # before the prompt, and a node can reach a turn by another route.
    for key in _HIDDEN_FROM_MODEL:
        if key not in node.excluded_llm_metadata_keys:
            node.excluded_llm_metadata_keys.append(key)


def recipient_allowlist(*addresses: str) -> Callable[[dict], bool]:
    """A scope rule for send_email: the recipient is exactly one of addresses."""
    allowed = {a.strip().lower() for a in addresses}
    return lambda args: str(args.get("to", "")).strip().lower() in allowed


_ADDRESS_OR_LINK = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|https?://[^\s\"'<>]+")


def _carried_terms(args: dict) -> list[str]:
    """The names, numbers, addresses and links in a call's string arguments."""
    terms: list[str] = []
    for value in args.values():
        if not isinstance(value, str):
            continue
        links = _ADDRESS_OR_LINK.findall(value)
        terms += links
        rest = _ADDRESS_OR_LINK.sub(" ", value)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", rest):
            terms += _key_terms(sentence)
    return list(dict.fromkeys(terms))


def ask_on_terminal(tool: str, args: dict) -> bool:
    """Show a released call as it would run, and run it only if the user types y."""
    print(f"\nAn untrusted source is in context. The model wants to call {tool}:")
    for key, value in args.items():
        print(f"  {key}: {value}")
    return input("Run it? [y/N] ").strip().lower() == "y"


def untrusted_carryover(tool: str, args: dict, question: str, nodes) -> list[str]:
    """Terms in args that only a source below the tool's tier says.

    A term the question or a trusted-enough source also contains is not
    counted: the user asked for it, or a source that could have unlocked the
    tool says it too.
    """
    required = TOOL_REQUIRED_TIER.get(tool)
    if required is None:
        return []
    trusted, untrusted = [question.lower()], []
    for node, tier in zip(nodes, tiers_from_nodes(nodes), strict=True):
        text = getattr(node, "node", node).get_content().lower()
        (trusted if tier >= required else untrusted).append(text)
    return [
        term
        for term in _carried_terms(args)
        if any(term.lower() in t for t in untrusted) and not any(term.lower() in t for t in trusted)
    ]


def run_gated_turn(
    llm,
    question: str,
    nodes,
    tools: dict[str, Callable[..., str]],
    scoped: dict[str, Callable[[dict], bool]] | None = None,
    approve: Callable[[str, dict], bool] | None = None,
) -> GatedTurn:
    """One question, answered from `nodes`, with only the tools their weakest tier allows.

    `tools` is name -> function, e.g. tools.TOOL_REGISTRY. A tool missing from
    the gate's tier table is never offered, and a call to it is refused.

    `scoped` maps a tool the tier would withhold to a rule over its call
    arguments (see recipient_allowlist). Such a tool is offered, but a call
    runs only if the rule returns True. None, the default, releases nothing.

    `approve`, if given, is asked about every released call that passed its
    rule and the content check (see ask_on_terminal); False refuses it.
    Calls the tier allows on its own are not asked about.
    """
    scoped = scoped or {}
    by_tier = [name for name in allowed_tools_for_nodes(nodes) if name in tools]
    released = [name for name in scoped if name in tools and name not in by_tier]
    allowed = by_tier + released
    withheld = [name for name in tools if name not in allowed]

    chunks = [getattr(n, "node", n) for n in nodes]
    for node in chunks:
        _hide(node)
    context = "\n\n".join(node.get_content(metadata_mode=MetadataMode.LLM) for node in chunks)
    prompt = TURN_PROMPT.format(context=context or "(nothing retrieved)", question=question)

    offered = [FunctionTool.from_defaults(fn=tools[name], name=name) for name in allowed]
    if offered:
        response = llm.chat_with_tools(offered, user_msg=prompt)
        calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    else:
        response = llm.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        calls = []

    turn = GatedTurn(answer=response.message.content or "", allowed=allowed, withheld=withheld)
    for call in calls:
        name, args = call.tool_name, dict(call.tool_kwargs or {})
        if name not in allowed:
            why = (
                "withheld: a source in context is below the tier it needs"
                if name in tools
                else "no such tool"
            )
            turn.refused.append((name, args, why))
            continue
        if name in released and not scoped[name](args):
            turn.refused.append(
                (name, args, "outside the scope allowed while an untrusted source is in context")
            )
            continue
        carried = untrusted_carryover(name, args, question, nodes) if name in released else []
        if carried:
            listed = ", ".join(f'"{t}"' for t in carried)
            turn.refused.append(
                (name, args, f"carries {listed}, which only an untrusted source in context says")
            )
            continue
        if name in released and approve is not None and not approve(name, args):
            turn.refused.append((name, args, "not approved by the user"))
            continue
        try:
            turn.ran.append((name, args, str(tools[name](**args))))
        except Exception as e:  # noqa: BLE001 -- a bad argument from the model is reported, not raised
            turn.refused.append((name, args, f"failed: {e}"))
    return turn
