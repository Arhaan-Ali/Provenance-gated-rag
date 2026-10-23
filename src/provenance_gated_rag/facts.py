"""Facts the trusted sources state, and the ones they disagree on.

M5 grounds a sentence in the source it cites. If two trusted sources say
different things -- the wiki names Elena Cho as CEO, a partner page names
someone else -- the answer is grounded in whichever one the model happened
to cite, and the other is never looked at. This module reads a few kinds of
fact out of every trusted document, so a disagreement is found whether or
not both documents were retrieved:

  - who holds a role:        "Elena Cho is the Chief Executive Officer"
                             "Sofia Marchetti, Chief Financial Officer, ..."
                             "The CEO of the company is Maria Lopez"
  - where the headquarters is: "Our headquarters is in Berlin"
  - the founding year:        "founded in 2015"

Deliberately narrow and deterministic: a pattern that misreads a sentence
reports a conflict that is not there, which declines a correct answer. What
the patterns do not cover, the entailment model (nli.py) can still catch
between retrieved sources. A sentence that negates ("is not", "no longer")
or speaks in the past ("was the CEO") states no current fact and is skipped.

Policy on a conflict: decline, and show both sources. Neither tier nor
date says which source is right, so the answer is left to a person.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

_NAME = r"[A-Z][a-z]+(?:[ -][A-Z][a-z'-]+)+"
_ROLE = (
    r"(?:Chief(?: [A-Z][a-z]+)+ Officer|C[A-Z]{1,3}O|President|Chair(?:man|woman|person)?"
    r"|Co-[Ff]ounder|Founder|Head of [A-Z][a-z]+|Vice President of [A-Z][a-z]+|VP of [A-Z][a-z]+)"
)
_ARTICLE = r"(?:(?:the|our|its|their) )?(?:company's )?"

_ROLE_PATTERNS = (
    # "Elena Cho is the Chief Executive Officer", "... has served as CEO"
    re.compile(rf"(?P<name>{_NAME}),? (?:is|serves as|has served as|has been|acts as|remains) "
               rf"{_ARTICLE}(?P<role>{_ROLE})"),
    # "Sofia Marchetti, Chief Financial Officer, joined in 2019"
    re.compile(rf"(?P<name>{_NAME}), {_ARTICLE}(?P<role>{_ROLE})\b"),
    # "The CEO of the company is Maria Lopez"
    re.compile(rf"\b(?:The|Our|Its) {_ARTICLE}(?P<role>{_ROLE})(?: of [\w ]+?)? is (?P<name>{_NAME})"),
)
_HQ = re.compile(r"\bheadquarter(?:s|ed)\b(?: is| are)? in (?P<place>[A-Z][a-z]+)")
_FOUNDED = re.compile(r"\b(?:founded|established|incorporated) in (?P<year>(?:19|20)\d\d)\b")

_NEGATED = re.compile(r"\b(?:not|no longer|never)\b|n't\b", re.IGNORECASE)
_PAST = re.compile(r"\b(?:former|formerly|previous|previously|was|were|until)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Fact:
    key: str        # what the fact is about, e.g. "CEO", "headquarters"
    value: str      # what the source says it is, e.g. "Elena Cho"
    where: str      # which source said it, e.g. "verified_internal_wiki/leadership_team.txt"
    sentence: str   # the sentence it was read from


def _role_key(role: str) -> str:
    """"Chief Executive Officer" and "CEO" are one key; other roles as written."""
    words = role.split()
    if len(words) > 1 and words[0] == "Chief" and words[-1] == "Officer":
        return "".join(w[0] for w in words).upper()
    return role.upper() if re.fullmatch(r"C[A-Z]{1,3}O", role) else role.lower()


def extract_facts(text: str, where: str) -> list[Fact]:
    facts = []
    # A blank line ends a sentence too: a heading ("... Leadership Team")
    # run into the paragraph under it made "Leadership Team Elena Cho" the
    # CEO. Single line breaks are hard wrapping inside a sentence.
    paragraphs = (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text))
    for sentence in (x for p in paragraphs for x in _SENTENCE_SPLIT.split(p) if x):
        if _NEGATED.search(sentence):
            continue
        # A founding year is told in the past ("was founded in 2015"); who
        # holds a role or where the headquarters is must be present tense.
        if not _PAST.search(sentence):
            for pattern in _ROLE_PATTERNS:
                for m in pattern.finditer(sentence):
                    facts.append(Fact(_role_key(m["role"]), m["name"], where, sentence))
            for m in _HQ.finditer(sentence):
                facts.append(Fact("headquarters", m["place"], where, sentence))
        for m in _FOUNDED.finditer(sentence):
            facts.append(Fact("founded", m["year"], where, sentence))
    return list(dict.fromkeys(facts))


def conflicts(facts: list[Fact]) -> dict[str, list[Fact]]:
    """Every key the facts give more than one value for, with the facts behind each."""
    by_key: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        by_key[fact.key].append(fact)
    return {
        key: found for key, found in by_key.items()
        if len({f.value.lower() for f in found}) > 1
    }


def describe(key: str, found: list[Fact]) -> str:
    """One line a person can act on: who says what."""
    seen: dict[str, list[str]] = defaultdict(list)
    for f in found:
        if f.where not in seen[f.value]:
            seen[f.value].append(f.where)
    said = "; ".join(f"'{value}' in {', '.join(wheres)}" for value, wheres in seen.items())
    return f"Trusted sources disagree on the {key}: {said}."


def _mentions(answer: str, key: str, found: list[Fact]) -> bool:
    low = answer.lower()
    if any(f.value.lower() in low for f in found):
        return True
    if key.isupper():   # an acronym key also matches the role spelled out
        spelled = r"\bChief(?: [A-Z][a-z]+)+ Officer\b"
        return bool(re.search(rf"\b{key}\b", answer)) or any(
            _role_key(m.group(0)) == key for m in re.finditer(spelled, answer))
    return key in low


def answer_conflicts(answer: str, facts: list[Fact]) -> list[str]:
    """A reason for each disputed fact the answer touches. Empty if it touches none."""
    return [describe(key, found) for key, found in conflicts(facts).items() if _mentions(answer, key, found)]


def facts_from_nodes(nodes, min_tier) -> list[Fact]:
    """The facts stated by nodes (or documents) at min_tier or above, neutralised ones skipped."""
    from provenance_gated_rag.capability_gate import tiers_from_nodes
    from provenance_gated_rag.sanitizer import REDACTION_MARKER

    found: list[Fact] = []
    for item, tier in zip(nodes, tiers_from_nodes(nodes), strict=True):
        node = getattr(item, "node", item)
        text = node.get_content()
        if tier < min_tier or text.strip() == REDACTION_MARKER:
            continue
        meta = node.metadata or {}
        where = "/".join(str(meta[k]) for k in ("origin", "source_file") if meta.get(k)) or "an unnamed source"
        found += extract_facts(text, where)
    return list(dict.fromkeys(found))
