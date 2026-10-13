"""Mock tools for the capability gate. Nothing here has a real side effect:
each one only reports what it would have done, so the demo can show which
actions a turn was allowed without anything actually being sent or written.

The names must match pgrag.core.capability_gate.TOOL_REQUIRED_TIER - a tool
missing from that table is never offered at all.
"""
from __future__ import annotations


def search_corpus(query: str) -> str:
    """Search the local document collection for more passages about a query."""
    return f"[mock] would search the corpus for {query!r}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone."""
    return f"[mock] would email {to!r}, subject {subject!r} ({len(body)} characters)"


def write_database(table: str, record: str) -> str:
    """Write a record into a database table."""
    return f"[mock] would write to table {table!r}: {record[:80]!r}"


def http_request(url: str, method: str = "GET") -> str:
    """Make an HTTP request to a URL."""
    return f"[mock] would send {method.upper()} {url}"


def web_search(query: str) -> str:
    """Search the public web."""
    return f"[mock] would search the web for {query!r}"


TOOLS = {fn.__name__: fn for fn in (search_corpus, send_email, write_database, http_request, web_search)}
