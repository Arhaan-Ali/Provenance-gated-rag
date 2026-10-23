"""Mock tools: something consequential for the capability gate to withhold.

Nothing here reaches the outside world. Each consequential tool records what
was attempted so a demo or a test can assert on it, which is the only way to
show the gate working: you have to be able to prove the model TRIED and was
refused, not just that nothing happened.

The tool set mirrors the fork's TOOL_REQUIRED_TIER so both repositories gate
the same names at the same thresholds. search_corpus is the harmless
counterpart that stays available at any tier -- it was called lookup_fact here
until we aligned with the fork.
"""
SENT_EMAILS = []
DB_WRITES = []
HTTP_REQUESTS = []
WEB_SEARCHES = []


def send_email(to: str, subject: str, body: str) -> str:
    SENT_EMAILS.append({"to": to, "subject": subject, "body": body})
    return f"Email sent to {to}"


def write_database(table: str, record: str) -> str:
    """Tier A: the only tool with a persistent effect.

    Worth the highest threshold for a reason the others do not have -- if
    anything written here ever becomes a corpus origin, tier-A context would
    be authoring its own future input.
    """
    DB_WRITES.append({"table": table, "record": record})
    return f"Wrote to {table}"


def http_request(url: str, method: str = "GET", body: str = "") -> str:
    """Tier A: arbitrary egress, and an ingestion channel in both directions.

    Whatever comes back re-enters the context with no origin and no tier, so a
    result is not covered by the gate's min() over retrieved chunks. Recorded,
    never sent.
    """
    HTTP_REQUESTS.append({"url": url, "method": method, "body": body})
    return f"{method} {url} -> (recorded, not sent)"


def web_search(query: str) -> str:
    """Tier B: exfiltration as much as retrieval -- the query crosses the
    trust boundary, and unlike send_email a query has no recipient to
    allowlist. Sits at B to match the fork; arguably belongs at A."""
    WEB_SEARCHES.append({"query": query})
    return f"Searched: {query}"


def search_corpus(query: str) -> str:
    """Tier D: always available, so a withheld turn can still answer."""
    return f"Looked up: {query}"


TOOL_REGISTRY = {
    "search_corpus": search_corpus,
    "send_email": send_email,
    "web_search": web_search,
    "write_database": write_database,
    "http_request": http_request,
}
