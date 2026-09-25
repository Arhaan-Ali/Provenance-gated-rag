from __future__ import annotations

from pathlib import Path



from pgrag.core.registry import (
    DEFAULT_TIER,
    origin_of,
    tier_for_document,
)

TIER_KEY = "_prov_tier"
ORIGIN_KEY = "_prov_origin"
RESERVED_KEYS = (TIER_KEY, ORIGIN_KEY)


HIDDEN_FROM_LLM = RESERVED_KEYS + ("file_path",)
HIDDEN_FROM_EMBED = RESERVED_KEYS


def _hide(node) -> None:
    """Keep our keys on the object but out of the text the LLM and embedder see."""
    for key in HIDDEN_FROM_LLM:
        if key not in node.excluded_llm_metadata_keys:
            node.excluded_llm_metadata_keys.append(key)
    for key in HIDDEN_FROM_EMBED:
        if key not in node.excluded_embed_metadata_keys:
            node.excluded_embed_metadata_keys.append(key)



def stamp_documents(documents, corpus_dir: Path, decisions):
    """Attach the approved tier to each document; drop the ones that must not be indexed.

    Returns (kept, dropped). A dropped document was excluded by the reviewer or
    sits outside any origin folder - either way it never reaches the index.
    """
    kept, dropped = [], []
    provenance: dict[str, dict[str, str]] = {}

    for doc in documents:
        path = Path(doc.metadata.get("file_path", ""))

        # Reserve the namespace: nothing upstream is allowed to supply these.
        for key in RESERVED_KEYS:
            doc.metadata.pop(key, None)

        tier = tier_for_document(path, corpus_dir, decisions)
        if tier is None:
            dropped.append(path)
            continue

        stamp = {TIER_KEY: tier.name, ORIGIN_KEY: origin_of(path, corpus_dir) or "?"}

        doc.metadata.update(stamp)
        provenance[doc.doc_id] = stamp

        # Structural, not textual: the tier rides on the object and never gets
        # rendered into the prompt, so it cannot be spoofed by document content.

        _hide(doc)

        kept.append(doc)

    return kept, dropped, provenance


def restamp_nodes(nodes, provenance: dict[str, dict[str, str]]):
    """Stamp last: overwrite the tier on every node from the map built at stamp time.

    Runs AFTER all transforms, so nothing that ran in between (an extractor, a
    buggy plugin, a poisoned reader) can change a node's tier.
    Fail closed: a node we can't trace back to a stamped document gets D.
    """
    for node in nodes:
        for key in RESERVED_KEYS:
            node.metadata.pop(key, None)
        stamp = provenance.get(node.ref_doc_id)
        if stamp is None:
            stamp = {TIER_KEY: DEFAULT_TIER.name, ORIGIN_KEY: "?"}
        node.metadata.update(stamp)
        _hide(node)
    return nodes


def to_nodes(documents, provenance: dict[str, dict[str, str]], transformations):
    """split -> (future extractors) -> restamp. Build the index from what this returns."""
    nodes = documents
    for transform in transformations:
        nodes = transform(nodes)
    return restamp_nodes(nodes, provenance)  # must stay the LAST step
