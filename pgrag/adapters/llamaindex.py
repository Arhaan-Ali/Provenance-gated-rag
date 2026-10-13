from __future__ import annotations

from pathlib import Path

from typing import Any

from pgrag.core.sanitizer import DEFAULT_THRESHOLD, REDACTION_MARKER, screen_text


from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from pgrag.core.registry import (
    DEFAULT_TIER,
    origin_of,
    tier_for_document,
)


TIER_KEY = "_prov_tier"
ORIGIN_KEY = "_prov_origin"
RESERVED_KEYS = (TIER_KEY, ORIGIN_KEY)

# Set by the sanitizer postprocessor, not by the stamp, so it is not a
# provenance key - but it still has to stay out of the prompt.
SANITIZER_FLAG_KEY = "_prov_sanitizer_flagged"

HIDDEN_FROM_LLM = RESERVED_KEYS + (SANITIZER_FLAG_KEY, "file_path")
HIDDEN_FROM_EMBED = RESERVED_KEYS + (SANITIZER_FLAG_KEY,)


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


class InstructionSanitizerPostprocessor(BaseNodePostprocessor):
    """Screen the chunks a query retrieved, i.e. exactly what the model will read.

    Two stages (see pgrag.core.sanitizer.screen_text): the patterns first, and
    only if they find nothing, the optional classifier. A flagged chunk is
    quarantined -- its text is replaced by the redaction marker, so the model
    never sees it -- and the query carries on with the rest.

    This is the second line behind sanitize_documents(), not a replacement:
    it can only judge chunks that were retrieved, so a hostile document whose
    directive sits in a chunk that was NOT retrieved is invisible here.
    """

    classifier: Any = None            # text -> score in [0, 1]; None = patterns only
    threshold: float = DEFAULT_THRESHOLD

    @classmethod
    def class_name(cls) -> str:
        return "InstructionSanitizerPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        for node_with_score in nodes:
            result = screen_text(node_with_score.node.text, self.classifier, self.threshold)
            if result.flagged:
                node_with_score.node.set_content(REDACTION_MARKER)
                node_with_score.node.metadata[SANITIZER_FLAG_KEY] = True
                _hide(node_with_score.node)
        return nodes
