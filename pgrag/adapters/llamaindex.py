from __future__ import annotations

from pathlib import Path



from pgrag.core.registry import (
    origin_of,
    tier_for_document,
)

TIER_KEY = "_prov_tier"
ORIGIN_KEY = "_prov_origin"
RESERVED_KEYS = (TIER_KEY, ORIGIN_KEY)




def stamp_documents(documents, corpus_dir: Path, decisions):
    """Attach the approved tier to each document; drop the ones that must not be indexed.

    Returns (kept, dropped). A dropped document was excluded by the reviewer or
    sits outside any origin folder - either way it never reaches the index.
    """
    kept, dropped = [], []

    for doc in documents:
        path = Path(doc.metadata.get("file_path", ""))

        # Reserve the namespace: nothing upstream is allowed to supply these.
        for key in RESERVED_KEYS:
            doc.metadata.pop(key, None)

        tier = tier_for_document(path, corpus_dir, decisions)
        if tier is None:
            dropped.append(path)
            continue

        doc.metadata[TIER_KEY] = tier.name
        doc.metadata[ORIGIN_KEY] = origin_of(path, corpus_dir) or "?"

        # Structural, not textual: the tier rides on the object and never gets
        # rendered into the prompt, so it cannot be spoofed by document content.
        for key in RESERVED_KEYS:
            if key not in doc.excluded_llm_metadata_keys:
                doc.excluded_llm_metadata_keys.append(key)
            if key not in doc.excluded_embed_metadata_keys:
                doc.excluded_embed_metadata_keys.append(key)

        kept.append(doc)

    return kept, dropped