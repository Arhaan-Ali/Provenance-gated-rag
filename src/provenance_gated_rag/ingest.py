"""M1 ingest pipeline: origin registry + review gate, run before indexing."""
import hashlib
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file import PDFReader

from provenance_gated_rag.config import get_embed_model, get_llm
from provenance_gated_rag.facts import conflicts, describe, facts_from_nodes
from provenance_gated_rag.grounding import GROUNDING_MIN_TIER, GroundedQueryEngine
from provenance_gated_rag.registry import (
    SUPPORTED_SUFFIXES,
    Origin,
    ReviewDecision,
    discover_origins,
    is_supported,
    load_latest_decisions,
    load_latest_rules_hash,
    load_rules,
    needs_review,
    rules_sha256,
    run_review_gate,
    save_decisions,
)
from provenance_gated_rag.sanitizer import (
    REDACTION_MARKER,
    Classifier,
    InstructionSanitizerPostprocessor,
    sanitize_documents,
)

# Relative to the folder the caller runs from. These used to point into this
# repository's data/ folder, which does not exist once the package is
# installed with pip. The repository's own scripts pass their paths.
DEFAULT_CORPUS_DIR = Path("corpus")
DEFAULT_DECISIONS_DIR = Path("decisions")
DEFAULT_TRUST_RULES_PATH = Path("trust.yaml")

# source_file is the file's path inside its origin. It rides with the
# provenance keys, out of the prompt: a file name is text the file's author
# chose, and it is only there so a run can say which file was neutralised.
PROVENANCE_KEYS = ("origin", "trust_tier", "source_file")

# One Document per FILE, not per page.
#
# PDFReader defaults to return_full_document=False, which yields one
# Document per page -- and sanitize_documents decides per Document. That
# hands the attacker back the seam the whole-document filter exists to
# close: put the directive on page 1 and the false claim on page 2, and
# only page 1 is neutralised. The surviving page then carries the claim with
# nothing incriminating left in it to match on.
#
# Every reader that loads this corpus must pass this, or the defence holds
# for .txt and not for .pdf. tests/test_sanitizer.py has the guard.
FILE_EXTRACTOR = {".pdf": PDFReader(return_full_document=True)}


def _hide_provenance_from_model(document) -> None:
    """Keep provenance on the object but out of what the model and embedder see.

    Rendered into the prompt, `trust_tier: B` is just more text in a window
    the attacker also writes to: a document can state its own competing tier,
    or argue about the one above it. Structural metadata cannot be argued
    with, so the tier rides on the object and never reaches the text.

    Excluded from embeddings too -- a tier letter is not part of what a chunk
    is about, and letting it into the vector only blurs retrieval.
    """
    for key in PROVENANCE_KEYS:
        if key not in document.excluded_llm_metadata_keys:
            document.excluded_llm_metadata_keys.append(key)
        if key not in document.excluded_embed_metadata_keys:
            document.excluded_embed_metadata_keys.append(key)


def _snapshot_approved_files(origin_dir: Path, approved: dict[str, str], snapshot: Path) -> list[Path]:
    """Copy the approved, unchanged files into snapshot; return the copies.

    A decision approves files, not a folder. Something dropped into an accepted
    folder after the review, or an approved file edited since, is left out
    here even if nothing re-ran the review gate -- needs_review asks about it
    on the next run, but the index built now must not contain it.

    Each file is read ONCE: the hash is checked on those bytes and those same
    bytes are what gets indexed, from a private copy. Hashing the file and
    then letting the reader open it again left a window in which a swap
    landed in the index with the old file's approval.
    """
    out = []
    for rel, sha in sorted(approved.items()):
        if not is_supported(Path(rel)):
            continue  # approved before the allowlist; never indexed now
        try:
            data = (origin_dir / rel).read_bytes()
        except OSError:
            continue  # deleted since the review, or not a regular file
        if hashlib.sha256(data).hexdigest() != sha:
            continue
        copy = snapshot / rel
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(data)
        out.append(copy)
    return out


def load_documents(
    corpus_dir: Path,
    decisions: list[ReviewDecision],
    drop_instruction_documents: bool = True,
    classifier: Classifier | None = None,
):
    """Load only non-excluded origins' documents, tagged with origin + tier metadata.

    Two filters run here, and they catch different things. The review gate's
    exclusions are per *origin*: a whole folder is trusted or it isn't.
    sanitize_documents is per *document*: an accepted origin can still carry
    a file that instructs a model, and reviewing a folder does not mean
    reading every file in it.

    Passing drop_instruction_documents=False keeps hostile content in the
    index on purpose; M2's label-propagation test needs a D-tier document
    there to prove its metadata survives retrieval.
    """
    corpus_dir = Path(corpus_dir)
    documents = []
    for decision in decisions:
        if decision.excluded:
            continue
        origin_dir = corpus_dir / decision.origin_name
        metadata = {"origin": decision.origin_name, "trust_tier": decision.tier.name}
        with tempfile.TemporaryDirectory(prefix="pgrag-snapshot-") as snapshot:
            source = {"input_dir": str(origin_dir), "required_exts": list(SUPPORTED_SUFFIXES)}
            root = origin_dir
            if decision.files is not None:
                approved = _snapshot_approved_files(origin_dir, decision.files, Path(snapshot))
                if not approved:
                    continue
                source = {"input_files": [str(p) for p in approved]}
                root = Path(snapshot)
            docs = SimpleDirectoryReader(
                **source,
                file_metadata=lambda path, metadata=metadata, root=root: {
                    **metadata, "source_file": _relative(path, root),
                },
                exclude_hidden=False,
                file_extractor=FILE_EXTRACTOR,
            ).load_data()
        for doc in docs:
            _hide_provenance_from_model(doc)
        documents.extend(docs)

    if drop_instruction_documents:
        documents, neutralised = sanitize_documents(documents, classifier)
        # Said out loud: before this, a neutralised file was replaced with a
        # marker and the run gave no sign it had happened.
        for doc in neutralised:
            print(
                f"Neutralised at ingestion (it instructs a model): "
                f"{doc.metadata.get('origin')}/{doc.metadata.get('source_file')} "
                f"- kept as a redaction marker at tier {doc.metadata.get('trust_tier')}"
            )
    # Said at ingestion, not only when a question happens to touch it: two
    # trusted sources that disagree is something to fix in the sources.
    for key, found in conflicts(facts_from_nodes(documents, GROUNDING_MIN_TIER)).items():
        print(describe(key, found))
    return documents


def _relative(path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def review_drifted_origins(
    origins: list[Origin],
    decisions_dir: Path,
    now: datetime | None = None,
    rules_changed: bool = False,
) -> list[ReviewDecision]:
    """Re-review only what moved; carry every settled decision forward unchanged.

    Prompting for every origin on every run trains the reviewer to press
    Enter, which is how a drifted tier gets waved through. Asking only about
    what actually changed keeps the prompt worth reading.
    """
    now = now or datetime.now(UTC)
    previous = load_latest_decisions(decisions_dir)

    to_review, carried = [], []
    for origin in origins:
        reason = needs_review(origin, previous.get(origin.name), now)
        if reason is None:
            carried.append(previous[origin.name])
        else:
            print(f"\nRe-reviewing {origin.name}: {reason}")
            to_review.append(origin)

    if carried and not to_review:
        print(f"\nNo origins changed; reusing {len(carried)} existing decision(s).")

    return carried + run_review_gate(to_review, previous, rules_changed)


def build_index(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    decisions_dir: Path = DEFAULT_DECISIONS_DIR,
    trust_rules_path: Path = DEFAULT_TRUST_RULES_PATH,
    classifier: Classifier | None = None,
) -> VectorStoreIndex:
    """Review, screen and index the corpus. Shared by both query engines below.

    classifier is the sanitizer's optional stage 2, e.g. an
    injection_classifier.InjectionClassifier(). None screens with the
    patterns alone, which needs no torch install.

    Paths may be str or Path, relative to the current folder or absolute.
    """
    corpus_dir, decisions_dir, trust_rules_path = Path(corpus_dir), Path(decisions_dir), Path(trust_rules_path)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"No corpus folder at {corpus_dir.resolve()}. Make one with a subfolder per source "
            f"(e.g. {corpus_dir}/wiki/, {corpus_dir}/inbox/), or pass corpus_dir=."
        )
    rules = load_rules(trust_rules_path)
    origins = discover_origins(corpus_dir, rules)
    rules_hash = rules_sha256(trust_rules_path)
    recorded = load_latest_rules_hash(decisions_dir)
    rules_changed = recorded is not None and recorded != rules_hash
    decisions = review_drifted_origins(origins, decisions_dir, rules_changed=rules_changed)
    save_decisions(decisions, decisions_dir, trust_rules_sha256=rules_hash)

    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()

    documents = load_documents(corpus_dir, decisions, classifier=classifier)

    # "No usable content", not "no documents". The sanitizer neutralises a
    # hostile document rather than removing it -- it has to, or deleting the
    # document would delete the tier that withholds tools -- so a corpus of
    # nothing but instruction documents now survives as a pile of redaction
    # markers. Counting documents would miss that, and the reason for
    # refusing has not changed: an index with no content reaches the model
    # with nothing to ground on, and it answers from memory instead.
    usable = [d for d in documents if d.text.strip() != REDACTION_MARKER]
    if not usable:
        raise ValueError(
            f"Nothing left to index from {corpus_dir}: every document was excluded by the "
            "review gate or neutralised as an instruction document. Refusing to build an "
            "index with no content, because querying one reaches the model with nothing to "
            "ground on and it answers from memory instead -- ungrounded, with nothing to "
            "say so."
        )

    return VectorStoreIndex.from_documents(documents)


def build_plain_query_engine(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    decisions_dir: Path = DEFAULT_DECISIONS_DIR,
    trust_rules_path: Path = DEFAULT_TRUST_RULES_PATH,
    similarity_top_k: int = 4,
    classifier: Classifier | None = None,
):
    """The reviewed pipeline with an UNGROUNDED answer. For comparisons, not for users.

    The capability gate stops what a turn can do, not what an answer says:
    measured (M6), attack text reached 3/9 of qwen3:8b's plain answers and
    1/9 of llama3.2:3b's with every other defence on, and 0/10 of the
    grounded ones. build_query_engine below is the grounded one. This is
    kept for the milestone demos and checks that measure M1-M4 on their own.

    Retrieved chunks are screened again at query time. Ingestion screens
    whole documents; this catches a chunk that only reads as an instruction
    on its own, and anything indexed by a path that skipped ingestion.
    """
    index = build_index(corpus_dir, decisions_dir, trust_rules_path, classifier)
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[InstructionSanitizerPostprocessor(classifier=classifier)],
    )


def build_query_engine(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    decisions_dir: Path = DEFAULT_DECISIONS_DIR,
    trust_rules_path: Path = DEFAULT_TRUST_RULES_PATH,
    similarity_top_k: int = 4,
    classifier: Classifier | None = None,
    judge=None,
) -> GroundedQueryEngine:
    """The reviewed pipeline. .query() answers from trusted sources or declines (M5).

    The default, because it is the only answer M6 found protected whatever
    the model does. build_plain_query_engine is the ungrounded one.

    judge is the optional entailment model, e.g. nli.EntailmentJudge(); None
    checks answers lexically only, which needs no torch install.
    """
    index = build_index(corpus_dir, decisions_dir, trust_rules_path, classifier)
    return GroundedQueryEngine(
        index,
        Settings.llm,
        similarity_top_k=similarity_top_k,
        node_postprocessors=[InstructionSanitizerPostprocessor(classifier=classifier)],
        judge=judge,
    )


# The name this had before grounding became the default.
build_grounded_query_engine = build_query_engine
