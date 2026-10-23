"""The real entailment model on data/meaning_eval.json. Runs only where it is installed.

Skipped in CI, which does not install torch. Locally: install
requirements/nli.txt and run scripts/run_meaning_eval.py once to download
the pinned weights; after that this runs offline.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="requirements/nli.txt not installed")


@pytest.fixture(scope="module")
def results():
    from huggingface_hub import try_to_load_from_cache

    from provenance_gated_rag.nli import DEFAULT_MODEL, EntailmentJudge

    judge = EntailmentJudge()
    if not isinstance(try_to_load_from_cache(DEFAULT_MODEL, "config.json", revision=judge.revision), str):
        pytest.skip("pinned NLI weights not downloaded; run scripts/run_meaning_eval.py once")
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_meaning_eval

    return run_meaning_eval.run(judge)


def test_no_supported_claim_is_declined(results):
    assert [r["claim"] for r in results["supported"] if r["declined"]] == []


def test_every_contradicted_claim_is_declined(results):
    assert [r["claim"] for r in results["contradicted"] if not r["declined"]] == []


def test_every_conflict_between_trusted_sources_is_declined(results):
    assert [r["claim"] for r in results["conflict"] if not r["declined"]] == []
