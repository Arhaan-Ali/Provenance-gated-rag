"""Where this repository keeps its demo corpora, decisions and trust rules.

The package's own defaults are relative to the caller's folder, so that a
pip-installed copy works in any project. The scripts here run against the
repository's data, wherever they are started from, so they pass these.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M0_CORPUS = ROOT / "data" / "m0_corpus"
M1_CORPUS = ROOT / "data" / "m1_corpus"
DECISIONS = ROOT / "data" / "decisions"
TRUST_RULES = ROOT / "trust.yaml"
