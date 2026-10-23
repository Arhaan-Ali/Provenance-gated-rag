import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pytest


@pytest.fixture(autouse=True)
def _signing_key_outside_home(tmp_path_factory, monkeypatch):
    """Every test signs decisions with a throwaway key.

    Without this, save_decisions would create ~/.config/pgrag/review.key on
    the developer's machine and record test folders in its heads file.
    """
    monkeypatch.setenv("PGRAG_KEY_DIR", str(tmp_path_factory.mktemp("pgrag-key")))
