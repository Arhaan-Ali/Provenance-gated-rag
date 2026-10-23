import pytest

from pgrag.core.signing import KEY_DIR_ENV


@pytest.fixture(autouse=True)
def _signing_key_dir(tmp_path_factory, monkeypatch):
    """Each test signs with its own throwaway key, never the real one in ~/.config."""
    monkeypatch.setenv(KEY_DIR_ENV, str(tmp_path_factory.mktemp("pgrag-key")))
