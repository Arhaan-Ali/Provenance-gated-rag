"""Pluggable settings, loaded from environment / .env."""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama by default, matching the collaborator's fork so retrieval scores are
# comparable between the two repositories. It also lets Ollama manage one
# memory pool for the LLM and the embedder instead of two processes competing
# for it, which is what forced CPU embedding under the HuggingFace path.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# The HuggingFace path is kept selectable, not for parity but for provenance:
# the retrieval scores quoted in README.md and docs/DEVELOPMENT.md from before
# this switch were measured with BAAI/bge-small-en-v1.5, and a claim nobody can
# re-run is not much of a claim. Set EMBEDDING_PROVIDER=huggingface with
# EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 to reproduce those numbers; it needs
# requirements/huggingface.txt installed.
HUGGINGFACE_EMBEDDING_DEVICE = os.getenv("HUGGINGFACE_EMBEDDING_DEVICE", "cpu")


def get_llm():
    """Return the configured LLM instance. Only 'ollama' is wired up so far."""
    if LLM_PROVIDER == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=300.0)
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def get_embed_model():
    """Return the configured embedding model."""
    if EMBEDDING_PROVIDER == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    if EMBEDDING_PROVIDER == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Defaults to CPU. On this machine Ollama's LLM holds most of the
        # unified memory's Metal allocation, and sentence-transformers on MPS
        # crashes with a GPU OOM rather than falling back. Running the embedder
        # through Ollama instead avoids the conflict entirely, which is the main
        # practical reason that is now the default.
        return HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL, device=HUGGINGFACE_EMBEDDING_DEVICE
        )

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def ollama_problems(base_url: str | None = None, models: tuple[str, ...] | None = None) -> list[str]:
    """What stops a live run from working, checked before anything else runs.

    Empty if Ollama answers and has every model the run needs. A demo that
    found out at query time used to crash after the review gate had already
    signed and saved the reviewer's decisions.
    """
    import json
    import urllib.error
    import urllib.request

    base_url = base_url or OLLAMA_BASE_URL
    if models is None:
        models = (OLLAMA_MODEL,) + ((EMBEDDING_MODEL,) if EMBEDDING_PROVIDER == "ollama" else ())
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            listed = json.load(response).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return [(
            f"Ollama is not running at {base_url}. Start it (open the Ollama app, or run "
            "`ollama serve`), then re-run."
        )]
    have = {m.get("name", "").removesuffix(":latest") for m in listed}
    return [
        f"Ollama has no model {m!r}. Run `ollama pull {m}`, then re-run."
        for m in models
        if m.removesuffix(":latest") not in have
    ]


def require_ollama(models: tuple[str, ...] | None = None) -> None:
    """Exit with the reason if a live run cannot work. Call it first, in scripts."""
    if LLM_PROVIDER != "ollama":
        return
    problems = ollama_problems(models=models)
    if problems:
        raise SystemExit("\n".join(["Cannot start: nothing was reviewed, saved or indexed."] + problems))
