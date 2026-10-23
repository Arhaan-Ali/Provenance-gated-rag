"""The pre-flight check: a live run stops before the review gate if Ollama cannot serve it."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from provenance_gated_rag import config
from provenance_gated_rag.config import ollama_problems, require_ollama


@pytest.fixture
def ollama():
    """A stand-in /api/tags, serving whatever model list the test sets."""
    state = {"models": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"models": [{"name": n} for n in state["models"]]}).encode()
            self.send_response(200 if self.path == "/api/tags" else 404)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def _closed_port_url():
    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    return f"http://127.0.0.1:{port}"


def test_nothing_is_wrong_when_every_model_is_there(ollama):
    ollama["models"] = ["qwen3:8b", "nomic-embed-text:latest"]

    assert ollama_problems(ollama["url"], ("qwen3:8b", "nomic-embed-text")) == []


def test_a_server_that_is_not_running_is_reported_with_how_to_start_it():
    problems = ollama_problems(_closed_port_url(), ("qwen3:8b",))

    assert len(problems) == 1
    assert "not running" in problems[0] and "ollama serve" in problems[0]


def test_each_missing_model_is_reported_with_the_pull_command(ollama):
    ollama["models"] = ["qwen3:8b"]

    problems = ollama_problems(ollama["url"], ("qwen3:8b", "nomic-embed-text", "llama3.2:3b"))

    assert problems == [
        "Ollama has no model 'nomic-embed-text'. Run `ollama pull nomic-embed-text`, then re-run.",
        "Ollama has no model 'llama3.2:3b'. Run `ollama pull llama3.2:3b`, then re-run.",
    ]


def test_a_different_size_of_the_same_model_does_not_count(ollama):
    ollama["models"] = ["qwen3:4b"]

    assert ollama_problems(ollama["url"], ("qwen3:8b",))


def test_require_ollama_exits_before_anything_is_reviewed(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", _closed_port_url())

    with pytest.raises(SystemExit, match="nothing was reviewed, saved or indexed"):
        require_ollama(("qwen3:8b",))


def test_require_ollama_passes_when_the_models_are_there(ollama, monkeypatch):
    ollama["models"] = ["qwen3:8b"]
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", ollama["url"])

    require_ollama(("qwen3:8b",))
