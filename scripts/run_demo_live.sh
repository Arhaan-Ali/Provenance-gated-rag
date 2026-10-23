#!/bin/zsh
# Run the demo live against Ollama, and stop Ollama afterwards.
#
# Starts `ollama serve` only if nothing is already listening, and stops only
# the server it started -- on success, on failure, or on Ctrl-C. Leaving the
# model loaded keeps the GPU busy and the laptop hot long after the demo ends.
set -u
cd "$(dirname "$0")/.."

OLLAMA="${OLLAMA:-$(command -v ollama || echo /Applications/Ollama.app/Contents/Resources/ollama)}"
URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
started=""

stop_ollama() {
  if [[ -n "$started" ]]; then
    kill "$started" 2>/dev/null
    pkill -f "ollama runner" 2>/dev/null
    echo "\nStopped the Ollama server this script started."
  fi
}
trap stop_ollama EXIT INT TERM

if ! curl -s "$URL/api/tags" >/dev/null; then
  echo "Starting Ollama..."
  "$OLLAMA" serve >/tmp/pgrag-ollama.log 2>&1 &
  started=$!
  for _ in {1..30}; do
    curl -s "$URL/api/tags" >/dev/null && break
    sleep 1
  done
  curl -s "$URL/api/tags" >/dev/null || { echo "Ollama did not start; see /tmp/pgrag-ollama.log"; exit 1; }
fi

.venv/bin/python scripts/run_demo.py "$@"
