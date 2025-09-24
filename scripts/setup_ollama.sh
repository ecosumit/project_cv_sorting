#!/usr/bin/env bash
set -euo pipefail
echo "Pulling Ollama models..."
ollama pull llama3:8b || true
ollama pull nomic-embed-text || true
echo "Done."
