#!/usr/bin/env bash
set -euo pipefail

# Setting Configs
export LLM_PROVIDER=deepseek
#export OPENROUTER_API_KEY=
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OR_MODEL_SLUG="${OR_MODEL_SLUG:-deepseek/deepseek-chat-v3-0324:free}"
export PERSIST_DIR="${PERSIST_DIR:-RAG/ProcessedDocuments/chroma_db}"
PORT="${PORT:-8000}"
QUESTION="${1:-What is CAPEX?}"

# Checking API key

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set. Export it and then rerun"
  exit 1
fi

echo "Using LLM_PROVIDER=$LLM_PROVIDER"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "OR_MODEL_SLUG=$OR_MODEL_SLUG"
echo "PERSIST_DIR=$PERSIST_DIR"

# build vector DB
python main.py --mode index --persist_dir "$PERSIST_DIR"

# Starting API server
echo "Starting Uvicorn on port $PORT ..."
uvicorn RAG.api:app --host 0.0.0.0 --port "$PORT" --reload > .server_deepseek.log 2>&1 &
PID=$!
sleep 3

# Sending request
echo "Sending test request to http://127.0.0.1:$PORT/ask"
curl -s -X POST "http://127.0.0.1:$PORT/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"${QUESTION}\", \"k\": 3, \"min_similarity\": 0.28}" | jq .

# Ending
echo "▶ Stopping server (PID $PID)"
kill "$PID" >/dev/null 2>&1 || true
wait "$PID" 2>/dev/null || true
