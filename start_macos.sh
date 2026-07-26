#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run ./install_macos.sh first."
  exit 1
fi
source .venv/bin/activate
export PYTHONPATH="$PWD"
HOST="${GAUSSCAPTURE_HOST:-127.0.0.1}"
PORT="${GAUSSCAPTURE_PORT:-7860}"
python -m uvicorn backend.main:app --host "$HOST" --port "$PORT" &
BACKEND_PID=$!
cleanup() {
  kill "$BACKEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT
if [ -d "frontend/node_modules" ]; then
  (cd frontend && npm run dev)
else
  echo "Backend is running at http://$HOST:$PORT"
  echo "Install frontend dependencies with: cd frontend && npm install"
  wait "$BACKEND_PID"
fi
