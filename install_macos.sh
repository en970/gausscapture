#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
echo "Using $($PYTHON_BIN --version)"

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Installs the gausscapture core, its CLI, and the server extra. Editable so
# that edits to src/ take effect without reinstalling.
python -m pip install -e ".[server]"

if command -v npm >/dev/null 2>&1; then
  (cd frontend && npm install)
else
  echo "npm was not found. Install Node.js 18+ to run the React frontend."
fi

echo
echo "Install complete. Check your environment with:"
echo "  .venv/bin/gausscapture doctor"
echo "Then start the app with:"
echo "  ./start_macos.sh"
