#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN=python3.10
  else
    PYTHON_BIN=python3
    echo "Python 3.10 or 3.11 is recommended. Falling back to $(python3 --version)."
  fi
fi
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if command -v npm >/dev/null 2>&1; then
  (cd frontend && npm install)
else
  echo "npm was not found. Install Node.js 18+ to run the React frontend."
fi
echo "Install complete. Run ./start_macos.sh"
