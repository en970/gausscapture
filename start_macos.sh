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
SCHEME="http"
SSL_ARGS=()

# Browsers only grant camera access in a "secure context": HTTPS, or localhost.
# A plain-HTTP LAN address is neither, so the phone capture PWA cannot open the
# camera when reached at http://192.168.x.x:7860/mobile/ -- getUserMedia is
# refused before any of our code runs. Serving the LAN interface therefore
# requires TLS, even with a self-signed certificate.
if [ "${GAUSSCAPTURE_HTTPS:-}" = "1" ] || { [ "$HOST" = "0.0.0.0" ] && [ "${GAUSSCAPTURE_HTTPS:-1}" != "0" ]; }; then
  CERT_DIR="${GAUSSCAPTURE_CERT_DIR:-.certs}"
  mkdir -p "$CERT_DIR"
  if [ ! -f "$CERT_DIR/dev.crt" ] || [ ! -f "$CERT_DIR/dev.key" ]; then
    LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
    echo "Generating a self-signed development certificate for $LAN_IP ..."
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
      -keyout "$CERT_DIR/dev.key" -out "$CERT_DIR/dev.crt" \
      -subj "/CN=$LAN_IP" \
      -addext "subjectAltName=IP:$LAN_IP,IP:127.0.0.1,DNS:localhost" 2>/dev/null
    echo "Certificate written to $CERT_DIR/ (git-ignored, development only)."
  fi
  SSL_ARGS=(--ssl-keyfile "$CERT_DIR/dev.key" --ssl-certfile "$CERT_DIR/dev.crt")
  SCHEME="https"
fi

python -m uvicorn backend.main:app --host "$HOST" --port "$PORT" "${SSL_ARGS[@]}" &
BACKEND_PID=$!
cleanup() { kill "$BACKEND_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [ "$HOST" = "0.0.0.0" ]; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
  echo
  echo "Backend:      $SCHEME://$LAN_IP:$PORT"
  echo "Phone capture: $SCHEME://$LAN_IP:$PORT/mobile/"
  if [ "$SCHEME" = "https" ]; then
    echo
    echo "The certificate is self-signed, so the phone will warn once."
    echo "  iOS Safari: Show Details -> visit this website"
    echo "  Android Chrome: Advanced -> Proceed"
    echo "Camera access requires this; browsers refuse it over plain HTTP."
  fi
  echo
else
  echo "Backend is running at $SCHEME://$HOST:$PORT"
fi

if [ -d "frontend/node_modules" ]; then
  (cd frontend && npm run dev)
else
  echo "Install frontend dependencies with: cd frontend && npm install"
  wait "$BACKEND_PID"
fi
