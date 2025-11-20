#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (this script's directory)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurable via env, with sensible defaults
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
APP_DIR="$ROOT_DIR/app"

echo "Starting services from: $ROOT_DIR"

# Start API
(
  cd "$ROOT_DIR"
  echo "Launching API: uvicorn api.main:app --host $API_HOST --port $API_PORT"
  uvicorn api.main:app --host "$API_HOST" --port "$API_PORT"
) &
API_PID=$!

# Start Web App
(
  cd "$APP_DIR"
  echo "Launching App (npm run dev) in $APP_DIR"
  npm run dev
) &
APP_PID=$!

cleanup() {
  echo "Shutting down..."
  kill "$API_PID" "$APP_PID" 2>/dev/null || true
  wait "$API_PID" "$APP_PID" 2>/dev/null || true
}

trap cleanup INT TERM

# If either process exits (macOS bash doesn't support wait -n), stop the other
EXIT_STATUS=0
while true; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    EXIT_STATUS=1
    break
  fi
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    EXIT_STATUS=1
    break
  fi
  sleep 1
done
echo "One process exited; stopping the rest."
cleanup
exit "$EXIT_STATUS"


