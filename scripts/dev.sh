#!/usr/bin/env bash
# Serve the whole FrugalRoute stack (the `make dev` equivalent, for systems without
# make). uvicorn (api) in the background + the app static server in the foreground
# with a same-origin /api reverse-proxy. Ctrl-C stops both.
#
#   scripts/dev.sh [api_port] [app_port]
set -euo pipefail

API_PORT="${1:-8000}"
APP_PORT="${2:-5500}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "FrugalRoute dev stack:  app http://localhost:${APP_PORT}/   (api on :${API_PORT})"

python -m uvicorn frugalroute_api.app:app --port "${API_PORT}" &
API_PID=$!
trap 'kill "${API_PID}" 2>/dev/null || true' EXIT

FRUGALROUTE_API_PROXY="http://localhost:${API_PORT}" \
  node "${REPO}/app/tests/e2e/static-server.mjs" "${APP_PORT}"
