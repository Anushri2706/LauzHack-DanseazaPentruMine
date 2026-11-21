#!/usr/bin/env bash
# One command: engine (FastAPI) on :8000 + town client (Vite) on :5173. Ctrl+C stops both.
set -e
cd "$(dirname "$0")"
if [ ! -d backend/.venv ]; then
  python3 -m venv backend/.venv && backend/.venv/bin/pip install -q -r backend/requirements.txt
fi
[ -d client/node_modules ] || (cd client && npm install)
(cd backend && .venv/bin/uvicorn app.main:app --port 8000) &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT
cd client && npx vite --port 5173 --open
