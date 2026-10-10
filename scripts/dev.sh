#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f "$HOME/.nvm/nvm.sh" ]; then
  source "$HOME/.nvm/nvm.sh"
  nvm use 22
fi
# Each owned server has a separate process group for complete shutdown.
setsid .venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
backend_pid=$!
setsid bash -c 'cd frontend; exec npm run dev' &
frontend_pid=$!
cleanup() { kill -TERM -- "-$backend_pid" "-$frontend_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
wait -n "$backend_pid" "$frontend_pid"