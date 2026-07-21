#!/usr/bin/env bash
# Hybrid orchestrator: Docker services (GROBID/Qdrant/Ollama) + native Streamlit app.
# Native app => uses MPS/unified memory on Apple Silicon (Docker can't reach the GPU).
#
# Usage: ./orchestrator.sh [run|stop|build]
#   run    (default) start services, wait until ready, launch the native app detached
#   stop   stop the native app and the Docker services
#   build  re-sync the native Python env after pyproject/lock changes (poetry install)
set -euo pipefail
cd "$(dirname "$0")"

# brew installs poetry/python here; ensure they're on PATH when run from Finder
export PATH="/opt/homebrew/bin:$PATH"

APP_PID_FILE=".app.pid"
APP_LOG="app.log"

app_running() { [ -f "$APP_PID_FILE" ] && kill -0 "$(cat "$APP_PID_FILE")" 2>/dev/null; }

cmd="${1:-run}"
case "$cmd" in
  run)
    if app_running; then
      echo "App already running (PID $(cat "$APP_PID_FILE")) on http://localhost:8501"
      exit 0
    fi

    echo "▶ Starting services (GROBID, Qdrant, Ollama)…"
    docker compose up -d grobid qdrant ollama

    # GROBID boot is slow (emulated on ARM); indexing before it's up would fail. Wait.
    printf "  waiting for Qdrant"
    until curl -sf http://localhost:6333/ >/dev/null 2>&1; do printf .; sleep 2; done; echo " ok"
    printf "  waiting for GROBID"
    until curl -sf http://localhost:8070/api/isalive >/dev/null 2>&1; do printf .; sleep 2; done; echo " ok"

    echo "▶ Launching app (native, MPS) detached…"
    nohup poetry run streamlit run zotero_rag/app.py --server.headless=true \
      >"$APP_LOG" 2>&1 &
    echo $! > "$APP_PID_FILE"

    # best-effort wait so we can confirm it's up (bounded; logs in $APP_LOG otherwise)
    for _ in $(seq 1 30); do
      curl -sf -o /dev/null http://localhost:8501 && break || sleep 1
    done
    echo "▶ App on http://localhost:8501 (logs: $APP_LOG). Stop with: ./orchestrator.sh stop"
    ;;
  stop)
    if app_running; then
      echo "▶ Stopping app (PID $(cat "$APP_PID_FILE"))…"
      kill "$(cat "$APP_PID_FILE")" 2>/dev/null || true
    fi
    rm -f "$APP_PID_FILE"
    echo "▶ Stopping services…"
    docker compose stop
    ;;
  build)
    echo "▶ Re-syncing native Python env…"
    poetry install
    ;;
  *)
    echo "Usage: ./orchestrator.sh [run|stop|build]" >&2
    exit 1
    ;;
esac
