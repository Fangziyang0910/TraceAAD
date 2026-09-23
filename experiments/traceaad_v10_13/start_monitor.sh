#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="traceaad_v1013_monitor"
PORT="${1:-8765}"
BATCH="${2:-}"

if tmux has-session -t "=${SESSION}" 2>/dev/null; then
    echo "monitor session already exists: ${SESSION}" >&2
    exit 1
fi

ARGS=(--host 0.0.0.0 --port "${PORT}")
if [[ -n "${BATCH}" ]]; then
    ARGS+=(--version "${BATCH}")
fi

cd "${ROOT}"
tmux new-session -d -s "${SESSION}" \
  "${ROOT}/.venv/bin/python -u -m experiments.traceaad_v10_13.monitor ${ARGS[*]}"
sleep 2
if ! tmux has-session -t "=${SESSION}" 2>/dev/null; then
    echo "monitor failed to start" >&2
    exit 1
fi
echo "TraceAAD V10.13 monitor: http://127.0.0.1:${PORT}"
echo "Attach: tmux attach -t ${SESSION}"
