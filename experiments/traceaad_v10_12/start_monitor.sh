#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="v1012_monitor"
PORT="${1:-8765}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SESSION}"
fi
cd "${ROOT}"
CMD="${ROOT}/.venv/bin/python -u -m experiments.traceaad_v10_12.monitor --host 0.0.0.0 --port ${PORT}"
tmux new-session -d -s "${SESSION}" "${CMD}"
sleep 2
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "TraceAAD V10.12 monitor: http://0.0.0.0:${PORT} (use this machine's reachable IP)"
    echo "Attach: tmux attach -t ${SESSION}"
else
    echo "Monitor failed to start"
    exit 1
fi
