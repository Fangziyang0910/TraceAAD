#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${1:-8765}"
SESSION="traceaad_monitor"

if tmux has-session -t "${SESSION}" 2>/dev/null; then tmux kill-session -t "${SESSION}"; fi
cd "${ROOT}"
CMD="${ROOT}/.venv/bin/python -u -m core.training_monitor --results-dir ${ROOT}/experiments/traceaad_v10_11/results --host 0.0.0.0 --port ${PORT}"
tmux new-session -d -s "${SESSION}" "${CMD}"
sleep 2
tmux has-session -t "${SESSION}" 2>/dev/null || { echo "Monitor failed to start"; exit 1; }
echo "TraceAAD training monitor: http://0.0.0.0:${PORT}"
echo "Attach: tmux attach -t ${SESSION}"
