"""Small persistence helpers for V10.13 runs."""

import json
import os
from dataclasses import asdict


def write_json(path, payload):
    """Replace a JSON file after writing the new value beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_journal(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class RunStorage:
    """Save model calls, evaluated nodes, iteration results, and current state."""

    def __init__(self, run_dir):
        self.events_path = run_dir / "events.jsonl"
        self.llm_calls_path = run_dir / "llm_calls.jsonl"
        self.nodes_path = run_dir / "nodes.jsonl"
        self.state_path = run_dir / "tree_state.json"
        self.summary_path = run_dir / "logs" / "run_summary.json"

    @staticmethod
    def append(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def record_call(self, record):
        self.append(self.llm_calls_path, record)

    def record_node(self, node):
        self.append(self.nodes_path, asdict(node))

    def record_event(self, record):
        self.append(self.events_path, record)

    def save_state(self, state):
        write_json(self.state_path, state)

    def save_summary(self, summary):
        write_json(self.summary_path, summary)
