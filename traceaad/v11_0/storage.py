"""Run journals (append-only analysis records) and atomic file writes."""

import hashlib
import json
import os
from dataclasses import asdict


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _complete_prefix(data):
    """Bytes through the end of the last complete record line.

    A torn final write lacks the trailing newline; those bytes are dropped.
    """
    if data and not data.endswith(b"\n"):
        newline = data.rfind(b"\n")
        return data[:newline + 1]
    return data


def read_journal(path):
    if not path.exists():
        return []
    with path.open("rb") as handle:
        data = _complete_prefix(handle.read())
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def truncate_torn_tail(path):
    """Drop a torn final record so later appends stay parseable."""
    if not path.exists():
        return
    with path.open("rb") as handle:
        data = handle.read()
    if data and not data.endswith(b"\n"):
        with path.open("r+b") as handle:
            handle.truncate(len(_complete_prefix(data)))


class RunStorage:
    """Own the three append journals; checkpoints live in tree_state.json."""

    def __init__(self, run_dir):
        self.events_path = run_dir / "events.jsonl"
        self.llm_calls_path = run_dir / "llm_calls.jsonl"
        self.nodes_path = run_dir / "nodes.jsonl"
        self.state_path = run_dir / "tree_state.json"
        self.summary_path = run_dir / "logs" / "run_summary.json"
        self.last_event = None
        self.last_response = None

    def append_record(self, path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def record_call(self, record):
        self.append_record(self.llm_calls_path, record)
        if "response" in record:
            self.last_response = (record["candidate_id"], record["response"])

    def record_node(self, node):
        self.append_record(self.nodes_path, asdict(node))

    def record_event(self, record):
        self.append_record(self.events_path, record)
        self.last_event = record

    def failed_response(self, candidate_id):
        if self.last_response and self.last_response[0] == candidate_id:
            return self.last_response[1]
        return next(record["response"] for record in reversed(read_journal(self.llm_calls_path))
                    if record.get("candidate_id") == candidate_id and "response" in record)
