"""One append-only JSONL journal for a V10.13 search run."""

from __future__ import annotations

import json
import os
from pathlib import Path


JOURNAL_NAME = "search.jsonl"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_journal(path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"incomplete search record at {path}:{number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid search record at {path}:{number}") from exc
            if not isinstance(record, dict) or "kind" not in record:
                raise RuntimeError(f"invalid search record at {path}:{number}")
            yield record


class RunStorage:
    """Persist calls immediately and commit each result with its resume state."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / JOURNAL_NAME
        self.summary_path = Path(run_dir) / "logs" / "run_summary.json"

    def _append(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            while data:
                written = os.write(fd, data)
                if written == 0:
                    raise OSError("could not append search record")
                data = data[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def load_state(self):
        state = None
        for record in read_journal(self.path):
            if record["kind"] in {"state", "evaluation_started", "candidate"} and "state" in record:
                state = record["state"]
        return state

    def load_summary(self):
        if not self.summary_path.exists():
            return None
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    def save_state(self, state):
        self._append({"kind": "state", "state": state})

    def reserve_evaluation(self, state):
        if not state.get("pending_evaluation"):
            raise ValueError("evaluation reservation requires a pending evaluation")
        self._append({"kind": "evaluation_started", "state": state})

    def record_call(self, record):
        self._append({"kind": "call", **record})

    def commit_candidate(self, event, node, state):
        if state.get("pending_evaluation"):
            raise ValueError("completed candidate cannot retain an evaluation reservation")
        self._append({"kind": "candidate", **event, "node": node, "state": state})

    def records(self, table):
        if table not in {"calls", "events", "nodes"}:
            raise ValueError(f"unsupported record table: {table}")
        values = []
        for record in read_journal(self.path):
            if table == "calls" and record["kind"] == "call":
                values.append({key: value for key, value in record.items() if key != "kind"})
            elif record["kind"] == "candidate":
                if table == "events":
                    values.append({key: value for key, value in record.items()
                                   if key not in {"kind", "node", "state"}})
                elif table == "nodes" and record.get("node") is not None:
                    values.append(record["node"])
        return values

    def save_summary(self, summary):
        write_json(self.summary_path, summary)
