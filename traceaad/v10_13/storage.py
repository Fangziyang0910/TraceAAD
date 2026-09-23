"""Run journals (append-only analysis records) and atomic file writes."""

import hashlib
import json
import os
import fcntl
from contextlib import contextmanager
from dataclasses import asdict


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    """Preserve torn bytes separately before restoring the complete journal prefix."""
    if not path.exists():
        return
    with path.open("rb") as handle:
        data = handle.read()
    if data and not data.endswith(b"\n"):
        prefix = _complete_prefix(data)
        tail = data[len(prefix):]
        backup = path.with_name(path.name + '.torn-' + hashlib.sha256(tail).hexdigest()[:16])
        with backup.open('wb') as handle:
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with path.open("r+b") as handle:
            handle.truncate(len(prefix))
            handle.flush()
            os.fsync(handle.fileno())


class RunStorage:
    """Own call, evaluation, node and event journals plus the checkpoint."""

    def __init__(self, run_dir):
        self.events_path = run_dir / "events.jsonl"
        self.llm_calls_path = run_dir / "llm_calls.jsonl"
        self.nodes_path = run_dir / "nodes.jsonl"
        self.state_path = run_dir / "tree_state.json"
        self.summary_path = run_dir / "logs" / "run_summary.json"
        self.evaluations_path = run_dir / 'evaluations.jsonl'
        self.lock_path = run_dir / '.writer.lock'
        self.last_event = None
        self.last_response = None
        self._indexes = {}

    @contextmanager
    def writer_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('another writer owns this run directory') from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def index(self, path, key):
        if path not in self._indexes:
            index = {}
            for record in read_journal(path):
                identity = record[key]
                if identity in index:
                    raise ValueError(f'duplicate {key} in {path.name}: {identity}')
                index[identity] = record
            self._indexes[path] = index
        return self._indexes[path]

    def append_once(self, path, record, key):
        index = self.index(path, key)
        old = index.get(record[key])
        if old is not None:
            if old != record:
                raise ValueError(f'conflicting {key} in {path.name}: {record[key]}')
            return
        self.append_record(path, record)
        index[record[key]] = record

    def append_record(self, path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_call(self, record):
        self.append_once(self.llm_calls_path, record, 'call_id')
        if "response" in record:
            self.last_response = (record["candidate_id"], record["response"])

    def record_node(self, node):
        self.append_once(self.nodes_path, asdict(node), 'id')

    def record_event(self, record):
        self.append_once(self.events_path, record, 'candidate_id')
        self.last_event = record

    def record_evaluation(self, record):
        self.append_once(self.evaluations_path, record, 'evaluation_id')

    def failed_response(self, candidate_id):
        if self.last_response and self.last_response[0] == candidate_id:
            return self.last_response[1]
        return next(record["response"] for record in reversed(read_journal(self.llm_calls_path))
                    if record.get("candidate_id") == candidate_id and "response" in record)
