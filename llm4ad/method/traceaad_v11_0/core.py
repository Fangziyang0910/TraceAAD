"""Persistence primitives and the single-parent search tree for TraceAAD V11.0."""

from dataclasses import dataclass
import hashlib
import json
import os


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_code(text: str) -> str:
    return "\n".join(text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


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


class UnknownEvaluation(RuntimeError):
    pass


@dataclass
class Node:
    id: int
    code: str
    idea: str
    fitness: float
    evaluation_id: int | None = None
    parent_id: int | None = None
    operator: str = "Init"
    donor_id: int | None = None


class SearchTree:
    def __init__(self):
        self.nodes = {}
        self.children = {}
        self.roots = []
        self.next_id = 0

    def _attach(self, node):
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.roots.append(node.id)
        else:
            self.children.setdefault(node.parent_id, []).append(node.id)

    def add(self, *, code, idea, fitness, evaluation_id, parent_id, operator, donor_id=None):
        node = Node(self.next_id, code, idea, fitness, evaluation_id, parent_id, operator, donor_id)
        self.next_id += 1
        self._attach(node)
        return node

    def add_raw(self, node):
        self._attach(node)
        self.next_id = max(self.next_id, node.id + 1)

    def all_nodes(self):
        return list(self.nodes.values())

    def best(self):
        return max(self.nodes.values(), key=lambda node: node.fitness)
