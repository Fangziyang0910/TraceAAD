"""Small append-only fact store and one serial, reserved resource ledger."""

from dataclasses import dataclass, field

from traceaad.v10_13.storage import RunStorage, read_journal
from .edits import source_id


class Facts(RunStorage):
    TABLES = ("artifact", "anchor", "attempt", "evaluation", "request", "revision", "comparison", "session")

    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.tables = {name: {} for name in self.TABLES}
        self.state = None
        for row in read_journal(self.path):
            if row["kind"] in self.tables:
                self.tables[row["kind"]][row["data"]["id"]] = row["data"]
            if "state" in row:
                self.state = row["state"]

    def add(self, table, data):
        if data["id"] in self.tables[table]:
            raise ValueError(f"immutable {table} record already exists: {data['id']}")
        self._append({"kind": table, "data": data})
        self.tables[table][data["id"]] = data
        return data

    def artifact(self, code, environment):
        key = source_id(environment + "\n" + code)
        if key not in self.tables["artifact"]:
            self.add("artifact", {"id": key, "code": code, "source_sha256": source_id(code),
                                  "environment": environment})
        return key

    def code(self, anchor):
        return self.tables["artifact"][anchor["artifact_id"]]["code"]

    def checkpoint(self, state):
        self.save_state(state)
        self.state = state


@dataclass
class Ledger:
    candidate_limit: int
    evaluation_limit: int
    candidates: int = 0
    evaluations: int = 0
    calls: int = 0
    tokens: int = 0
    search_limit: int = 0
    channel_used: dict = field(default_factory=lambda: {"init": 0, "main": 0, "trial": 0, "recheck": 0})
    reservation: dict | None = None

    def can_reserve(self, channel, count, repeats, caps):
        if self.reservation is not None:
            return False
        return (self.candidates + count <= self.candidate_limit
                and self.evaluations + count * repeats <= self.evaluation_limit
                and (channel not in caps or self.channel_used[channel] + count <= caps[channel]))

    def reserve(self, channel, count, repeats, caps):
        if not self.can_reserve(channel, count, repeats, caps):
            raise ValueError("cannot reserve a complete session")
        self.reservation = {"channel": channel, "candidates": count, "evaluations": count * repeats}

    def consume_candidate(self):
        if not self.reservation or self.reservation["candidates"] < 1:
            raise RuntimeError("unreserved candidate")
        self.reservation["candidates"] -= 1
        self.candidates += 1
        self.channel_used[self.reservation["channel"]] += 1

    def consume_evaluation(self):
        if not self.reservation or self.reservation["evaluations"] < 1:
            raise RuntimeError("unreserved evaluation")
        self.reservation["evaluations"] -= 1
        self.evaluations += 1

    def release(self):
        unused = self.reservation
        self.reservation = None
        return unused
