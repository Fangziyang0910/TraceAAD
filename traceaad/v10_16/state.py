"""Append-only facts and checkpoints: programs, generation events and evaluations."""

from traceaad.v10_13.storage import RunStorage, read_journal


class Facts(RunStorage):
    TABLES = ("artifact", "node", "attempt", "evaluation", "request")

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

    def checkpoint(self, state):
        self.save_state(state)
        self.state = state
