"""Incremental curves from the single canonical event format."""

from collections import Counter
from pathlib import Path
from threading import RLock
import json
import math

from traceaad.storage import committed_size, read_json
from experiments.infra.result_storage import Programs


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


class TrainingHistory:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.lock = RLock()
        self.sources = Programs(run_dir)
        self.identity, self.offset, self.stamp = None, 0, None
        self.boundary_tail = None
        self.events, self.clock, self.node_offsets = [], {}, {}
        self.programs = {}
        self.recorded_budget = 0
        self.progress = {}
        self.result = ([], [], {}, {})
        self.config = read_json(self.run_dir / "run_config.json", {})

    def read(self):
        with self.lock:
            path = self.run_dir / "events.jsonl"
            if not path.exists():
                return self.result
            stat = path.stat()
            committed = committed_size(self.run_dir)
            identity = (stat.st_dev, stat.st_ino)
            metadata = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None
                             for p in (self.run_dir / 'resume.json', self.run_dir / 'summary.json'))
            stamp = (committed, stat.st_mtime_ns, metadata)
            # A rewritten journal may reuse its inode. Check the committed
            # boundary before treating its contents as an appended tail.
            with path.open("rb") as handle:
                handle.seek(max(0, self.offset - 128))
                tail = handle.read(min(128, self.offset)).hex()
            replaced = self.boundary_tail is not None and tail != self.boundary_tail
            if identity != self.identity or committed < self.offset or replaced:
                self.offset, self.identity = 0, identity
                self.events, self.programs, self.node_offsets = [], {}, {}
                self.clock, self.stamp = {}, None
                self.recorded_budget = 0
                self.boundary_tail = None
            if stamp == self.stamp:
                return self.result
            previous_offset = self.offset
            with path.open("rb") as handle:
                handle.seek(self.offset)
                while handle.tell() < committed:
                    raw = handle.readline(committed - handle.tell())
                    if not raw.endswith(b"\n"):
                        break
                    row = json.loads(raw)
                    data = row.get('data', {})
                    if data.get('counts_budget') is True:
                        self.recorded_budget = max(self.recorded_budget, data.get('sample_order', 0))
                        if data.get('event') == 'sample_registered' and self.events:
                            last = self.events[-1]
                            if last['candidate'] == row.get('profiler_sample_order'):
                                last['evaluation'] = data['sample_order']
                    if row.get("program"):
                        program = row["program"]
                        self.programs[str(program["id"])] = program
                        if row["kind"] != "candidate":
                            for event in self.events:
                                if event.get("node_id") == program["id"] and event["status"] == "pending":
                                    event.update(status=program["status"], valid=program["valid"], fitness=program["fitness"])
                    if row["kind"] in {"candidate", "delivery_failed"}:
                        self.node_offsets[str(row["candidate_id"])] = row.get("node_id")
                        attempt = row.get("attempt") or row.get("program") or {}
                        self.events.append({
                            "evaluation": row["budget_used"], "candidate": row["candidate_id"],
                            "node_id": row.get("node_id"), "fitness": row["fitness"],
                            "operator": row["operator"], "status": row["status"], "valid": row["valid"],
                            **{k: attempt.get(k) for k in ("reason", "channel", "parent_id", "repair_of", "created_at")},
                            "idea": (attempt.get("idea") or "")[:280]})
                    if row.get("progress"):
                        progress = row["progress"]
                        if row["kind"] in {"candidate", "delivery_failed"} or not self.clock:
                            self.clock = {"completed": progress["attempts"], "elapsed": progress.get("elapsed"),
                                          "started_at": progress.get("started_at"), "completed_at": row["ts"]}
                        self.clock["phase"] = progress["phase"]
                    self.offset = handle.tell()
                handle.seek(max(0, self.offset - 128))
                self.boundary_tail = handle.read(min(128, self.offset)).hex()
            if self.stamp is not None and self.offset == previous_offset and self.stamp[2] == metadata:
                self.stamp = stamp
                return self.result
            self.stamp = stamp
            self.result = self._build()
            return self.result

    def timing_snapshot(self):
        with self.lock:
            self.read()
            return dict(self.clock)

    def progress_snapshot(self):
        with self.lock:
            self.read()
            return dict(self.progress)

    def node(self, identifier, *, by_node=False):
        with self.lock:
            self.read()
            node_id = identifier if by_node else self.node_offsets.get(str(identifier))
            program = self.programs.get(str(node_id))
            if program is None:
                return None
            code = self.sources.get(program["id"])
            return {**program, "code": code or ""}

    def _build(self):
        events = sorted(self.events, key=lambda e: e["evaluation"])
        successful = [event for event in events if event["valid"]]
        node_ids = {str(e.get("node_id") if e.get("node_id") is not None else e.get("candidate"))
                    for e in successful if e.get("node_id") is not None or e.get("candidate") is not None}
        unit = self.config.get("budget_axis", "候选尝试")
        state = read_json(self.run_dir / 'resume.json', {}).get('method_state', {})
        summary = read_json(self.run_dir / 'summary.json', {})
        used = max(self.recorded_budget, state.get('sample_count', state.get('sample_order', 0)),
                   summary.get('method_sample_count', 0),
                   max((e['evaluation'] for e in events), default=0))
        self.progress = {"source": "events", "x_label": unit,
                         "budget_used": used,
                         "valid_nodes": len(node_ids),
                         "candidate_count": len(events), "valid_candidate_count": len(successful),
                         "valid_rate": len(successful) / len(events) if events else None}
        best = None
        points, recent = [], []
        operators, outcomes = Counter(), Counter()
        for event in events:
            score = event["fitness"]
            operators[event["operator"]] += 1
            outcomes[str(event["status"])] += 1
            recent.append(event)
            if event["valid"] and score is not None and (best is None or score < best):
                point = {**event,
                         "kind": "initial" if best is None else "breakthrough",
                         "gain": None if best is None else best - score}
                points.append(point)
                best = score
        # Extend the incumbent through failures and regressions, without
        # misattributing its fitness/operator to the last candidate.
        if points and used > points[-1]["evaluation"]:
            points.append({"evaluation": used, "fitness": best, "kind": "progress"})
        self.progress["best_fitness"] = best
        return points, recent[-12:][::-1], dict(operators), dict(outcomes)
