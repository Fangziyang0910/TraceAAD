"""Incremental, read-only training history for the shared monitor."""

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from threading import RLock
from traceaad.common.storage import write_json


SOURCES = ("native", "events.jsonl", "evaluations.csv", "samples", "method_events.jsonl",
           "artifacts/candidates.jsonl")
META = ("scope", "action", "reference_mode", "channel", "parent_id", "repair_of", "idea", "created_at")


NODE_ATTEMPT = re.compile(rb'"attempt_id":\s*(\d+)\s*[,}]')
NODE_ID = re.compile(rb'^\{"kind":"node","data":\{"id":(\d+)[,}]')
ARCHIVE_SOURCE = re.compile(rb'"source":\s*"([^"\\]*)"\s*}\s*$')
PROGRAM_SOURCES = {"best_history.jsonl", "best_program.py", "nodes.jsonl", "checkpoints/latest.json"}
SUCCESS = {"ok", "valid", "expanded", "success", "improve", "regress", "plateau"}


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


class TrainingHistory:
    """Retain small event projections; only read bytes appended since last refresh.

    An unfinished final line is retried on the next read. Search checkpoints,
    source code and prompts are never retained in the monitor cache. The parsed
    projection is mirrored to ``logs/monitor_cache.json`` so a fresh process
    resumes at the same journal offset instead of re-parsing gigabytes.
    """

    CACHE_VERSION = 7
    CACHE_NAME = "monitor_cache.json"

    def __init__(self, run_dir: Path, minimize: bool):
        self.run_dir = Path(run_dir)
        self.minimize = minimize
        self.lock = RLock()
        self.identity = None
        self.offset = 0
        self.stamp = None
        self.streams = {}
        self.counters = {}
        self.axes = {}
        self.sample_stamp = None
        self.attempts = {}
        self.clock = {}
        self.node_offsets = {}
        self.sample_locations = {}
        self.progress = {}
        self.archived_best = None
        self.result = ([], [], {}, {})
        self._sidecar_loaded = False
        self._sidecar_offset = -1

    def _load_sidecar(self) -> None:
        try:
            payload = json.loads((self.run_dir / "logs" / self.CACHE_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return
        if payload.get("version") != self.CACHE_VERSION:
            return
        try:
            self.identity = tuple(payload["identity"])
            self.offset = int(payload["offset"])
            self.stamp = tuple(payload["stamp"])
            self.streams = payload["streams"]
            self.counters = payload["counters"]
            self.axes = payload["axes"]
            self.sample_stamp = payload["sample_stamp"]
            self.attempts = payload["attempts"]
            self.clock = payload["clock"]
            self.node_offsets = payload["node_offsets"]
            self.sample_locations = payload["sample_locations"]
            self.progress = payload["progress"]
            self.archived_best = payload["archived_best"]
            self.result = payload["result"]
            self._sidecar_offset = self.offset
        except (KeyError, TypeError, ValueError):
            self.identity, self.offset, self.stamp = None, 0, None
            self.streams, self.attempts, self.clock = {}, {}, {}
            self.counters, self.sample_stamp = {}, None
            self.axes = {}
            self.node_offsets = {}
            self.sample_locations, self.progress, self.archived_best = {}, {}, None
            self.result = ([], [], {}, {})

    def _save_sidecar(self, *, force=False) -> None:
        if (self.offset == self._sidecar_offset and not force) or self.identity is None:
            return
        payload = {"version": self.CACHE_VERSION, "identity": list(self.identity),
                   "offset": self.offset, "stamp": list(self.stamp or ()),
                   "streams": self.streams, "attempts": self.attempts,
                   "counters": self.counters, "sample_stamp": self.sample_stamp,
                   "axes": self.axes,
                   "clock": self.clock, "node_offsets": self.node_offsets,
                   "sample_locations": self.sample_locations, "progress": self.progress,
                   "archived_best": self.archived_best,
                   "result": self.result}
        target = self.run_dir / "logs" / self.CACHE_NAME
        try:
            write_json(target, payload)
            self._sidecar_offset = self.offset
        except OSError:
            pass  # read-only run dir: keep working, just reparse next time

    def timing_snapshot(self):
        with self.lock:
            self.read()
            snapshot = dict(self.clock)
            try:
                completed = datetime.fromisoformat(snapshot.get("completed_at"))
            except (TypeError, ValueError):
                completed = None
            # Copied native checkpoints have no timezone. Their preserved file
            # timestamp is portable; active elapsed time still supplies the rate.
            if (completed is not None and completed.tzinfo is None and self.stamp
                    and finite(snapshot.get("elapsed")) is not None):
                snapshot["completed_at"] = datetime.fromtimestamp(
                    self.stamp[1] / 1e9, timezone.utc).isoformat()
            return snapshot

    def progress_snapshot(self):
        """Progress, curve and candidate statistics use the same event stream."""
        with self.lock:
            self.read()
            return dict(self.progress)

    def read(self):
        with self.lock:
            if not self._sidecar_loaded:
                self._sidecar_loaded = True
                self._load_sidecar()
            path = next((p for p in (self.run_dir / "search.jsonl",
                        self.run_dir / "events.jsonl", self.run_dir / "logs/method_events.jsonl")
                        if p.exists()), None)
            if path is None:
                if self.identity and self.identity[0] != "samples":
                    self.streams, self.sample_stamp, self.node_offsets = {}, None, {}
                    self.axes = {}
                    self.result, self.progress = ([], [], {}, {}), {}
                    self.archived_best, self.offset = None, 0
                self.identity = ("samples", str(self.run_dir), 0)
                self.stamp = ()
                self.clock = {}
                if self._read_samples():
                    self.result = self._build()
                    self._save_sidecar(force=True)
                return self.result
            stat = path.stat()
            identity = (str(path), stat.st_dev, stat.st_ino)
            stamp = (stat.st_size, stat.st_mtime_ns)
            if (identity != self.identity or stat.st_size < self.offset or
                    (stat.st_size == self.offset and self.stamp not in (None, stamp))):
                self.identity, self.offset = identity, 0
                self.streams, self.attempts = {}, {}
                self.counters, self.sample_stamp = {}, None
                self.axes = {}
                self.clock = {}
                self.node_offsets = {}
                self.sample_locations, self.progress, self.archived_best = {}, {}, None
                self.result = ([], [], {}, {})
                self.stamp = None
                self._sidecar_offset = -1
            # Baseline event logs can omit initialization and failed samples.
            # Prefer their complete profiler history, reloading only on change.
            samples_changed = path.name == "method_events.jsonl" and self._read_samples()
            if stamp == self.stamp:
                if samples_changed:
                    self.result = self._build()
                    self._save_sidecar(force=True)
                return self.result
            changed = samples_changed
            last_checkpoint = None
            with path.open("rb") as handle:
                handle.seek(self.offset)
                while handle.tell() < stat.st_size:
                    start = handle.tell()
                    raw = handle.readline(stat.st_size - handle.tell())
                    if not raw.endswith(b"\n"):
                        break
                    if path.name == "search.jsonl":
                        source_match = ARCHIVE_SOURCE.search(raw[-512:])
                        if source_match:
                            source = source_match.group(1).decode("utf-8")
                            if source == "checkpoints/latest.json":
                                self.node_offsets["@checkpoint"] = start
                                self.offset = handle.tell()
                                continue
                            if source not in SOURCES and source not in PROGRAM_SOURCES:
                                self.offset = handle.tell()
                                continue
                    if path.name == "search.jsonl" and raw.startswith(b'{"kind":"state",'):
                        # Decode only the newest checkpoint in this appended chunk.
                        # Its phase changes even when selection consumes no candidates.
                        last_checkpoint = raw
                        self.offset = handle.tell()
                        continue
                    if path.name == "search.jsonl" and raw.startswith((
                            b'{"kind":"request",', b'{"kind":"call",')):
                        self.offset = handle.tell()
                        continue
                    if (path.name == "search.jsonl"
                            and raw.startswith((b'{"kind":"node"', b'{"kind":"evaluation"',
                                                b'{"kind":"artifact"'))
                            and b'"source"' not in raw):
                        # Heavy records the projection never uses; skip the decode,
                        # but index node lines so the best code can be fetched later.
                        if raw.startswith(b'{"kind":"node"'):
                            match = NODE_ATTEMPT.search(raw[-4096:])
                            if match:
                                self.node_offsets[match.group(1).decode()] = start
                            match = NODE_ID.search(raw[:120])
                            if match:
                                self.node_offsets["node:" + match.group(1).decode()] = start
                        self.offset = handle.tell()
                        continue
                    record = json.loads(raw)
                    source = record.get("source")
                    data = record.get("data")
                    if (source in SOURCES and isinstance(data, dict) and data.get("code")
                            and data.get("node_id") is not None):
                        self.node_offsets["node:" + str(data["node_id"])] = start
                    program = record.get("program") or record.get("node")
                    if (record.get("kind") == "candidate" and source is None
                            and isinstance(program, dict)):
                        self.node_offsets[str(record.get("candidate_id"))] = start
                        self.node_offsets["node:" + str(program.get("id"))] = start
                    elif source == "best_history.jsonl":
                        self.node_offsets[str((record.get("data") or {}).get("child_id"))] = start
                        self.node_offsets["node:" + str((record.get("data") or {}).get("child_id"))] = start
                    elif source == "best_program.py":
                        self.node_offsets["@best_program"] = start
                    elif source == "nodes.jsonl":
                        node = record.get("data") or {}
                        self.node_offsets[str(node.get("id"))] = start
                        self.node_offsets["node:" + str(node.get("id"))] = start
                        score = finite(node.get("fitness"))
                        if score is not None and (self.archived_best is None or score > self.archived_best["fitness"]):
                            self.archived_best = {"candidate": node.get("id"), "fitness": score}
                    if record.get("kind") in {"candidate", "state"}:
                        last_checkpoint = None
                    if path.name == "search.jsonl":
                        changed = self._journal_record(record) or changed
                    else:
                        changed = self._append(path.name, record) or changed
                    self.offset = handle.tell()
            if last_checkpoint is not None:
                self._journal_record(json.loads(last_checkpoint))
            self.stamp = stamp
            if changed:
                self.result = self._build()
            self._save_sidecar(force=samples_changed)
            return self.result

    def _read_samples(self):
        paths = sorted(p for p in (self.run_dir / "logs/samples").glob("samples_*.json")
                       if p.name != "samples_best.json")
        if not paths and not self.streams.get("samples") and self.sample_stamp is None:
            self.sample_stamp = []
            return False
        try:
            stamp = []
            for path in paths:
                stat = path.stat()
                stamp.append([path.name, stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino])
            if stamp == self.sample_stamp:
                return False
            records, locations = [], {}
            for path in paths:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    records.extend(data)
                    for index, record in enumerate(data):
                        if isinstance(record, dict):
                            locations[str(record.get("sample_order"))] = [path.name, index]
        except (OSError, ValueError):
            return False  # A profiler file may be in the middle of a write.
        self.streams["samples"] = []
        for record in records:
            self._append("samples", record)
        self.sample_stamp = stamp
        self.sample_locations = locations
        return True

    def node(self, attempt_id, *, by_node=False):
        """Fetch one program by candidate id without scanning the journal."""
        with self.lock:
            self.read()
            location = self.sample_locations.get(str(attempt_id))
            if location:
                try:
                    data = json.loads((self.run_dir / "logs/samples" / location[0]).read_text(encoding="utf-8"))
                    sample = data[location[1]]
                    return {**sample, "id": sample.get("sample_order"),
                            "code": sample.get("program") or sample.get("function"),
                            "fitness": finite(sample.get("score")), "idea": sample.get("algorithm") or ""}
                except (OSError, ValueError, IndexError, TypeError):
                    return None
            key = "node:" + str(attempt_id) if by_node else str(attempt_id)
            offset = self.node_offsets.get(key)
            checkpoint = (offset is None and str(attempt_id).isdigit()
                          and "@checkpoint" in self.node_offsets)
            if checkpoint:
                offset = self.node_offsets["@checkpoint"]
            if offset is None or self.identity is None:
                return None
            try:
                with open(self.identity[0], "rb") as handle:
                    handle.seek(offset)
                    record = json.loads(handle.readline())
            except (OSError, ValueError):
                return None
            if record.get("kind") == "candidate" and record.get("source") is None:
                return record.get("program") or record.get("node")
            data = record.get("data")
            if checkpoint and isinstance(data, dict):
                nodes = data.get("nodes") or []
                if isinstance(nodes, dict):
                    nodes = list(nodes.values())
                return next((n for n in nodes if str(n.get("id")) == str(attempt_id)), None)
            if record.get("source") == "best_program.py" and isinstance(data, str):
                return {"code": data}
            if isinstance(data, dict):
                return {**data, "id": data.get("id", data.get("child_id")),
                        "code": data.get("code") or data.get("program") or ""}
            return None

    def _journal_record(self, record):
        source = record.get("source")
        if source in SOURCES:
            return self._append(source, record.get("data"))
        if source is not None:
            return False
        if record.get("kind") == "attempt":
            data = record.get("data", {})
            self.attempts[data.get("id")] = {k: data.get(k) for k in META}
        elif record.get("kind") == "candidate":
            # The current format commits one attempt, its new program
            # and the recovery state together. The remaining native formats
            # already carry their candidate projection at the top level.
            if "attempt" in record:
                attempt = record["attempt"]
                program = record.get("program") or {}
                record = {**record, **attempt, "candidate_id": attempt["id"],
                          "node_id": attempt.get("program_id"), "node": program,
                          "fitness": program.get("fitness") if attempt["status"] == "valid" else None,
                          "operator": attempt["action"]}
            state = record.get("state") or {}
            clock = {
                "completed": record.get("budget_used"),
                "elapsed": finite(state.get("elapsed")),
                "started_at": state.get("started_at"),
                "completed_at": record.get("ts"),
                "phase": state.get("phase"),
            }
            # V10.15 candidates carry neither timestamps nor state; keep the
            # clock derived from the latest checkpoint instead of blanking it.
            self.clock = {**self.clock, **{k: v if v is not None else self.clock.get(k)
                                           for k, v in clock.items()}}
            return self._append("native", record)
        elif record.get("kind") == "state":
            state = record.get("state") or {}
            if state.get("phase"):
                self.clock["phase"] = state["phase"]
            # V10.15 candidates carry no timestamps, but every checkpoint has
            # cumulative attempts, active time and the start; derive the clock.
            completed = state.get("attempts")
            if isinstance(completed, (int, float)) and state.get("started_at"):
                try:
                    started = datetime.fromisoformat(state["started_at"])
                except (TypeError, ValueError):
                    started = None
                if started is not None:
                    self.clock["completed"] = int(completed)
                    self.clock["started_at"] = state["started_at"]
                    elapsed = finite(state.get("elapsed"))
                    if elapsed is not None:
                        self.clock["elapsed"] = elapsed
                        self.clock["completed_at"] = (
                            started + timedelta(seconds=elapsed)).isoformat()
        return False

    def _append(self, source, event):
        if not isinstance(event, dict):
            return False
        axis = "评价次数"
        if source == "evaluations.csv":
            index, score = event.get("slot", event.get("eval_count")), event.get("fitness")
            if "slot" in event:
                axis = "预算槽位"
        elif source == "artifacts/candidates.jsonl":
            # Candidate order also includes attempts which never called the evaluator.
            index = (self.counters.get(source, 0) + int(event["evaluator_called"] is True)
                     if "evaluator_called" in event else event.get("order"))
            score = event.get("child_fitness")
            if "evaluator_called" not in event:
                axis = "候选序号"
        elif source == "samples":
            index, score = event.get("sample_order"), event.get("score")
            axis = "样本次数"
        elif source == "method_events.jsonl":
            axis = "样本次数"
            if event.get("event") == "epoch":
                index, score = event.get("sample_count"), event.get("best_perf")
            elif event.get("event") == "sample_registered":
                index, score = event.get("profiler_sample_order"), event.get("score")
            elif event.get("event") == "expand":
                index = event.get("sample_order", event.get("profiler_sample_order"))
                score = event.get("child_score")
            else:
                return False
        else:
            index = event.get("budget_used", event.get("evaluation_id"))
            score = event.get("fitness")
            if "slot_consumed" in event:
                axis = "预算槽位"
            if index is None and source == "events.jsonl":
                if "slot_consumed" in event:
                    consumed = event["slot_consumed"] is True
                elif "step" in event and event.get("status") in {"ok", "eval_failed", "invalid_output"}:
                    # V10.2 logs the same step for every initialization attempt.
                    # Only parsed programs (including failed evaluations) consume budget.
                    consumed = event["status"] != "invalid_output"
                else:
                    return False
                index = self.counters.get(source, 0) + int(consumed)
        position = finite(index)
        if position is None or position < 0 or not position.is_integer():
            return False
        self.counters[source] = int(position)
        self.axes.setdefault(source, axis)
        node = event.get("node") or {}
        candidate = event.get("candidate_id", event.get("id"))
        if candidate is None:
            candidate = event.get("child_id", event.get("sample_order", event.get("order")))
        meta = {k: event.get(k, node.get(k)) for k in META}
        if source == "native":
            meta.update(self.attempts.pop(candidate, {}))
        operator = (meta.get("scope") or meta.get("action") or event.get("operator")
                    or event.get("action") or event.get("intent") or node.get("operator") or "unknown")
        reference = meta.get("reference_mode")
        if reference and reference != "None":
            operator = f"{operator} · {reference}"
        status = event.get("status") or event.get("outcome") or "unknown"
        if source == "samples":
            status = "ok" if finite(score) is not None else "eval_failed"
        valid = finite(score) is not None and (status in SUCCESS or status == "unknown")
        if source == "artifacts/candidates.jsonl":
            valid = finite(score) is not None and event.get("evaluator_called", True) is True
            if finite(score) is not None and event.get("evaluator_called") is False:
                status = "duplicate"
        self.streams.setdefault(source, []).append({
            "evaluation": int(position), "candidate": candidate,
            "node_id": event.get("node_id", node.get("id")),
            "fitness": finite(score), "operator": str(operator),
            "status": status,
            "valid": valid,
            "reason": event.get("reason") or event.get("error_type"),
            **{k: meta.get(k) for k in ("channel", "parent_id", "repair_of", "created_at")},
            "idea": str(meta.get("idea") or event.get("algorithm") or "")[:280],
        })
        return True

    def _build(self):
        source = next((s for s in SOURCES if any(e["fitness"] is not None
                       for e in self.streams.get(s, []))), None)
        if source is None:
            source = next((s for s in SOURCES if self.streams.get(s)), None)
        events = sorted(self.streams.get(source, []), key=lambda e: e["evaluation"])
        successful = [event for event in events if event["valid"]]
        node_ids = {str(e.get("node_id") if e.get("node_id") is not None else e.get("candidate"))
                    for e in successful if e.get("node_id") is not None or e.get("candidate") is not None}
        # Missing identifiers in legacy logs still represent individual evaluations.
        unidentified = sum(e.get("node_id") is None and e.get("candidate") is None for e in successful)
        unit = self.axes.get(source, "评价次数")
        if source == "native":
            config = self.run_dir / "run_config.json"
            try:
                method = json.loads(config.read_text(encoding="utf-8")).get("method", "")
            except (OSError, ValueError, AttributeError):
                method = ""
            unit = "评价次数" if method.startswith("v1013") else "候选尝试"
        self.progress = {"source": source, "x_label": unit,
                         "budget_used": max((e["evaluation"] for e in events), default=0),
                         "valid_nodes": len(node_ids) + unidentified,
                         "candidate_count": len(events), "valid_candidate_count": len(successful),
                         "valid_rate": len(successful) / len(events) if events else None}
        best = None
        points, recent = [], []
        operators, outcomes = Counter(), Counter()
        last = None
        for event in events:
            score = event["fitness"]
            operators[event["operator"]] += 1
            outcomes[str(event["status"])] += 1
            value = (-score if self.minimize else score) if score is not None else None
            recent.append({**event, "value": value})
            if score is not None and (best is None or score > best):
                point = {**event, "value": value,
                         "kind": "initial" if best is None else "breakthrough",
                         "gain": None if best is None else score - best}
                points.append(point)
                best = score
            last = event
        # Extend the incumbent through failures and regressions, without
        # misattributing its fitness/operator to the last candidate.
        if points and last["evaluation"] > points[-1]["evaluation"]:
            points.append({"evaluation": last["evaluation"], "fitness": best,
                           "value": -best if self.minimize else best, "kind": "progress"})
        self.progress["best_fitness"] = best
        return points, recent[-12:][::-1], dict(operators), dict(outcomes)
