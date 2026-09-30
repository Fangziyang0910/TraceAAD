"""Incremental, read-only training history for the shared monitor."""

from collections import Counter
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
from threading import RLock


SOURCES = ("native", "events.jsonl", "evaluations.csv", "method_events.jsonl",
           "artifacts/candidates.jsonl")
META = ("scope", "action", "reference_mode", "channel", "parent_id", "repair_of", "idea", "created_at")


NODE_ATTEMPT = re.compile(rb'"attempt_id":\s*(\d+)\s*}\s*}\s*$')


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

    CACHE_VERSION = 2
    CACHE_NAME = "monitor_cache.json"

    def __init__(self, run_dir: Path, minimize: bool):
        self.run_dir = Path(run_dir)
        self.minimize = minimize
        self.lock = RLock()
        self.identity = None
        self.offset = 0
        self.stamp = None
        self.streams = {}
        self.attempts = {}
        self.clock = {}
        self.node_offsets = {}
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
            self.attempts = payload["attempts"]
            self.clock = payload["clock"]
            self.node_offsets = payload["node_offsets"]
            self.result = payload["result"]
            self._sidecar_offset = self.offset
        except (KeyError, TypeError, ValueError):
            self.identity, self.offset, self.stamp = None, 0, None
            self.streams, self.attempts, self.clock = {}, {}, {}
            self.node_offsets = {}
            self.result = ([], [], {}, {})

    def _save_sidecar(self) -> None:
        if self.offset == self._sidecar_offset or self.identity is None:
            return
        payload = {"version": self.CACHE_VERSION, "identity": list(self.identity),
                   "offset": self.offset, "stamp": list(self.stamp or ()),
                   "streams": self.streams, "attempts": self.attempts,
                   "clock": self.clock, "node_offsets": self.node_offsets,
                   "result": self.result}
        target = self.run_dir / "logs" / self.CACHE_NAME
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)
            self._sidecar_offset = self.offset
        except OSError:
            pass  # read-only run dir: keep working, just reparse next time

    def timing_snapshot(self):
        with self.lock:
            self.read()
            return dict(self.clock)

    def read(self):
        with self.lock:
            if not self._sidecar_loaded:
                self._sidecar_loaded = True
                self._load_sidecar()
            path = next((p for p in (self.run_dir / "search.jsonl",
                        self.run_dir / "events.jsonl", self.run_dir / "logs/method_events.jsonl")
                        if p.exists()), None)
            if path is None:
                self.clock = {}
                return [], [], {}, {}
            stat = path.stat()
            identity = (str(path), stat.st_dev, stat.st_ino)
            stamp = (stat.st_size, stat.st_mtime_ns)
            if (identity != self.identity or stat.st_size < self.offset or
                    (stat.st_size == self.offset and self.stamp not in (None, stamp))):
                self.identity, self.offset = identity, 0
                self.streams, self.attempts = {}, {}
                self.clock = {}
                self.node_offsets = {}
                self.result = ([], [], {}, {})
                self.stamp = None
            if stamp == self.stamp:
                return self.result
            changed = False
            last_checkpoint = None
            with path.open("rb") as handle:
                handle.seek(self.offset)
                while handle.tell() < stat.st_size:
                    start = handle.tell()
                    raw = handle.readline(stat.st_size - handle.tell())
                    if not raw.endswith(b"\n"):
                        break
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
                            match = NODE_ATTEMPT.search(raw[-160:])
                            if match:
                                self.node_offsets[match.group(1).decode()] = start
                        self.offset = handle.tell()
                        continue
                    record = json.loads(raw)
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
            self._save_sidecar()
            return self.result

    def node(self, attempt_id):
        """Decode one indexed V10.15 node (with code) by its attempt id."""
        with self.lock:
            self.read()
            offset = self.node_offsets.get(str(attempt_id))
            if offset is None or self.identity is None:
                return None
            try:
                with open(self.identity[0], "rb") as handle:
                    handle.seek(offset)
                    record = json.loads(handle.readline())
            except (OSError, ValueError):
                return None
            data = record.get("data") if record.get("kind") == "node" else None
            return data if isinstance(data, dict) else None

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
        if source == "evaluations.csv":
            index, score = event.get("slot"), event.get("fitness")
        elif source == "artifacts/candidates.jsonl":
            index, score = event.get("order"), event.get("child_fitness")
        elif source == "method_events.jsonl":
            if event.get("event") == "epoch":
                index, score = event.get("sample_count"), event.get("best_perf")
            elif event.get("event") == "sample_registered":
                index, score = event.get("profiler_sample_order"), event.get("score")
            else:
                return False
        else:
            index = event.get("budget_used", event.get("evaluation_id"))
            score = event.get("fitness")
        position = finite(index)
        if position is None or position < 0 or not position.is_integer():
            return False
        node = event.get("node") or {}
        candidate = event.get("candidate_id", event.get("id"))
        meta = {k: event.get(k, node.get(k)) for k in META}
        if source == "native":
            meta.update(self.attempts.pop(candidate, {}))
        operator = meta.get("scope") or meta.get("action") or event.get("operator") or event.get("action") or node.get("operator") or "unknown"
        reference = meta.get("reference_mode")
        if reference and reference != "None":
            operator = f"{operator} · {reference}"
        self.streams.setdefault(source, []).append({
            "evaluation": int(position), "candidate": candidate,
            "node_id": event.get("node_id", node.get("id")),
            "fitness": finite(score), "operator": str(operator),
            "status": event.get("status") or event.get("outcome") or "unknown",
            "reason": event.get("reason") or event.get("error_type"),
            **{k: meta.get(k) for k in ("channel", "parent_id", "repair_of", "created_at")},
            "idea": str(meta.get("idea") or "")[:280],
        })
        return True

    def _build(self):
        source = next((s for s in SOURCES if any(e["fitness"] is not None
                       for e in self.streams.get(s, []))), None)
        if source is None:
            source = next((s for s in SOURCES if self.streams.get(s)), None)
        events = sorted(self.streams.get(source, []), key=lambda e: e["evaluation"])
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
        return points, recent[-12:][::-1], dict(operators), dict(outcomes)
