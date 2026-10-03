"""Read-only training monitor for experiment results."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import gzip
import hashlib
import json
from pathlib import Path
from threading import Lock
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from experiments.infra.artifacts import pick_best_sample
from experiments.infra.monitor_history import TrainingHistory
from experiments.infra.monitor_results import (
    SCALES, TEST_SCALES, batch_result_files, load_batch_heldout, load_selection, rep_of)
from experiments.infra.monitor_timing import batch_timing, search_timing
from traceaad.v10_13.storage import JOURNAL_NAME, RunStorage, read_journal


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments_result"
HTML_FILE = Path(__file__).with_name("monitor.html")

TASKS = {
    "tsp_construct": {"label": "TSP 构造", "direction": "min", "unit": "距离"},
    "cvrp_aco": {"label": "CVRP-ACO", "direction": "min", "unit": "距离"},
    "op_aco": {"label": "OP-ACO", "direction": "max", "unit": "收益"},
    "online_bin_packing": {"label": "在线装箱", "direction": "min", "unit": "箱数"},
    "vrptw_construct": {"label": "VRPTW 构造", "direction": "min", "unit": "距离"},
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _stamp(path: Path):
    try:
        stat = path.stat()
        return (stat.st_size, stat.st_mtime_ns)
    except OSError:
        return None


def _compact_curve(points, cap: int = 200):
    """Thin a run-best curve for list payloads, keeping endpoints and records."""
    if not isinstance(points, list) or len(points) <= cap:
        return points
    keep: set[int] = {0, len(points) - 1}
    for index, point in enumerate(points):
        if isinstance(point, dict) and point.get("kind") in {"initial", "breakthrough"}:
            keep.add(index)
    step = len(points) / cap
    cursor = 0.0
    while cursor < len(points):
        keep.add(round(cursor))
        cursor += step
    return [points[index] for index in sorted(keep)]


LIST_POINT_FIELDS = ("evaluation", "fitness", "value", "kind", "gain", "candidate", "operator")


def _list_curve(points):
    """Overview curves carry only what the sparkline and its tooltip need."""
    return [{key: point.get(key) for key in LIST_POINT_FIELDS if point.get(key) is not None}
            for point in (_compact_curve(points) or []) if isinstance(point, dict)]


def _json_bytes(payload: Any):
    return None if payload is None else json.dumps(
        payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _last_candidate(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            handle.seek(max(0, end - 131_072))
            lines = handle.read().decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("kind") == "candidate":
            return value
    if path.exists():
        last = {}
        for value in read_journal(path):
            if value.get("kind") == "candidate":
                last = value
        return last
    return {}


def _run_node_stats(run_dir: Path) -> tuple[int, float | None]:
    path = run_dir / JOURNAL_NAME
    count, best = 0, None
    for item in read_journal(path):
        node = item.get("node") if item["kind"] == "candidate" else None
        if node is not None:
            count += 1
            fitness = node["fitness"]
            best = fitness if best is None else max(best, fitness)
    return count, best


def _objective(fitness: float | None, task: str) -> float | None:
    if fitness is None:
        return None
    return -fitness if TASKS[task]["direction"] == "min" else fitness


def _search_events(run_dir: Path):
    journal = run_dir / JOURNAL_NAME
    if journal.exists():
        with journal.open(encoding="utf-8") as handle:
            for line in handle:
                # Request/call/state lines carry prompts and RNG blobs and are
                # irrelevant here; skip them before the JSON decode.
                if line.startswith(('{"kind":"request"', '{"kind":"call"', '{"kind":"state"')):
                    continue
                record = json.loads(line)
                if "source" in record:
                    source = record["source"]
                    if source in {"events.jsonl", "evaluations.csv", "artifacts/candidates.jsonl"}:
                        yield source, record.get("data")
                elif record.get("kind") == "candidate":
                    yield "native", record
    else:
        for path in (run_dir / "events.jsonl", run_dir / "logs/method_events.jsonl"):
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        yield path.name, json.loads(line)
                break


@lru_cache(maxsize=128)
def _history(run_dir: Path, task: str):
    return TrainingHistory(run_dir, minimize=TASKS[task]["direction"] == "min")


def _search_trend(run_dir: Path, task: str):
    return _history(Path(run_dir), task).read()


class V1013Monitor:
    def __init__(self, results_root: Path = DEFAULT_RESULTS_ROOT / "traceaad_v10_13", batch: str | None = None):
        self.results_root = results_root
        self.default_batch = batch

    def batches(self) -> list[dict[str, Any]]:
        batches = []
        for path in self.results_root.glob("batch_*.json"):
            manifest = _read_json(path)
            if not manifest.get("plan") or not str(manifest.get("method", "")).startswith("v1013"):
                continue
            batches.append({
                "id": manifest.get("batch", path.stem.removeprefix("batch_")),
                "created_at": manifest.get("created_at"),
                "updated_at": manifest.get("updated_at"),
            })
        batches.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        if self.default_batch:
            return [item for item in batches if item["id"] == self.default_batch]
        return batches[:1]

    def _manifest(self, batch: str | None) -> dict[str, Any]:
        chosen = batch or self.default_batch
        if chosen:
            manifest = _read_json(self.results_root / f"batch_{chosen}.json")
            if manifest.get("plan") and str(manifest.get("method", "")).startswith("v1013"):
                return manifest
        available = self.batches()
        return _read_json(self.results_root / f"batch_{available[0]['id']}.json") if available else {}

    def _run_dir(self, row: dict[str, Any]) -> Path:
        return self.results_root / row["task"] / row["run_name"]

    def _run_summary(self, row: dict[str, Any], now: float | None = None) -> dict[str, Any]:
        task = row["task"]
        run_dir = self._run_dir(row)
        config = _read_json(run_dir / "run_config.json")
        final = _read_json(run_dir / "logs/run_summary.json")
        last_event = _last_candidate(run_dir / JOURNAL_NAME)
        params = config.get("method_params") or {}

        status = str(row.get("status") or "queued")
        active = status in {"running", "launching"}
        final_status = final.get("status")
        if final_status == "finished":
            status = "finished"
            active = False
        elif final_status in {"error", "interrupted"} and not active:
            status = "blocked"
        elif status == "launching":
            status = "running"

        budget = int(final.get("budget") or params.get("budget") or 1000)
        budget_used = int(
            (last_event.get("budget_used") if active else final.get("budget_used"))
            or last_event.get("budget_used")
            or 0
        )
        valid_nodes = None if active else final.get("num_nodes")
        best = final.get("best") or {}
        best_fitness = last_event.get("best_fitness") if active else best.get("fitness")
        if best_fitness is None:
            best_fitness = last_event.get("best_fitness")
        if valid_nodes is None or best_fitness is None:
            journal_count, journal_best = _run_node_stats(run_dir)
            valid_nodes = journal_count if valid_nodes is None else valid_nodes
            best_fitness = journal_best if best_fitness is None else best_fitness
        valid_nodes = int(valid_nodes or 0)
        if best_fitness is not None:
            best_fitness = float(best_fitness)

        result = {
            "task": task,
            "name": row["run_name"],
            "repeat": int(row.get("repeat") or config.get("repeat") or 0),
            "seed": row.get("seed", config.get("seed")),
            "backend": row.get("backend", config.get("backend")),
            "status": status,
            "budget": budget,
            "budget_used": budget_used,
            "valid_nodes": valid_nodes,
            "best_fitness": best_fitness,
            "best_value": _objective(best_fitness, task),
            "updated_at": last_event.get("ts") or final.get("finished_at") or row.get("started_at"),
            "error": None if active else final.get("error") or row.get("last_error"),
            "curve": _search_trend(run_dir, task)[0],
            "x_label": "评价次数",
        }
        timing_summary = {**final, "started_at": final.get("started_at") or row.get("started_at")}
        result["timing"] = search_timing(result, timing_summary,
                                         _history(run_dir, task).timing_snapshot(), unit="评价", now=now)
        return result

    def overview(self, batch: str | None = None) -> dict[str, Any]:
        manifest = self._manifest(batch)
        if not manifest:
            return {"batch": None, "summary": {}, "tasks": []}

        now = time.time()
        runs = [self._run_summary(row, now) for row in manifest["plan"]]
        status_counts = Counter(run["status"] for run in runs)
        task_groups = []
        for key, meta in TASKS.items():
            task_runs = sorted(
                (run for run in runs if run["task"] == key),
                key=lambda run: run["repeat"],
            )
            if task_runs:
                task_groups.append({"key": key, **meta, "runs": task_runs})

        return {
            "batch": manifest.get("batch"),
            "updated_at": manifest.get("updated_at"),
            "summary": {
                "runs": len(runs),
                "finished": status_counts["finished"],
                "running": status_counts["running"],
                "queued": status_counts["queued"],
                "blocked": status_counts["blocked"],
                "budget_used": sum(run["budget_used"] for run in runs),
                "budget": sum(run["budget"] for run in runs),
                "valid_nodes": sum(run["valid_nodes"] for run in runs),
                "timing": batch_timing(runs),
            },
            "tasks": task_groups,
        }

    def run_detail(self, batch: str, task: str, name: str) -> dict[str, Any] | None:
        manifest = self._manifest(batch)
        row = next(
            (item for item in manifest.get("plan", [])
             if item.get("task") == task and item.get("run_name") == name),
            None,
        )
        if row is None or task not in TASKS:
            return None

        summary = self._run_summary(row)
        run_dir = self._run_dir(row)
        storage = RunStorage(run_dir)
        nodes = storage.records("nodes")
        curve, recent, operators, outcomes = _search_trend(run_dir, task)

        valid_nodes = [node for node in nodes if node.get("fitness") is not None]
        best_node = max(valid_nodes, key=lambda node: float(node["fitness"]), default=None)
        if best_node is None:
            final_best = _read_json(run_dir / "logs/run_summary.json").get("best")
            best_node = final_best if isinstance(final_best, dict) else None

        return {
            **summary,
            "task_meta": TASKS[task],
            "curve": curve,
            "operators": operators,
            "outcomes": outcomes,
            "recent": recent,
            "best": None if best_node is None else {
                "id": best_node.get("id", best_node.get("node_id")),
                "fitness": float(best_node["fitness"]),
                "value": _objective(float(best_node["fitness"]), task),
                "operator": best_node.get("operator"),
                "idea": best_node.get("idea") or "",
                "code": best_node.get("code") or "",
            },
        }


class ResultsMonitor:
    """List experiment directories; reuse the detailed V10.13 reader where applicable."""

    def __init__(self, results_root: Path = DEFAULT_RESULTS_ROOT,
                 experiment: str | None = None, batch: str | None = None):
        self.results_root = Path(results_root)
        self.default_experiment = experiment
        self.v1013 = V1013Monitor(self.results_root / "traceaad_v10_13", batch)
        self._progress_cache = {}

    def _recorded_progress(self, run_dir: Path) -> tuple[int, int]:
        journal = run_dir / JOURNAL_NAME
        if not journal.exists():
            return 0, 0
        stamp = (journal.stat().st_size, journal.stat().st_mtime_ns)
        cached = self._progress_cache.get(run_dir)
        if cached and cached[0] == stamp:
            return cached[1]
        streams = {}
        for source, event in _search_events(run_dir):
            if not isinstance(event, dict):
                continue
            if source == "artifacts/candidates.jsonl":
                counted = bool(event.get("evaluator_called"))
                valid = isinstance(event.get("child_fitness"), (int, float))
                position = None
            elif source in {"events.jsonl", "native"}:
                counted = event.get("budget_used") is not None
                valid = isinstance(event.get("fitness"), (int, float))
                position = event.get("budget_used")
            else:
                continue
            record = streams.setdefault(source, [0, 0])
            if counted:
                record[0] = max(record[0], int(position)) if position is not None else record[0] + 1
            record[1] += valid
        result = tuple(streams.get("native", streams.get("events.jsonl", streams.get("artifacts/candidates.jsonl", [0, 0]))))
        self._progress_cache[run_dir] = stamp, result
        return result

    def _journal_is_hot(self, run_dir: Path, *, max_age_sec: float = 1200.0) -> bool:
        """A run whose journal was appended recently is live even without a summary.

        Covers methods that only write ``run_summary.json`` at completion; the
        margin exceeds the LLM request timeout plus service-error retries.
        """
        journal = run_dir / JOURNAL_NAME
        try:
            return (time.time() - journal.stat().st_mtime) < max_age_sec
        except OSError:
            return False

    def _journal_newer_than_summary(self, run_dir: Path) -> bool:
        """True when the journal kept growing after the summary was written."""
        journal, summary = run_dir / JOURNAL_NAME, run_dir / "logs" / "run_summary.json"
        try:
            return journal.stat().st_mtime_ns > summary.stat().st_mtime_ns
        except OSError:
            return False

    def batches(self) -> list[dict[str, str]]:
        entries = []
        for directory in self.results_root.iterdir():
            if directory.is_dir() and any(directory.glob("*/*/run_config.json")):
                entries.append((self._latest_activity(directory), directory.name))
        entries.sort(reverse=True)
        return [{"id": name, "label": name} for _, name in entries]

    @staticmethod
    def _latest_activity(directory: Path) -> float:
        latest = 0.0
        for journal in directory.glob("*/*/search.jsonl"):
            try:
                latest = max(latest, journal.stat().st_mtime)
            except OSError:
                pass
        return latest

    def state_signature(self, experiment: str | None):
        """Cheap change-detection stamp for a batch: run journal/summary stats."""
        experiment = experiment or next((item["id"] for item in self.batches()), None)
        stamps = []
        if experiment:
            for config_path in sorted((self.results_root / experiment).glob("*/*/run_config.json")):
                run_dir = config_path.parent
                stamps.append((run_dir.name,
                               _stamp(run_dir / JOURNAL_NAME),
                               _stamp(run_dir / "logs" / "run_summary.json")))
        return (experiment, tuple(stamps))

    def run_signature(self, experiment: str, task: str, name: str):
        run_dir = self.results_root / experiment / task / name
        return (experiment, task, name, _stamp(run_dir / JOURNAL_NAME),
                _stamp(run_dir / "logs" / "run_summary.json"))

    def _runs(self, experiment: str):
        if experiment not in {item["id"] for item in self.batches()}:
            return []
        rows = []
        now = time.time()
        for config_path in sorted((self.results_root / experiment).glob("*/*/run_config.json")):
            run_dir = config_path.parent
            config = _read_json(config_path)
            summary = _read_json(run_dir / "logs/run_summary.json")
            raw_status = summary.get("status")
            journal_hot = self._journal_is_hot(run_dir)
            if raw_status == "finished":
                status = "finished"
            elif raw_status == "running":
                status = "running"
            elif raw_status == "unknown":
                status = "unknown"
            elif journal_hot and (
                not raw_status
                or (raw_status == "service_unavailable" and self._journal_newer_than_summary(run_dir))
            ):
                # A hot journal with no summary yet (summary written at
                # completion), or one that kept growing past a service-pause
                # summary: the run is live.
                status = "running"
            else:
                status = "blocked" if raw_status else "queued"
            params = config.get("method_params") or {}
            best = summary.get("best") or {}
            if not isinstance(best, dict):
                best = {}
            score = best.get("fitness", summary.get("best_score"))
            selection = best.get("selection_fitness")
            budget = summary.get("budget", summary.get("budget_slots", params.get("budget", params.get("max_sample_nums", 0))))
            used = summary.get("budget_used", summary.get("budget_slots", summary.get("num_samples", summary.get("evaluator_call_count", 0))))
            nodes = summary.get("num_nodes", summary.get("n_algorithms", summary.get("evaluate_success_program_num", 0)))
            if raw_status == "unknown" or (status == "running" and journal_hot):
                used, nodes = self._recorded_progress(run_dir)
            task = config.get("task", run_dir.parent.name)
            if task not in TASKS:
                continue
            native_v1014 = config.get("method") == "v1014" or experiment == "traceaad_v10_14"
            row = {
                "task": task, "name": run_dir.name, "repeat": config.get("repeat"),
                "seed": config.get("seed"), "backend": config.get("backend"),
                "status": status, "raw_status": raw_status, "budget": int(budget or 0),
                "budget_used": int(used or 0), "valid_nodes": int(nodes or 0),
                "best_fitness": score, "best_value": _objective(score, task),
                "selection_fitness": selection, "selection_value": _objective(selection, task),
                "updated_at": summary.get("finished_at"), "run_dir": run_dir,
                "selection_info": load_selection(run_dir),
                "curve": _list_curve(_search_trend(run_dir, task)[0]),
                "x_label": "候选尝试" if native_v1014 else "已记录序号",
            }
            row["timing"] = search_timing(row, summary, _history(run_dir, task).timing_snapshot(),
                                           unit="候选" if native_v1014 else "预算单位", now=now)
            rows.append(row)
        return rows

    def overview(self, experiment: str | None = None):
        experiment = experiment or (self.batches()[0]["id"] if self.batches() else None)
        if experiment == "traceaad_v10_13":
            result = self.v1013.overview()
            return {**result, "batch": experiment}
        runs = self._runs(experiment) if experiment else []
        counts = Counter(row["status"] for row in runs)
        groups = []
        for task, meta in TASKS.items():
            task_runs = [row for row in runs if row["task"] == task]
            if task_runs:
                groups.append({"key": task, **meta, "runs": [
                    {key: value for key, value in row.items() if key != "run_dir"}
                    for row in task_runs
                ]})
        return {
            "batch": experiment, "updated_at": max((row["updated_at"] or "" for row in runs), default=""),
            "summary": {
                "runs": len(runs), "finished": counts["finished"],
                "running": counts["running"], "queued": counts["queued"],
                "blocked": counts["blocked"], "unknown": counts["unknown"],
                "budget_used": sum(row["budget_used"] for row in runs),
                "budget": sum(row["budget"] for row in runs),
                "valid_nodes": sum(row["valid_nodes"] for row in runs),
                "timing": batch_timing(runs),
            },
            "tasks": groups,
        }

    # ---------- cross-batch comparison ----------

    def cohorts(self):
        """Comparable result sets: one per batch, or one per held-out variant."""
        items = []
        for batch in self.batches():
            variants = load_batch_heldout(self.results_root / batch["id"])
            names = sorted(variants) or [""]
            for variant in names:
                tasks = sorted(variants.get(variant, {}))
                items.append({
                    "id": f"{batch['id']}::{variant}" if variant else batch["id"],
                    "batch": batch["id"], "variant": variant,
                    "label": f"{batch['id']} · {variant}" if variant and len(names) > 1 else batch["id"],
                    "heldout_tasks": tasks,
                })
        return items

    def compare_signature(self, cohort_ids):
        batches = sorted({cohort.split("::")[0] for cohort in cohort_ids})
        return tuple((batch, self.state_signature(batch),
                      tuple((str(path), _stamp(path)) for path in batch_result_files(self.results_root / batch)))
                     for batch in batches)

    def _task_rows(self, batch: str):
        if batch == "traceaad_v10_13":
            return [row for group in self.v1013.overview().get("tasks", []) for row in group["runs"]]
        return self._runs(batch)

    def compare(self, cohort_ids):
        known = {item["id"]: item for item in self.cohorts()}
        chosen = [known[cohort] for cohort in cohort_ids if cohort in known]
        rows_by_batch = {item["batch"]: self._task_rows(item["batch"]) for item in chosen}
        heldout_by_batch = {item["batch"]: load_batch_heldout(self.results_root / item["batch"]) for item in chosen}
        tasks = []
        for task, meta in TASKS.items():
            entries = []
            for item in chosen:
                heldout = heldout_by_batch[item["batch"]].get(item["variant"], {}).get(task, {})
                named = heldout.get("runs", {})
                rows = [row for row in rows_by_batch[item["batch"]] if row["task"] == task]
                if named:
                    rows = [row for row in rows if row["name"] in named]
                elif item["variant"]:
                    rows = [row for row in rows if item["variant"] in row["name"]]
                runs = []
                for row in rows:
                    selection = row.get("selection_info") or {}
                    runs.append({
                        "name": row["name"], "rep": row.get("repeat") or rep_of(row["name"]),
                        "status": row.get("status"),
                        # the final program's training score (selected node for V10.14+)
                        "train": row.get("best_fitness") if row.get("status") == "finished"
                        else (row.get("curve") or [{}])[-1].get("fitness", row.get("best_fitness")),
                        "selection": selection.get("fitness", row.get("selection_fitness")),
                        "ties": selection.get("ties"), "finalists": selection.get("finalists"),
                        "heldout": {str(k): v for k, v in named.get(row["name"], {}).items()},
                    })
                present = {run["name"] for run in runs}
                for name, scores in named.items():  # held-out results whose run dir is gone
                    if name not in present:
                        runs.append({"name": name, "rep": rep_of(name), "status": None, "train": None,
                                     "selection": None, "heldout": {str(k): v for k, v in scores.items()}})
                if runs:
                    runs.sort(key=lambda run: (run["rep"] is None, run["rep"] or 0, run["name"]))
                    entries.append({"id": item["id"], "source": heldout.get("source"), "runs": runs})
            if entries:
                tasks.append({"key": task, **meta, "cohorts": entries, "scales": [
                    {"key": str(scale), "label": str(scale), "test": scale in TEST_SCALES[task]}
                    for scale in SCALES[task]]})
        return {"cohorts": [{k: item[k] for k in ("id", "label", "batch", "variant")} for item in chosen],
                "tasks": tasks}

    def run_detail(self, experiment: str, task: str, name: str):
        if experiment == "traceaad_v10_13":
            return self.v1013.run_detail(self.v1013.default_batch or "", task, name)
        row = next((row for row in self._runs(experiment)
                    if row["task"] == task and row["name"] == name), None)
        if row is None:
            return None
        run_dir = row["run_dir"]
        summary = _read_json(run_dir / "logs/run_summary.json")
        best = summary.get("best")
        curve, recent, operators, outcomes = _search_trend(run_dir, task)
        if not isinstance(best, dict) or not isinstance(best.get("code"), str):
            # Live V10.15 runs: the incumbent's node line is indexed by the
            # incremental history, so fetch just that line.
            record = next((point for point in reversed(curve)
                           if point.get("kind") in {"initial", "breakthrough"}), None)
            node = _history(run_dir, task).node(record["candidate"]) if record else None
            if node and isinstance(node.get("code"), str):
                best = node
        if not isinstance(best, dict) or not isinstance(best.get("code"), str):
            try:
                sample, _ = pick_best_sample(run_dir, allow_incomplete=True)
                best = {"code": sample["program"], "fitness": sample["score"],
                        "operator": sample.get("operator")}
            except (RuntimeError, OSError, ValueError, KeyError):
                best = None
        if best is None:
            journal = run_dir / JOURNAL_NAME
            if journal.exists():
                with journal.open(encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        node = record.get("data") if record.get("kind") == "node" else None
                        if (isinstance(node, dict) and isinstance(node.get("code"), str)
                                and isinstance(node.get("fitness"), (int, float))
                                and (best is None or node["fitness"] > best["fitness"])):
                            best = node
        return {
            **{key: value for key, value in row.items() if key != "run_dir"},
            "task_meta": TASKS[task], "curve": curve, "operators": operators,
            "outcomes": outcomes, "recent": recent,
            "best": None if best is None else {
                "id": best.get("id", best.get("node_id")),
                "fitness": best.get("fitness"),
                "value": _objective(best.get("fitness"), task),
                "operator": best.get("operator") or best.get("action"), "idea": best.get("idea") or "",
                "code": best.get("code") or "",
            },
        }


class ResponseCache:
    """Serialized-JSON cache keyed by request, validated by a cheap signature.

    Concurrent requests for the same key share one build instead of each
    re-parsing the same journals.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._store: dict[Any, tuple[Any, bytes]] = {}
        self._key_locks: dict[Any, Lock] = {}

    def get_or_build(self, key, signature, builder):
        with self._lock:
            key_lock = self._key_locks.setdefault(key, Lock())
        with key_lock:
            with self._lock:
                hit = self._store.get(key)
                if hit is not None and hit[0] == signature:
                    return hit[1], True
            body = builder()
            with self._lock:
                self._store[key] = (signature, body)
            return body, False


def make_request_handler(monitor: ResultsMonitor) -> type[BaseHTTPRequestHandler]:
    cache = ResponseCache()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if parsed.path in {"/", "/index.html"}:
                return self._serve_html()
            if parsed.path == "/api/batches":
                return self._send_json({"batches": monitor.batches()})
            if parsed.path == "/api/state":
                batch = params.get("batch", [None])[0]
                signature = monitor.state_signature(batch)
                body, _ = cache.get_or_build(
                    ("state", batch), signature,
                    lambda: _json_bytes(monitor.overview(batch)))
                return self._send_body(body, signature)
            if parsed.path == "/api/cohorts":
                return self._send_json({"cohorts": monitor.cohorts()})
            if parsed.path == "/api/compare":
                cohorts = [c for c in params.get("cohorts", [""])[0].split(",") if c][:12]
                signature = monitor.compare_signature(cohorts)
                body, _ = cache.get_or_build(
                    ("compare", tuple(cohorts)), signature,
                    lambda: _json_bytes(monitor.compare(cohorts)))
                return self._send_body(body, signature)
            if parsed.path == "/api/run":
                batch = params.get("batch", [""])[0]
                task = params.get("task", [""])[0]
                name = params.get("name", [""])[0]
                signature = monitor.run_signature(batch, task, name)
                body, _ = cache.get_or_build(
                    ("run", batch, task, name), signature,
                    lambda: _json_bytes(monitor.run_detail(batch, task, name)))
                if body is None:
                    return self.send_error(404, "Run not found")
                return self._send_body(body, signature)
            self.send_error(404, "Endpoint not found")

        def _serve_html(self) -> None:
            try:
                body = HTML_FILE.read_bytes()
            except OSError:
                return self.send_error(404, "Page not found")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any) -> None:
            self._send_body(_json_bytes(payload), None)

        def _send_body(self, body: bytes, signature) -> None:
            etag = None
            if signature is not None:
                etag = '"' + hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()[:20] + '"'
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            headers = {"Content-Type": "application/json; charset=utf-8",
                       # Stored but always revalidated: conditional polls get 304.
                       "Cache-Control": "private, max-age=0" if etag else "no-store"}
            if etag:
                headers["ETag"] = etag
            payload = body
            if len(body) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                payload = gzip.compress(body, 5)
                headers["Content-Encoding"] = "gzip"
            self.send_response(200)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment results monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--experiment", default="traceaad_v10_13")
    parser.add_argument("--batch", "--version", dest="batch", default=None)
    args = parser.parse_args()

    monitor = ResultsMonitor(args.results_dir, args.experiment, args.batch)
    server = ThreadingHTTPServer((args.host, args.port), make_request_handler(monitor))
    selected = monitor.overview().get("batch") or "无实验"
    print(f"Experiment monitor: http://127.0.0.1:{args.port} ({selected})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
