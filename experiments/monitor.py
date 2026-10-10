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

from experiments.infra.monitor_history import TrainingHistory, finite
from experiments.infra.monitor_results import (
    batch_result_files, load_batch_heldout, rep_of)
from benchmarks.tasks import SCALES, TEST_SCALES
from experiments.infra.monitor_timing import batch_timing, search_timing
from traceaad.common.storage import JOURNAL_NAME, live_snapshot, normalize_live_record, selected_program


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments_result"
HTML_FILE = Path(__file__).with_name("monitor.html")

TASKS = {
    "tsp_construct": {"label": "TSP 构造", "unit": "距离"},
    "cvrp_aco": {"label": "CVRP-ACO", "unit": "距离"},
    "op_aco": {"label": "OP-ACO", "unit": "负收益"},
    "online_bin_packing": {"label": "在线装箱", "unit": "箱数"},
    "vrptw_construct": {"label": "VRPTW 构造", "unit": "距离"},
    "fssp_gls": {"label": "FSSP-GLS", "unit": "参考偏差 %"},
    "graph_colouring": {"label": "图着色", "unit": "参考偏差 %"},
    "jssp_construct": {"label": "作业车间调度", "unit": "参考偏差 %"},
}


def _run_config_paths(directory: Path):
    """List canonical runs, excluding temporary aliases used by live writers."""
    return sorted(path for path in directory.glob("*/*/run_config.json")
                  if not path.parent.is_symlink() and not path.parent.parent.is_symlink())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    live = live_snapshot(path.parent) if path.name == "summary.json" else None
    value = normalize_live_record(value, live["task"]) if live else value
    return value if isinstance(value, dict) else {}


def _stamp(path: Path):
    try:
        stat = path.stat()
        return (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
    except OSError:
        return None


def _history_stamp(run_dir: Path):
    return tuple(_stamp(run_dir / name) for name in ("events.jsonl", "programs.jsonl"))


def _run_stamp(run_dir: Path):
    return (_history_stamp(run_dir), tuple(_stamp(run_dir / name) for name in (
        "run_config.json", "resume.json", "summary.json", "selection.json", "best_program.py")))


def _candidate_summary(runs):
    count = sum(run.get("candidate_count") or 0 for run in runs)
    valid = sum(run.get("valid_candidate_count") or 0 for run in runs)
    return {"candidate_count": count, "valid_candidate_count": valid,
            "valid_rate": valid / count if count else None}


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


LIST_POINT_FIELDS = ("evaluation", "fitness", "kind", "gain", "candidate", "operator")


def _list_curve(points):
    """Overview curves carry only what the sparkline and its tooltip need."""
    return [{key: point.get(key) for key in LIST_POINT_FIELDS if point.get(key) is not None}
            for point in (_compact_curve(points) or []) if isinstance(point, dict)]


def _json_bytes(payload: Any):
    return None if payload is None else json.dumps(
        payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


@lru_cache(maxsize=128)
def _history(run_dir: Path, task: str):
    return TrainingHistory(run_dir, minimize=True)


def _search_trend(run_dir: Path, task: str):
    return _history(Path(run_dir), task).read()


def _program_view(program, task):
    if not isinstance(program, dict):
        return None
    score = finite(program.get("fitness"))
    return {"id": program.get("id"), "fitness": score,
            "operator": program.get("action"),
            "idea": program.get("idea") or "", "code": program.get("code") or ""}


def _programs(run_dir, task, summary, curve):
    history = _history(run_dir, task)
    point = next((p for p in reversed(curve) if p["kind"] in {"initial", "breakthrough"}), None)
    search = history.node(point["node_id"], by_node=True) if point and point.get("node_id") is not None else None
    if search and point:
        search = {**search, "fitness": point["fitness"]}
    selected = selected_program(run_dir)
    search, selected = _program_view(search, task), _program_view(selected, task)
    breakthroughs = []
    for p in curve:
        if p["kind"] not in {"initial", "breakthrough"} or p.get("node_id") is None:
            continue
        view = _program_view(history.node(p["node_id"], by_node=True), task) or {"id": p.get("candidate")}
        breakthroughs.append({**view, "fitness": p["fitness"],
                              "evaluation": p["evaluation"], "node_id": p["node_id"], "operator": p.get("operator"),
                              "parent_id": p.get("parent_id"), "gain": p.get("gain"),
                              "idea": view.get("idea") or p.get("idea") or ""})
    return {"search_best": search, "selected_best": selected, "best": selected or search,
            "breakthroughs": breakthroughs}


def _merged_heldout(batch_dir):
    """Held-out variants of one batch, with backfills merged into the cohort they complete.

    A backfill tests runs of the batch that were left untested, so it joins the
    cohort whose runs it does not overlap. Other variants are separate
    conditions or arms and stay separate.
    """
    def cells(tasks):
        return {(task, name) for task, data in tasks.items() for name in data["runs"]}

    variants = load_batch_heldout(batch_dir)
    groups = []
    for name in sorted(variants, key=lambda v: (v.startswith("backfill"), v)):
        tasks = variants[name]
        target = (next((g for g in groups if not cells(g["tasks"]) & cells(tasks)), None)
                  if name.startswith("backfill") else None)
        if target is None:
            groups.append({"names": [name], "tasks": {
                task: {"source": data["source"], "runs": dict(data["runs"]),
                       "verification": dict(data["verification"])} for task, data in tasks.items()}})
            continue
        target["names"].append(name)
        for task, data in tasks.items():
            merged = target["tasks"].setdefault(task, {"source": data["source"], "runs": {}, "verification": {}})
            merged["runs"].update(data["runs"])
            merged["verification"].update(data["verification"])
    return {"" if "" in g["names"] else "+".join(g["names"]): g["tasks"] for g in groups}


class ResultsMonitor:
    """List experiment directories through one historical-data reader."""

    def __init__(self, results_root: Path = DEFAULT_RESULTS_ROOT,
                 experiment: str | None = None, batch: str | None = None):
        self.results_root = Path(results_root)
        self.default_experiment = experiment
        self.batch_filter = batch

    def _journal_is_hot(self, run_dir: Path, *, max_age_sec: float = 1200.0) -> bool:
        """A run whose journal was appended recently is live even without a summary.

        Covers methods that only write ``summary.json`` at completion; the
        margin exceeds the LLM request timeout plus service-error retries.
        """
        journal = run_dir / JOURNAL_NAME
        try:
            return (time.time() - journal.stat().st_mtime) < max_age_sec
        except OSError:
            return False

    def _journal_newer_than_summary(self, run_dir: Path) -> bool:
        """True when the journal kept growing after the summary was written."""
        journal, summary = run_dir / JOURNAL_NAME, run_dir / "summary.json"
        try:
            return journal.stat().st_mtime_ns > summary.stat().st_mtime_ns
        except OSError:
            return False

    def batches(self) -> list[dict[str, str]]:
        entries = []
        if not self.results_root.is_dir():
            return entries
        for directory in self.results_root.iterdir():
            if directory.is_dir() and not directory.is_symlink() and _run_config_paths(directory):
                entries.append((self._latest_activity(directory), directory.name))
        entries.sort(reverse=True)
        return [{"id": name, "label": name} for _, name in entries]

    @staticmethod
    def _latest_activity(directory: Path) -> float:
        latest = 0.0
        for config in _run_config_paths(directory):
            journal = config.parent / "events.jsonl"
            try:
                latest = max(latest, journal.stat().st_mtime)
            except OSError:
                pass
        return latest

    def default_batch(self):
        batches = self.batches()
        if self.default_experiment in {item["id"] for item in batches}:
            return self.default_experiment
        return batches[0]["id"] if batches else None

    def state_signature(self, experiment: str | None):
        """File dependencies plus a 15-second clock for unfinished searches."""
        experiment = experiment or self.default_batch()
        directory = self.results_root / experiment if experiment else self.results_root
        stamps, unfinished = [], False
        for path in _run_config_paths(directory) if experiment else []:
            run_dir = path.parent
            stamps.append((str(run_dir.relative_to(directory)), _run_stamp(run_dir)))
            unfinished |= _read_json(run_dir / "summary.json").get("status") != "finished"
        manifests = tuple((p.name, _stamp(p)) for p in sorted(directory.glob("batch_*.json")))
        return experiment, tuple(stamps), manifests, int(time.time() // 15) if unfinished else None

    def run_signature(self, experiment: str, task: str, name: str):
        run_dir = self.results_root / experiment / task / name
        unfinished = _read_json(run_dir / "summary.json").get("status") != "finished"
        manifests = tuple((p.name, _stamp(p)) for p in sorted((self.results_root / experiment).glob("batch_*.json")))
        return experiment, task, name, _run_stamp(run_dir), manifests, int(time.time() // 15) if unfinished else None

    def _manifest_runs(self, experiment):
        rows = {}
        for path in sorted((self.results_root / experiment).glob("batch_*.json")):
            manifest = _read_json(path)
            if self.batch_filter and manifest.get("batch") != self.batch_filter:
                continue
            for row in manifest.get("plan", []):
                name = row.get("run_name") or Path(row.get("run_dir", "")).name
                rows[(row.get("task"), name)] = {**row, 'run_batch': manifest.get('batch'),
                    'batch_status': manifest.get('status')}
        return rows

    def _runs(self, experiment: str, task_filter=None, name_filter=None, run_batch=None):
        if experiment not in {item["id"] for item in self.batches()}:
            return []
        rows = []
        manifest_runs = self._manifest_runs(experiment)
        now = time.time()
        for config_path in _run_config_paths(self.results_root / experiment):
            run_dir = config_path.parent
            if ((task_filter and run_dir.parent.name != task_filter)
                    or (name_filter and run_dir.name != name_filter)):
                continue
            metadata = manifest_runs.get((run_dir.parent.name, run_dir.name), {})
            if run_batch and metadata.get('run_batch') != run_batch:
                continue
            if self.batch_filter and not metadata:
                continue
            config = _read_json(config_path)
            summary = _read_json(run_dir / "summary.json")
            raw_status = summary.get("status")
            journal_hot = self._journal_is_hot(run_dir)
            if raw_status == "finished":
                status = "finished"
            elif metadata.get('batch_status') == 'stopped_by_user' or metadata.get('status') == 'stopped_by_user':
                status = 'stopped'
            elif raw_status == "stopped" and not self._journal_newer_than_summary(run_dir):
                status = "stopped"
            elif metadata.get("status") in {"running", "launching"} or raw_status == "running":
                status = "running"
            elif raw_status == "unknown":
                status = "unknown"
            elif journal_hot and (
                not raw_status
                or (raw_status in {"service_unavailable", "stopped"} and self._journal_newer_than_summary(run_dir))
            ):
                # A hot journal with no summary yet (summary written at
                # completion), or one that kept growing past a service-pause
                # summary: the run is live.
                status = "running"
            else:
                status = "blocked" if raw_status else "queued"
            best = summary.get("best") or {}
            if not isinstance(best, dict):
                best = {}
            score = finite(best.get("fitness"))
            budget = summary.get("budget", config.get("budget", 0))
            used = summary.get("budget_used", 0)
            nodes = summary.get("num_nodes", 0)
            task = config.get("task", run_dir.parent.name)
            if task not in TASKS:
                continue
            history = _history(run_dir, task)
            curve = history.read()[0]
            progress = history.progress_snapshot()
            if progress.get("candidate_count"):
                nodes = progress["valid_nodes"]
                if status != "finished" or not used:
                    used = progress["budget_used"]
                if status != "finished" or score is None:
                    score = progress["best_fitness"]
            row = {
                "task": task, "name": run_dir.name, "repeat": config.get("repeat", metadata.get("repeat")),
                "run_batch": metadata.get('run_batch'),
                "seed": config.get("seed", metadata.get("seed")), "backend": config.get("backend", metadata.get("backend")),
                "status": status, "raw_status": raw_status, "budget": int(budget or 0),
                "budget_used": int(used or 0), "valid_nodes": int(nodes or 0),
                "best_fitness": score,
                "updated_at": summary.get("finished_at") or history.clock.get("completed_at") or metadata.get("started_at"), "run_dir": run_dir,
                "error": None if status == "running" else summary.get("error") or metadata.get("last_error"),
                "curve": _list_curve(curve),
                **{k: progress.get(k, 0) for k in ("candidate_count", "valid_candidate_count")},
                "valid_rate": progress.get("valid_rate"),
                "x_label": progress.get("x_label", "已记录序号"),
            }
            timing_summary = {**summary, "started_at": summary.get("started_at") or metadata.get("started_at")}
            row["timing"] = search_timing(row, timing_summary, _history(run_dir, task).timing_snapshot(),
                                           unit={"候选尝试": "候选", "样本次数": "样本", "预算槽位": "预算槽",
                                                 "候选序号": "候选"}.get(row["x_label"], "评价"), now=now)
            rows.append(row)
        return rows

    def overview(self, experiment: str | None = None, run_batch=None):
        experiment = experiment or self.default_batch()
        run_batches = sorted({r['run_batch'] for r in self._manifest_runs(experiment).values()
                              if r.get('run_batch')}, reverse=True) if experiment else []
        if run_batch == 'latest':
            run_batch = run_batches[0] if run_batches else None
        runs = self._runs(experiment, run_batch=run_batch) if experiment else []
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
            "run_batch": run_batch, "run_batches": run_batches,
            "summary": {
                "runs": len(runs), "finished": counts["finished"],
                "running": counts["running"], "queued": counts["queued"],
                "blocked": counts["blocked"], "unknown": counts["unknown"],
                "stopped": counts["stopped"],
                "budget_used": sum(row["budget_used"] for row in runs),
                "budget": sum(row["budget"] for row in runs),
                "valid_nodes": sum(row["valid_nodes"] for row in runs),
                **_candidate_summary(runs),
                "timing": batch_timing(runs),
            },
            "tasks": groups,
        }

    # ---------- cross-batch comparison ----------

    def cohorts(self):
        """Comparable result sets: one per batch, or one per held-out variant."""
        items = []
        for batch in self.batches():
            variants = _merged_heldout(self.results_root / batch["id"])
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
        return self._runs(batch)

    def compare(self, cohort_ids):
        known = {item["id"]: item for item in self.cohorts()}
        chosen = [known[cohort] for cohort in cohort_ids if cohort in known]
        rows_by_batch = {item["batch"]: self._task_rows(item["batch"]) for item in chosen}
        heldout_by_batch = {item["batch"]: _merged_heldout(self.results_root / item["batch"]) for item in chosen}
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
                    rows = [row for row in rows if any(v in row["name"] for v in item["variant"].split("+"))]
                runs = []
                for row in rows:
                    runs.append({
                        "name": row["name"], "rep": row.get("repeat") or rep_of(row["name"]),
                        "status": row.get("status"),
                        # the final program's training score (selected node for V10.14+)
                        "train": row.get("best_fitness") if row.get("status") == "finished"
                        else (row.get("curve") or [{}])[-1].get("fitness", row.get("best_fitness")),
                        "heldout": {str(k): v for k, v in named.get(row["name"], {}).items()},
                    })
                present = {run["name"] for run in runs}
                for name, scores in named.items():  # held-out results whose run dir is gone
                    if name not in present:
                        runs.append({"name": name, "rep": rep_of(name), "status": None, "train": None,
                                     "heldout": {str(k): v for k, v in scores.items()}})
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
        row = next((row for row in self._runs(experiment, task, name)
                    if row["task"] == task and row["name"] == name), None)
        if row is None:
            return None
        run_dir = row["run_dir"]
        summary = _read_json(run_dir / "summary.json")
        curve, recent, operators, outcomes = _search_trend(run_dir, task)
        return {**{key: value for key, value in row.items() if key != "run_dir"},
                "task_meta": TASKS[task], "curve": curve, "operators": operators,
                "outcomes": outcomes, "recent": recent,
                "development": _history(run_dir, task).development_snapshot(),
                **_programs(run_dir, task, summary, curve)}


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
                return self._serve_asset(HTML_FILE, "text/html")
            if parsed.path in {"/monitor.css", "/monitor.js"}:
                content_type = "text/css" if parsed.path.endswith(".css") else "text/javascript"
                return self._serve_asset(HTML_FILE.with_name(parsed.path[1:]), content_type)
            if parsed.path == "/api/batches":
                return self._send_json({"batches": monitor.batches(), "default_batch": monitor.default_batch()})
            if parsed.path == "/api/state":
                batch = params.get("batch", [None])[0]
                run_batch = params.get("run_batch", [None])[0]
                signature = (monitor.state_signature(batch), run_batch)
                body, _ = cache.get_or_build(
                    ("state", batch, run_batch), signature,
                    lambda: _json_bytes(monitor.overview(batch, run_batch)))
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

        def _serve_asset(self, path, content_type) -> None:
            try:
                body = path.read_bytes()
            except OSError:
                return self.send_error(404, "Page not found")
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
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
    parser.add_argument("--experiment", default=None)
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
