"""Read-only training monitor for experiment results."""

from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from experiments.infra.artifacts import pick_best_sample
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


def _sample(points: list[dict[str, Any]], limit: int = 240) -> list[dict[str, Any]]:
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    indexes = {round(index * step) for index in range(limit)}
    return [point for index, point in enumerate(points) if index in indexes]


def _search_events(run_dir: Path):
    journal = run_dir / JOURNAL_NAME
    if journal.exists():
        with journal.open(encoding="utf-8") as handle:
            for line in handle:
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


def _search_trend(run_dir: Path, task: str):
    streams = {}
    for source, event in _search_events(run_dir):
        if not isinstance(event, dict):
            continue
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
                continue
        else:
            index = event.get("budget_used", event.get("evaluation_id"))
            score = event.get("fitness")
        if index is None or score is None:
            continue
        try:
            index, score = int(index), float(score)
        except (ValueError, TypeError):
            continue
        if not (index >= 0 and abs(score) < float("inf")):
            continue
        streams.setdefault(source, []).append((index, score, event))
    for source in ("native", "events.jsonl", "evaluations.csv",
                   "method_events.jsonl", "artifacts/candidates.jsonl"):
        if source in streams:
            break
    else:
        return [], [], {}, {}
    best = None
    points, recent = [], []
    operators, outcomes = Counter(), Counter()
    for index, score, event in sorted(streams[source], key=lambda row: row[0]):
        best = score if best is None else max(best, score)
        points.append({"evaluation": index, "value": _objective(best, task)})
        operator = event.get("operator") or event.get("action") or "unknown"
        status = event.get("status") or event.get("outcome") or "unknown"
        operators[str(operator)] += 1
        outcomes[str(status)] += 1
        recent.append({"evaluation": index, "operator": operator,
                       "status": status, "value": _objective(score, task)})
    return _sample(points), recent[-12:][::-1], dict(operators), dict(outcomes)


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

    def _run_summary(self, row: dict[str, Any]) -> dict[str, Any]:
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

        return {
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
        }

    def overview(self, batch: str | None = None) -> dict[str, Any]:
        manifest = self._manifest(batch)
        if not manifest:
            return {"batch": None, "summary": {}, "tasks": []}

        runs = [self._run_summary(row) for row in manifest["plan"]]
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
        events = storage.records("events")
        nodes = storage.records("nodes")

        best_fitness = None
        curve = []
        status_counts: Counter[str] = Counter()
        operator_counts: Counter[str] = Counter()
        for event in events:
            status_counts[str(event.get("status") or "unknown")] += 1
            operator_counts[str(event.get("operator") or "unknown")] += 1
            fitness = event.get("fitness")
            if fitness is not None:
                fitness = float(fitness)
                best_fitness = fitness if best_fitness is None else max(best_fitness, fitness)
            if event.get("budget_used") is not None and best_fitness is not None:
                curve.append({
                    "evaluation": int(event["budget_used"]),
                    "value": _objective(best_fitness, task),
                })

        valid_nodes = [node for node in nodes if node.get("fitness") is not None]
        best_node = max(valid_nodes, key=lambda node: float(node["fitness"]), default=None)
        if best_node is None:
            final_best = _read_json(run_dir / "logs/run_summary.json").get("best")
            best_node = final_best if isinstance(final_best, dict) else None

        recent = []
        for event in events[-12:][::-1]:
            fitness = event.get("fitness")
            recent.append({
                "candidate": event.get("candidate_id"),
                "evaluation": event.get("budget_used") or event.get("evaluation_id"),
                "operator": event.get("operator"),
                "status": event.get("status"),
                "value": _objective(float(fitness), task) if fitness is not None else None,
                "reason": event.get("reason") or event.get("error_type"),
            })

        return {
            **summary,
            "task_meta": TASKS[task],
            "curve": _sample(curve),
            "operators": dict(operator_counts),
            "outcomes": dict(status_counts),
            "recent": recent,
            "best": None if best_node is None else {
                "id": best_node.get("id", best_node.get("node_id")),
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
            elif source == "events.jsonl":
                counted = event.get("budget_used") is not None
                valid = isinstance(event.get("fitness"), (int, float))
                position = event.get("budget_used")
            else:
                continue
            record = streams.setdefault(source, [0, 0])
            if counted:
                record[0] = max(record[0], int(position)) if position is not None else record[0] + 1
            record[1] += valid
        result = tuple(streams.get("events.jsonl", streams.get("artifacts/candidates.jsonl", [0, 0])))
        self._progress_cache[run_dir] = stamp, result
        return result

    def batches(self) -> list[dict[str, str]]:
        names = []
        for directory in self.results_root.iterdir():
            if directory.is_dir() and any(directory.glob("*/*/run_config.json")):
                names.append(directory.name)
        names.sort(reverse=True)
        if self.default_experiment in names:
            names.remove(self.default_experiment)
            names.insert(0, self.default_experiment)
        return [{"id": name, "label": name} for name in names]

    def _runs(self, experiment: str):
        if experiment not in {item["id"] for item in self.batches()}:
            return []
        rows = []
        for config_path in sorted((self.results_root / experiment).glob("*/*/run_config.json")):
            run_dir = config_path.parent
            config = _read_json(config_path)
            summary = _read_json(run_dir / "logs/run_summary.json")
            raw_status = summary.get("status")
            if raw_status == "finished":
                status = "finished"
            elif raw_status == "unknown":
                status = "unknown"
            else:
                status = "blocked" if raw_status else "queued"
            params = config.get("method_params") or {}
            best = summary.get("best") or {}
            if not isinstance(best, dict):
                best = {}
            score = best.get("fitness", summary.get("best_score"))
            budget = summary.get("budget", summary.get("budget_slots", params.get("budget", params.get("max_sample_nums", 0))))
            used = summary.get("budget_used", summary.get("budget_slots", summary.get("num_samples", summary.get("evaluator_call_count", 0))))
            nodes = summary.get("num_nodes", summary.get("n_algorithms", summary.get("evaluate_success_program_num", 0)))
            if raw_status == "unknown":
                used, nodes = self._recorded_progress(run_dir)
            task = config.get("task", run_dir.parent.name)
            if task not in TASKS:
                continue
            rows.append({
                "task": task, "name": run_dir.name, "repeat": config.get("repeat"),
                "seed": config.get("seed"), "backend": config.get("backend"),
                "status": status, "raw_status": raw_status, "budget": int(budget or 0),
                "budget_used": int(used or 0), "valid_nodes": int(nodes or 0),
                "best_fitness": score, "best_value": _objective(score, task),
                "updated_at": summary.get("finished_at"), "run_dir": run_dir,
            })
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
            },
            "tasks": groups,
        }

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
        curve, recent, operators, outcomes = _search_trend(run_dir, task)
        return {
            **{key: value for key, value in row.items() if key != "run_dir"},
            "task_meta": TASKS[task], "curve": curve, "operators": operators,
            "outcomes": outcomes, "recent": recent,
            "best": None if best is None else {
                "id": best.get("id", best.get("node_id")),
                "value": _objective(best.get("fitness"), task),
                "operator": best.get("operator"), "idea": best.get("idea") or "",
                "code": best.get("code") or "",
            },
        }


def make_request_handler(monitor: ResultsMonitor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if parsed.path in {"/", "/index.html"}:
                return self._serve_html()
            if parsed.path == "/api/batches":
                return self._send_json({"batches": monitor.batches()})
            if parsed.path == "/api/state":
                return self._send_json(monitor.overview(params.get("batch", [None])[0]))
            if parsed.path == "/api/run":
                batch = params.get("batch", [""])[0]
                task = params.get("task", [""])[0]
                name = params.get("name", [""])[0]
                detail = monitor.run_detail(batch, task, name)
                return self._send_json(detail) if detail else self.send_error(404, "Run not found")
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
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
