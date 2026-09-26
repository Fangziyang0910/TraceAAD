"""Small V10.13 batch monitor.

The monitor consumes the V10.13 batch manifest, run configuration, search
journal, and final summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from traceaad.v10_13.storage import JOURNAL_NAME, RunStorage, read_journal


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments_result/runs/traceaad_v10_13"
HTML_FILE = Path(__file__).with_name("training_monitor.html")

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


class V1013Monitor:
    def __init__(self, results_root: Path = DEFAULT_RESULTS_ROOT, batch: str | None = None):
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


def make_request_handler(monitor: V1013Monitor) -> type[BaseHTTPRequestHandler]:
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
    parser = argparse.ArgumentParser(description="TraceAAD V10.13 training monitor")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--batch", "--version", dest="batch", default=None)
    args = parser.parse_args()

    monitor = V1013Monitor(args.results_dir, args.batch)
    server = ThreadingHTTPServer((args.host, args.port), make_request_handler(monitor))
    selected = monitor.overview().get("batch") or "无批次"
    print(f"TraceAAD V10.13 monitor: http://127.0.0.1:{args.port} ({selected})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
