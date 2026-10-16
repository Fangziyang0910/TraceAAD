"""Dashboard views over committed experiment records."""

from collections import Counter
from functools import lru_cache
from pathlib import Path
import json
import time

from benchmarks.tasks import FIXED_TASKS, SCALES, TEST_SCALES
from experiments.infra.result_storage import heldout_identity, selected_program
from experiments.infra.timing import batch_timing, search_timing
from traceaad.storage import JOURNAL_NAME, read_json
from .history import TrainingHistory, finite

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "experiments_result"

def batch_result_files(batch_dir):
    batch_dir = Path(batch_dir)
    return sorted(batch_dir.glob("*/*/heldout.json"))


def load_batch_heldout(batch_dir):
    batch_dir = Path(batch_dir)
    output = {}
    for path in batch_result_files(batch_dir):
        for result in read_json(path, []):
            task, variant, scale = result["task"], result["variant"], result["scale"]
            name = path.parent.name
            group = output.setdefault(variant, {}).setdefault(task, {"source": "heldout.json", "runs": {}, "verification": {}})
            status = heldout_identity(batch_dir / task / name, result)
            group["verification"].setdefault(name, {})[scale] = status
            values = group["runs"].setdefault(name, {})
            if status in {"verified", "legacy"}:
                values[int(scale) if scale.isdigit() else scale] = finite(result["fitness"])
    return output


TASKS = {
    "tsp_construct": {"label": "TSP 构造", "unit": "距离"},
    "cvrp_aco": {"label": "CVRP-ACO", "unit": "距离"},
    "op_aco": {"label": "OP-ACO", "unit": "负收益"},
    "online_bin_packing": {"label": "在线装箱", "unit": "箱数"},
    "vrptw_construct": {"label": "VRPTW 构造", "unit": "距离"},
    "fssp_gls": {"label": "FSSP-GLS", "unit": "makespan"},
    "graph_colouring": {"label": "图着色", "unit": "颜色数"},
    "jssp_construct": {"label": "作业车间调度", "unit": "makespan"},
}


def _run_config_paths(directory: Path):
    """List canonical runs, excluding temporary aliases used by live writers."""
    return sorted(path for path in directory.glob("*/*/run_config.json")
                  if not path.parent.is_symlink() and not path.parent.parent.is_symlink())


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
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


@lru_cache(maxsize=128)
def _history(run_dir: Path):
    return TrainingHistory(run_dir)


def _program_view(program):
    if not isinstance(program, dict):
        return None
    score = finite(program.get("fitness"))
    return {"id": program.get("id"), "fitness": score,
            "operator": program.get("action"),
            "idea": program.get("idea") or "", "code": program.get("code") or ""}


def _programs(run_dir, curve):
    history = _history(run_dir)
    point = next((p for p in reversed(curve) if p["kind"] in {"initial", "breakthrough"}), None)
    search = history.node(point["node_id"], by_node=True) if point and point.get("node_id") is not None else None
    if search and point:
        search = {**search, "fitness": point["fitness"]}
    selected = selected_program(run_dir)
    search, selected = _program_view(search), _program_view(selected)
    breakthroughs = []
    for p in curve:
        if p["kind"] not in {"initial", "breakthrough"} or p.get("node_id") is None:
            continue
        view = _program_view(history.node(p["node_id"], by_node=True)) or {"id": p.get("candidate")}
        breakthroughs.append({**view, "fitness": p["fitness"],
                              "evaluation": p["evaluation"], "node_id": p["node_id"], "operator": p.get("operator"),
                              "parent_id": p.get("parent_id"), "gain": p.get("gain"),
                              "idea": view.get("idea") or p.get("idea") or ""})
    return {"search_best": search, "selected_best": selected, "best": selected or search,
            "breakthroughs": breakthroughs}


class ResultsMonitor:
    """Read current experiment records for the dashboard."""

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
            if raw_status == 'archived' or metadata.get('batch_status') == 'archived':
                status = 'archived'
            elif raw_status == "finished":
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
            score_unit = TASKS[task]['unit']
            if (task in FIXED_TASKS and config.get('score_metric') != 'raw_objective'
                    and summary.get('raw_search', {}).get('status') not in ('complete', 'complete_with_missing')):
                score_unit = '参考偏差 %'
            history = _history(run_dir)
            curve = history.read()[0]
            progress = history.progress_snapshot()
            if progress.get("candidate_count"):
                nodes = progress["valid_nodes"]
                if status != "finished" or not used:
                    used = progress["budget_used"]
                if status != "finished" or score is None:
                    score = progress["best_fitness"]
            if status == 'stopped' and budget and used >= budget and not summary.get('error'):
                status = 'search_complete'
            row = {
                "task": task, "name": run_dir.name, "repeat": config.get("repeat", metadata.get("repeat")),
                "run_batch": metadata.get('run_batch'),
                "seed": config.get("seed", metadata.get("seed")), "backend": config.get("backend", metadata.get("backend")),
                "status": status, "raw_status": raw_status, "budget": int(budget or 0),
                "budget_used": int(used or 0), "valid_nodes": int(nodes or 0),
                "best_fitness": score,
                "score_unit": score_unit,
                "updated_at": summary.get("finished_at") or history.clock.get("completed_at") or metadata.get("started_at"), "run_dir": run_dir,
                "error": None if status == "running" else summary.get("error") or metadata.get("last_error"),
                "curve": _list_curve(curve),
                **{k: progress.get(k, 0) for k in ("candidate_count", "valid_candidate_count")},
                "valid_rate": progress.get("valid_rate"),
                "x_label": progress.get("x_label", "已记录序号"),
            }
            timing_summary = {**summary, "started_at": summary.get("started_at") or metadata.get("started_at")}
            row["timing"] = search_timing(row, timing_summary, _history(run_dir).timing_snapshot(),
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
                units = {row['score_unit'] for row in task_runs}
                groups.append({"key": task, **meta,
                    "unit": next(iter(units)) if len(units) == 1 else '评分口径混合', "runs": [
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
                "search_complete": counts['search_complete'],
                "archived": counts['archived'],
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

    def compare(self, cohort_ids):
        known = {item["id"]: item for item in self.cohorts()}
        chosen = [known[cohort] for cohort in cohort_ids if cohort in known]
        rows_by_batch = {item["batch"]: self._runs(item["batch"]) for item in chosen}
        heldout_by_batch = {item["batch"]: load_batch_heldout(self.results_root / item["batch"]) for item in chosen}
        tasks = []
        for task, meta in TASKS.items():
            entries = []
            for item in chosen:
                heldout = heldout_by_batch[item["batch"]].get(item["variant"], {}).get(task, {})
                named = heldout.get("runs", {})
                rows = [row for row in rows_by_batch[item["batch"]] if row["task"] == task]
                if named or item["variant"]:
                    rows = [row for row in rows if row["name"] in named]
                runs = []
                for row in rows:
                    runs.append({
                        "name": row["name"], "rep": row["repeat"],
                        "status": row.get("status"),
                        # the frozen program's training score
                        "train": row.get("best_fitness") if row.get("status") == "finished"
                        else (row.get("curve") or [{}])[-1].get("fitness", row.get("best_fitness")),
                        "heldout": {str(k): v for k, v in named.get(row["name"], {}).items()},
                    })
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
        curve, recent, operators, outcomes = _history(run_dir).read()
        return {**{key: value for key, value in row.items() if key != "run_dir"},
                "task_meta": {**TASKS[task], "unit": row['score_unit']}, "curve": curve, "operators": operators,
                "outcomes": outcomes, "recent": recent,
                **_programs(run_dir, curve)}
