"""Read held-out results for the monitor's cross-batch comparison.

Scores are always fitness (higher is better). Two layouts are understood:

* per-run ``heldout_<split>.json`` written by ``traceaad_v10_15.heldout``;
* batch-level ``results.json`` from the shared evaluator, in any of its
  ``results_by_split`` / ``results_by_size`` / ``eval_results_by_size`` /
  ``eval_results_by_scale`` shapes.

Native result identities are checked against the frozen final program. Legacy
exports without identity fields remain visible with their verification status.
"""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
import math
from pathlib import Path
import re


SCALES = {
    "tsp_construct": (50, 100, 200),
    "vrptw_construct": (50, 100, 200),
    "cvrp_aco": (20, 50, 100, 200),
    "op_aco": (50, 100, 200),
    "online_bin_packing": ("1k_100", "1k_500", "5k_100", "5k_500", "10k_100", "10k_500"),
}
# Held-out scales matching what the search trained on ("test"); the rest probe
# generalisation to other sizes.
TEST_SCALES = {
    "tsp_construct": {50}, "vrptw_construct": {50}, "cvrp_aco": {50}, "op_aco": {50},
    "online_bin_packing": {"1k_100", "1k_500", "5k_100", "5k_500"},
}
# Result sets that are not a batch's formal held-out evaluation.
SKIP_SOURCE = re.compile(r"(_rep\d+$|incomplete|stopped|budget\d+|per_run|_orig_)")
REP = re.compile(r"rep(\d+)")


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def rep_of(name) -> int | None:
    match = REP.search(str(name or ""))
    return int(match.group(1)) if match else None


def scale_of_split(task: str, split: str):
    """Map a V10.15 held-out split name to the task's canonical scale key."""
    if task == "online_bin_packing":
        parts = split.split("_")
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            return f"{int(parts[1]) // 1000}k_{parts[2]}"
        return None
    if split == "eval":
        return 50
    match = re.fullmatch(r"(?:test|eval)_(\d+)", split)
    return int(match.group(1)) if match else None


def _entries(payload: dict, task: str):
    """Yield (scale, results) pairs from any shared-evaluator results.json."""
    for split, block in (payload.get("results_by_split") or {}).items():
        match = re.fullmatch(r"test_(\d+)", split)  # paper_test_* is another protocol
        if match and isinstance(block, dict):
            yield int(match.group(1)), block.get("results") or []
    for key in ("results_by_size", "eval_results_by_size"):
        for block in (payload.get(key) or {}).values():
            if isinstance(block, dict) and block.get("problem_size") is not None:
                yield int(block["problem_size"]), block.get("results") or []
    for block in (payload.get("eval_results_by_scale") or {}).values():
        if isinstance(block, dict) and block.get("n_items") and block.get("capacity"):
            yield f"{int(block['n_items']) // 1000}k_{int(block['capacity'])}", block.get("results") or []


def _task_of(path: Path, payload: dict, batch_dir: Path) -> str | None:
    if payload.get("task") in SCALES:
        return payload["task"]
    for part in path.relative_to(batch_dir).parts[:-1]:
        for task in SCALES:
            if part == task or part.startswith(task + "_"):
                return task
    return None


def batch_result_files(batch_dir: Path):
    """All held-out files under a batch, for change detection and parsing."""
    files = [p for p in batch_dir.glob("**/results.json") if len(p.relative_to(batch_dir).parts) <= 4]
    files += list(batch_dir.glob("*/*/heldout_*.json"))
    return sorted(files)


VARIANT_DIR = re.compile(r"^heldout_(?:\d{8}_)?(.+)$")


def variant_of(path: Path, batch_dir: Path) -> str:
    """``heldout_<date>_<name>`` directories hold one variant each; reruns share the name."""
    match = VARIANT_DIR.match(path.relative_to(batch_dir).parts[0])
    return match.group(1) if match else ""


def load_batch_heldout(batch_dir: Path) -> dict[str, dict[str, dict]]:
    """Return ``{variant: {task: {"source": str, "runs": {run_name: {scale: fitness}}}}}``.

    Within a variant and task, the result set covering the most runs (then the
    newest) is used; partial reruns and budget-truncated evaluations are skipped.
    """
    batch_dir = Path(batch_dir)
    candidates: dict[tuple, list] = {}
    native: dict[str, dict] = {}
    verification: dict[str, dict] = {}
    for path in batch_result_files(batch_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if path.name.startswith("heldout_"):
            config = _read_json(path.parent / "run_config.json")
            task = config.get("task", path.parent.parent.name)
            split = path.stem.removeprefix("heldout_")
            scale = scale_of_split(task, split) if task in SCALES else None
            if scale is not None:
                state = ("split_mismatch" if payload.get("split") != split
                         else _heldout_identity(path.parent, payload))
                checks = verification.setdefault(task, {}).setdefault(path.parent.name, {})
                checks[str(scale)] = state
                scores = native.setdefault(task, {}).setdefault(path.parent.name, {})
                if state in {"verified", "legacy"}:
                    scores[scale] = _finite(payload.get("fitness"))
            continue
        task = _task_of(path, payload, batch_dir)
        if task is None or SKIP_SOURCE.search(path.parent.name):
            continue
        runs: dict[str, dict] = {}
        checks: dict[str, dict] = {}
        for scale, results in _entries(payload, task):
            for result in results:
                if result.get("run_name"):
                    name = str(result["run_name"])
                    state = (_heldout_identity(batch_dir / task / name, {"task": task, **result})
                             if result.get("key") is not None else "legacy")
                    checks.setdefault(name, {})[str(scale)] = state
                    scores = runs.setdefault(name, {})
                    if state in {"verified", "legacy"}:
                        scores[scale] = _finite(result.get("eval_score"))
        if runs:
            key = (variant_of(path, batch_dir), task)
            candidates.setdefault(key, []).append(
                (len(runs), str(payload.get("created_at") or ""), str(path.parent.relative_to(batch_dir)), runs, checks))
    output: dict[str, dict] = {}
    for (variant, task), options in candidates.items():
        _, _, source, runs, checks = max(options, key=lambda item: (item[0], item[1]))
        output.setdefault(variant, {})[task] = {"source": source, "runs": runs, "verification": checks}
    for task, runs in native.items():  # per-run V10.15 results
        previous = output.setdefault("", {}).get(task, {})
        merged = {name: dict(scores) for name, scores in previous.get("runs", {}).items()}
        checks = {name: dict(states) for name, states in previous.get("verification", {}).items()}
        for name, scores in runs.items():
            merged.setdefault(name, {}).update(scores)
            checks.setdefault(name, {}).update(verification[task][name])
            # Explicitly rejected results must not fall back to another stale export.
            for scale, state in verification[task][name].items():
                if state not in {"verified", "legacy"}:
                    key = int(scale) if scale.isdigit() else scale
                    merged[name].pop(key, None)
        output[""][task] = {"source": "heldout_<split>.json", "runs": merged,
                              "verification": checks}
    return output


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def selection_identity(payload):
    """V10.14 and current selection files name the same two identity fields."""
    return (payload.get("selected_node", payload.get("selected_anchor")),
            payload.get("selected_key", payload.get("selected_source_sha256")))


@lru_cache(maxsize=1024)
def _frozen_program(run_dir: Path, stamps):
    selection = _read_json(run_dir / "selection.json")
    summary = _read_json(run_dir / "logs/run_summary.json")
    node, key = selection_identity(selection)
    if not key or node is None or summary.get("status") != "finished":
        return node, key, False
    best = summary.get("best") or {}
    if best.get("id", best.get("node_id")) != node:
        return node, key, False
    try:
        actual = hashlib.sha256((run_dir / "best_program.py").read_bytes()).hexdigest()
    except OSError:
        return node, key, False
    return node, key, actual == key


def _heldout_identity(run_dir: Path, payload: dict) -> str:
    config = _read_json(run_dir / "run_config.json")
    task = config.get("task", run_dir.parent.name)
    if payload.get("task") != task:
        return "task_mismatch"
    if payload.get("key") is None and payload.get("node_id") is None:
        return "legacy"  # Historical evaluator exports did not store program identity.
    paths = (run_dir / "selection.json", run_dir / "logs/run_summary.json", run_dir / "best_program.py")
    stamps = []
    for path in paths:
        try:
            stat = path.stat()
            stamps.append((stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino))
        except OSError:
            stamps.append(None)
    node, key, frozen = _frozen_program(run_dir, tuple(stamps))
    if not frozen:
        return "program_unverified"
    if payload.get("key") != key or payload.get("node_id") != node:
        return "program_mismatch"
    return "verified"


def load_selection(run_dir: Path) -> dict | None:
    """Selection outcome, including how many finalists tied with the winner."""
    try:
        payload = json.loads((Path(run_dir) / "selection.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        return None
    selected, _ = selection_identity(payload)
    chosen = next((r for r in results if isinstance(r, dict) and selected is not None
                   and r.get("node_id", r.get("anchor_id")) == selected), None)
    def score(result):
        if not isinstance(result, dict):
            return None
        value = result.get("fitness")
        outcome = result.get("outcome")
        if value is None and isinstance(outcome, dict):
            value = outcome.get("fitness")
        return _finite(value)
    winner = score(chosen) if chosen else None
    valid = [score(r) for r in results]
    return {
        "fitness": winner,
        "finalists": len(results),
        "failed": sum(v is None for v in valid),
        "ties": sum(v is not None and winner is not None and abs(v - winner) <= 1e-9 * max(1.0, abs(winner))
                    for v in valid),
        "distinct": len({round(v, 9) for v in valid if v is not None}),
    }
