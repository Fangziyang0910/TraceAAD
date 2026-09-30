"""Read held-out results for the monitor's cross-batch comparison.

Scores are always fitness (higher is better). Two layouts are understood:

* per-run ``heldout_<split>.json`` written by ``traceaad_v10_15.heldout``;
* batch-level ``results.json`` from the shared evaluator, in any of its
  ``results_by_split`` / ``results_by_size`` / ``eval_results_by_size`` /
  ``eval_results_by_scale`` shapes.

Both describe the same protocols (ACO ``test_<n>`` splits; generated eval
instances with seed 2025), so their numbers are comparable.
"""

from __future__ import annotations

import json
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
    except (TypeError, ValueError):
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
    for path in batch_result_files(batch_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if path.name.startswith("heldout_"):
            task = payload.get("task")
            scale = scale_of_split(task, str(payload.get("split", ""))) if task in SCALES else None
            if scale is not None:
                native.setdefault(task, {}).setdefault(path.parent.name, {})[scale] = _finite(payload.get("fitness"))
            continue
        task = _task_of(path, payload, batch_dir)
        if task is None or SKIP_SOURCE.search(path.parent.name):
            continue
        runs: dict[str, dict] = {}
        for scale, results in _entries(payload, task):
            for result in results:
                if result.get("run_name"):
                    runs.setdefault(str(result["run_name"]), {})[scale] = _finite(result.get("eval_score"))
        if runs:
            key = (variant_of(path, batch_dir), task)
            candidates.setdefault(key, []).append(
                (len(runs), str(payload.get("created_at") or ""), str(path.parent.relative_to(batch_dir)), runs))
    output: dict[str, dict] = {}
    for (variant, task), options in candidates.items():
        _, _, source, runs = max(options, key=lambda item: (item[0], item[1]))
        output.setdefault(variant, {})[task] = {"source": source, "runs": runs}
    for task, runs in native.items():  # per-run V10.15 results
        output.setdefault("", {})[task] = {"source": "heldout_<split>.json", "runs": runs}
    return output


def _read_config(run_dir: Path) -> dict:
    try:
        value = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_selection(run_dir: Path) -> dict | None:
    """Selection outcome, including how many finalists tied with the winner."""
    try:
        payload = json.loads((Path(run_dir) / "selection.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        return None
    chosen = next((r for r in results if r.get("node_id") == payload.get("selected_node")), None)
    winner = _finite(chosen.get("fitness")) if chosen else None
    valid = [_finite(r.get("fitness")) for r in results]
    return {
        "fitness": winner,
        "finalists": len(results),
        "failed": sum(v is None for v in valid),
        "ties": sum(v is not None and winner is not None and abs(v - winner) <= 1e-9 * max(1.0, abs(winner))
                    for v in valid),
        "distinct": len({round(v, 9) for v in valid if v is not None}),
    }
