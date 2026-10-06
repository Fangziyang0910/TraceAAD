"""Selection and held-out views from the canonical result files."""

import hashlib
from pathlib import Path
import re

from traceaad.common.storage import Programs, read_json
from functools import lru_cache
from .monitor_history import finite

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


def rep_of(name):
    match = re.search(r"rep(\d+)", str(name))
    return int(match.group(1)) if match else None


def scale_of_split(task, split):
    if task == "online_bin_packing":
        _, items, capacity = split.split("_")
        return f"{int(items)//1000}k_{capacity}"
    return 50 if split == "eval" else int(split.split("_")[-1])


def batch_result_files(batch_dir):
    batch_dir = Path(batch_dir)
    paths = sorted(batch_dir.glob("*/*/heldout.json"))
    if (batch_dir / "orphan_heldout.json").exists():
        paths.append(batch_dir / "orphan_heldout.json")
    return paths


@lru_cache(maxsize=1024)
def _frozen(run_dir, signature):
    config = read_json(run_dir / "run_config.json", {})
    summary = read_json(run_dir / "summary.json", {})
    best = summary.get("best") or {}
    exported = run_dir / "best_program.py"
    code = exported.read_text(encoding="utf-8") if exported.exists() else Programs(run_dir).get(best.get("key"))
    valid = bool(summary.get("status") == "finished" and code and
                 hashlib.sha256(code.encode()).hexdigest() == best.get("key"))
    return config.get("task"), best.get("id"), best.get("key"), valid


def heldout_identity(run_dir, result):
    run_dir = Path(run_dir)
    signature = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in
                      (run_dir / "run_config.json", run_dir / "summary.json", run_dir / "best_program.py", run_dir / "programs.jsonl"))
    task, node, key, frozen = _frozen(run_dir, signature)
    if task and task != result["task"]:
        return "task_mismatch"
    status = result["verification"]
    if status != "verified":
        return status
    if not frozen:
        return "program_unverified"
    if (result.get("key"), result.get("node_id")) != (key, node):
        return "program_mismatch"
    return "verified"


def load_batch_heldout(batch_dir):
    batch_dir = Path(batch_dir)
    output = {}
    for path in batch_result_files(batch_dir):
        for result in read_json(path, []):
            task, variant, scale = result["task"], result["variant"], result["scale"]
            name = result.get("run_name") or path.parent.name
            group = output.setdefault(variant, {}).setdefault(task, {"source": "heldout.json", "runs": {}, "verification": {}})
            status = heldout_identity(batch_dir / task / name, result)
            group["verification"].setdefault(name, {})[scale] = status
            values = group["runs"].setdefault(name, {})
            if status in {"verified", "legacy"}:
                values[int(scale) if scale.isdigit() else scale] = finite(result["fitness"])
    return output


def load_selection(run_dir):
    payload = read_json(Path(run_dir) / "selection.json", {})
    results = payload.get("results", [])
    if not results:
        return None
    winner = next((r["fitness"] for r in results if r["node_id"] == payload["selected_node"]), None)
    values = [finite(r["fitness"]) for r in results]
    return {"fitness": winner, "finalists": len(results), "failed": sum(v is None for v in values),
            "ties": sum(v is not None and winner is not None and abs(v-winner) <= 1e-9*max(1,abs(winner)) for v in values),
            "distinct": len({round(v,9) for v in values if v is not None})}
