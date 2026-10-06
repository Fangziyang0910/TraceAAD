"""Selection and held-out views from the canonical result files."""

from pathlib import Path
import re

from traceaad.common.storage import heldout_identity, read_json
from .monitor_history import finite


def rep_of(name):
    match = re.search(r"rep(\d+)", str(name))
    return int(match.group(1)) if match else None


def batch_result_files(batch_dir):
    batch_dir = Path(batch_dir)
    paths = sorted(batch_dir.glob("*/*/heldout.json"))
    if (batch_dir / "orphan_heldout.json").exists():
        paths.append(batch_dir / "orphan_heldout.json")
    return paths


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
