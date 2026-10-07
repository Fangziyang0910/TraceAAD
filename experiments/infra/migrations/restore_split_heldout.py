"""Restore held-out results that were stored one run and one split per directory.

The 2026-10-03 backfills wrote ``heldout_<date>_<variant>/<task>/<run>_<split>/results.json``.
The 2026-10-06 conversion kept, per variant and task, only the one results file
covering the most runs, so each backfill lost all but one run-split. This adds
the missing records to each run's ``heldout.json``; existing records are kept
and must agree with the source.

    uv run python -m experiments.infra.migrations.restore_split_heldout [--apply]
"""

import argparse
import json
import math
from pathlib import Path

from traceaad.common.storage import read_json, write_json

from .legacy_results import SCALES, _entries, _task_of, variant_of

ROOT = Path(__file__).resolve().parents[3] / "experiments_result"


def collect(batch_dir: Path):
    """Yield (task, run_name, variant, scale, value) from per-split held-out dirs."""
    for path in sorted(batch_dir.glob("heldout_*/*/*/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = _task_of(path, payload, batch_dir)
        if task not in SCALES:
            continue
        for scale, results in _entries(payload, task):
            for result in results:
                if result.get("run_name"):
                    yield task, str(result["run_name"]), variant_of(path, batch_dir), str(scale), result.get("eval_score")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    added, conflicts, missing_runs = {}, [], []
    for batch_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        updates = {}
        for task, name, variant, scale, value in collect(batch_dir):
            run = batch_dir / task / name
            if not (run / "run_config.json").exists():
                missing_runs.append(str(run))
                continue
            records = updates.setdefault(run, read_json(run / "heldout.json", []))
            existing = next((r for r in records if r["variant"] == variant and str(r["scale"]) == scale), None)
            fitness = value if isinstance(value, (int, float)) and math.isfinite(value) else None
            if existing is not None:
                if existing.get("fitness") != fitness:
                    conflicts.append((str(run), variant, scale, existing.get("fitness"), fitness))
                continue
            records.append({"variant": variant, "scale": scale, "task": task,
                            "fitness": fitness, "verification": "legacy"})
            added[batch_dir.name] = added.get(batch_dir.name, 0) + 1
        if args.apply:
            for run, records in updates.items():
                write_json(run / "heldout.json", records)
    print(json.dumps({"added": added, "conflicts": conflicts, "missing_runs": missing_runs,
                      "applied": args.apply}, indent=2))


if __name__ == "__main__":
    main()
