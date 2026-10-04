"""Evaluate every selected program in a V10.17 batch at all report sizes."""

import argparse
import json
from pathlib import Path

from traceaad.v10_13.storage import write_json

from .heldout import evaluate_run


SPLITS = {
    "tsp_construct": ("eval_50", "eval_100", "eval_200"),
    "vrptw_construct": ("eval_50", "eval_100", "eval_200"),
    "cvrp_aco": ("test_20", "test_50", "test_100", "test_200"),
    "op_aco": ("test_50", "test_100", "test_200"),
    "online_bin_packing": tuple(f"eval_{items}_{capacity}"
                                for items in (1000, 5000, 10000) for capacity in (100, 500)),
}


def jobs(manifest):
    if manifest.get("method") != "v1017":
        raise ValueError("expected a V10.17 batch manifest")
    plan = manifest["plan"]
    if len(plan) != 20:
        raise ValueError("expected twenty search runs")
    return [(row["task"], row["repeat"], Path(row["run_dir"]), split)
            for row in plan for split in SPLITS[row["task"]]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float,
                        help="explicit limit for every held-out evaluation in this batch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    plan = jobs(manifest)
    report = []
    for task, repeat, run_dir, split in plan:
        summary_path = run_dir / "logs" / "run_summary.json"
        ready = (summary_path.exists() and
                 json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "finished")
        if not ready:
            outcome = {"status": "waiting"}
        elif args.dry_run:
            outcome = {"status": "existing" if (run_dir / f"heldout_{split}.json").exists() else "ready"}
        else:
            outcome = {"status": "evaluated", **evaluate_run(
                run_dir, split=split, workers=args.workers,
                timeout_seconds=args.timeout_seconds)}
        report.append({"task": task, "repeat": repeat, "run_dir": str(run_dir),
                       "split": split, **outcome})
        if not args.dry_run:
            write_json(args.batch_manifest.with_name(args.batch_manifest.stem + "_heldout.json"),
                       {"batch": manifest["batch"], "results": report})
    print(json.dumps({"batch": manifest["batch"], "jobs": len(report),
                      "waiting": sum(item["status"] == "waiting" for item in report),
                      "ready": sum(item["status"] == "ready" for item in report)}, indent=2))


if __name__ == "__main__":
    main()
