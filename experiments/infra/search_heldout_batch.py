"""Evaluate a search batch's selected programs at all report sizes."""

import argparse
import json
from pathlib import Path

from benchmarks.tasks import PRIMARY_SPLITS, SPLITS

from traceaad.common.storage import read_json, write_json

from .search_heldout import evaluate_run
from .batch_status import run_path


def jobs(manifest, root=None, primary=False):
    plan = manifest["plan"]
    return [(row["task"], row["repeat"], (run_path(root, row) if root else Path(row["run_dir"])), split)
            for row in plan for split in (PRIMARY_SPLITS if primary else SPLITS)[row["task"]]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="defaults to each run's saved execution settings")
    parser.add_argument('--scheduler-socket', help='override saved CPU scheduler; empty string disables it')
    parser.add_argument("--timeout-seconds", type=float,
                        help="per-instance limit for new V10.21 runs; total limit for legacy runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--primary", action="store_true", help="only primary same-scale tests for the six-task suite")
    args = parser.parse_args(argv)
    manifest = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    plan = jobs(manifest, args.batch_manifest.parent, args.primary)
    report = []
    for task, repeat, run_dir, split in plan:
        summary_path = run_dir / "summary.json"
        ready = (summary_path.exists() and
                 json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "finished")
        if not ready:
            outcome = {"status": "waiting"}
        elif args.dry_run:
            exists = any(r.get("split") == split and not r["variant"]
                         for r in read_json(run_dir / "heldout.json", []))
            outcome = {"status": "existing" if exists else "ready"}
        else:
            outcome = {"status": "evaluated", **evaluate_run(
                run_dir, split=split, workers=args.workers,
                timeout_seconds=args.timeout_seconds, scheduler_socket=args.scheduler_socket)}
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
