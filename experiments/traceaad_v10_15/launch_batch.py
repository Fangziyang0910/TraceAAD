"""Prepare and launch five-task × four-repeat V10.15 runs on prior backend routes."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from experiments.infra.base import REPO_ROOT, RESULTS_ROOT, TASK_SHORT
from experiments.infra.launcher import check_backends, is_session_alive, launch_command, write_json_atomic
from traceaad.v10_15 import Config


def build_plan(previous, batch):
    source = previous["plan"]
    tasks = Counter(item["task"] for item in source)
    if len(source) != 20 or len(tasks) != 5 or set(tasks.values()) != {4}:
        raise ValueError("expected five tasks and four repeats per task")
    plan = []
    for item in source:
        task, repeat, backend, seed = (item[k] for k in ("task", "repeat", "backend", "seed"))
        if seed != repeat - 1:
            raise ValueError("prior batch does not use repeat - 1 seeds")
        short = TASK_SHORT[task]
        name = f"{batch}_{short}_traceaad_v10_15_rep{repeat}"
        session = f"{batch}_{short}_v1015_r{repeat}"
        command = ["uv", "run", "python", "-m", "experiments.traceaad_v10_15.run",
                   "--task", task, "--backend", backend, "--repeat", str(repeat),
                   "--seed", str(seed), "--run-name", name, "--budget=1000",
                   "--eval-workers=4"]
        plan.append({"task": task, "repeat": repeat, "seed": seed, "backend": backend,
                     "session": session, "run_name": name,
                     "run_dir": str(RESULTS_ROOT / "traceaad_v10_15" / task / name),
                     "command": command, "status": "planned"})
    if len({item["run_name"] for item in plan}) != 20:
        raise ValueError("duplicate run identities")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-batch", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.batch.replace("_", "").replace("-", "").isalnum():
        raise ValueError("batch must be alphanumeric")
    previous = json.loads(args.from_batch.read_text(encoding="utf-8"))
    plan = build_plan(previous, args.batch)
    if args.dry_run:
        print(json.dumps({"batch": args.batch, "plan": plan, "policy": asdict(Config())}, indent=2))
        return
    root = RESULTS_ROOT / "traceaad_v10_15"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"batch_{args.batch}.json"
    if manifest_path.exists():
        raise ValueError("batch manifest already exists")
    for item in plan:
        if Path(item["run_dir"]).exists() or is_session_alive(item["session"]):
            raise ValueError(f"existing run or session: {item['run_name']}")
    old_live = [item["session"] for item in previous["plan"] if is_session_alive(item["session"])]
    if old_live:
        raise RuntimeError(f"previous batch still has live sessions: {old_live}")
    check_backends(item["backend"] for item in plan)
    files = sorted([*(REPO_ROOT / "traceaad/v10_15").glob("*.py"),
                    *(REPO_ROOT / "experiments/traceaad_v10_15").glob("*.py"),
                    REPO_ROOT / "core/llm.py", REPO_ROOT / "core/evaluate.py",
                    REPO_ROOT / "traceaad/v10_13/storage.py",
                    REPO_ROOT / "traceaad/v10_13/parsing.py"])
    manifest = {"batch": args.batch, "method": "v1015", "created_at": datetime.now().astimezone().isoformat(),
                "status": "launching", "previous_batch": str(args.from_batch),
                "search_policy": asdict(Config()), "sampling": previous.get("sampling"),
                "services": previous.get("services"), "repeats": 4, "eval_workers": 4,
                "implementation_files": {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in files}, "plan": plan}
    write_json_atomic(manifest_path, manifest)
    for item in plan:
        launch_command(item["session"], item["command"])
        item.update(status="started_unverified", started_at=datetime.now().astimezone().isoformat())
        write_json_atomic(manifest_path, manifest)
        print(item["session"], item["backend"], flush=True)
    manifest["status"] = "started_pending_verification"
    write_json_atomic(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
