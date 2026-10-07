"""Prepare a current search version using the routes in a previous batch."""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from experiments.infra.base import REPO_ROOT, RESULTS_ROOT, SAMPLING_PROFILES, TASK_SHORT
from experiments.infra.launcher import launch_plan
from traceaad.common.config import REVISION

from benchmarks.tasks import evaluation_limits


def build_plan(previous, batch, experiment, module):
    source = previous["plan"]
    plan = []
    for item in source:
        task, repeat, backend, seed = (item[k] for k in ("task", "repeat", "backend", "seed"))
        if seed != repeat - 1:
            raise ValueError("prior batch does not use repeat - 1 seeds")
        short = TASK_SHORT[task]
        name = f"{batch}_{short}_{experiment}_rep{repeat}"
        tag = experiment.removeprefix("traceaad_").replace("v10_", "v10")
        session = f"{batch}_{short}_{tag}_r{repeat}"
        command = ["uv", "run", "python", "-m", module,
                   "--task", task, "--backend", backend, "--repeat", str(repeat),
                   "--seed", str(seed), "--run-name", name, "--budget=1000",
                   "--eval-workers=4", "--experiment", experiment]
        plan.append({"task": task, "repeat": repeat, "seed": seed, "backend": backend,
                     "session": session, "run_name": name,
                     "run_dir": str(RESULTS_ROOT / experiment / task / name),
                     "command": command, "status": "planned"})
    if len({item["run_name"] for item in plan}) != len(plan):
        raise ValueError("duplicate run identities")
    return plan


def main(config_class, experiment, module, method, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-batch", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", default=experiment,
                        help="results directory under experiments_result")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.batch.replace("_", "").replace("-", "").isalnum():
        raise ValueError("batch must be alphanumeric")
    previous = json.loads(args.from_batch.read_text(encoding="utf-8"))
    if not args.experiment.replace("_", "").isalnum():
        raise ValueError("experiment must be alphanumeric with underscores")
    plan = build_plan(previous, args.batch, args.experiment, module)
    sampling = {**SAMPLING_PROFILES[False], "enable_thinking": False}
    if args.dry_run:
        print(json.dumps({"batch": args.batch, "experiment": args.experiment, "plan": plan,
                          "policy": asdict(config_class()), "sampling": sampling,
                          "evaluation_limits": evaluation_limits()}, indent=2))
        return
    root = RESULTS_ROOT / args.experiment
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"batch_{args.batch}.json"
    hashes = implementation_files(module)
    manifest = {"batch": args.batch, "method": method, "revision": REVISION, "created_at": datetime.now().astimezone().isoformat(),
                "status": "launching", "previous_batch": str(args.from_batch),
                "experiment": args.experiment, "search_policy": asdict(config_class()),
                # Requests carry every sampling control, so servers' own defaults do not apply.
                "sampling": sampling, "evaluation_limits": evaluation_limits(),
                "services": previous.get("services"),
                "repeats": max(item["repeat"] for item in plan), "eval_workers": 4,
                "implementation_files": hashes, "plan": plan}
    launch_plan(manifest_path, manifest)


def implementation_files(module):
    version = module.split(".")[1].removeprefix("traceaad_")
    # A version may build on an earlier one (V10.20 on V10.17), so every method package is hashed.
    folders = ("experiments/infra", "experiments/traceaad_" + version, "core")
    files = sorted({p for folder in folders for p in (REPO_ROOT / folder).glob("*.py")}
                   | set((REPO_ROOT / "traceaad").rglob("*.py")))
    files += sorted((REPO_ROOT / "benchmarks").rglob("*.py"))
    files += [REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"]
    return {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
