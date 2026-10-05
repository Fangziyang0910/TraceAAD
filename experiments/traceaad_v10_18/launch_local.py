"""Launch the 15-run V10.18 batch on this machine (five tasks x three repeats).

Searches and evaluations run here; the model is served by the two server3 Qwen
services, alternated per task and repeat as in the V10.15-6/V10.16/V10.17
batches (8 / 7 runs). Seeds 0-2, four evaluation workers, 1000 generations per
run. Evaluations on this machine run faster than on server3, so wall-clock
limits are not equivalent across the two hosts. Without ``--launch`` it prints the plan.

    uv run python -m experiments.traceaad_v10_18.launch_local --batch 20261004_local_v1018
    uv run python -m experiments.traceaad_v10_18.launch_local --batch 20261004_local_v1018 --launch
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import urllib.request

from experiments.infra.base import BACKENDS, SAMPLING_PROFILES, TASKS, _process_cmdlines
from experiments.infra.env import resolve_llm_api_key
from experiments.infra.launcher import check_backends, is_session_alive, launch_command, write_json_atomic
from experiments.traceaad_v10_18.launch_batch import build_plan, evaluation_limits
from traceaad.v10_18 import Config

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "traceaad_v10_18"
BACKEND_NAMES = ("server3", "server3b")


def plan_for(batch):
    source = {"plan": [{"task": task, "repeat": repeat, "seed": repeat - 1,
                        "backend": "server3" if (i + repeat) % 2 else "server3b"}
                       for i, task in enumerate(TASKS) for repeat in range(1, 5)]}
    plan = [item for item in build_plan(source, batch, EXPERIMENT) if item["repeat"] <= 3]
    for item in plan:
        item["command"][:3] = [str(ROOT / ".venv/bin/python")]
        item["startup_log"] = str(ROOT / "experiments_result" / EXPERIMENT / "launch_logs" / (item["session"] + ".log"))
    assert len(plan) == 15
    assert Counter(p["task"] for p in plan) == Counter({task: 3 for task in TASKS})
    assert Counter(p["backend"] for p in plan) == Counter(server3=8, server3b=7)
    assert all(p["seed"] == p["repeat"] - 1 for p in plan)
    return plan


def served_models():
    services = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for name in BACKEND_NAMES:
        profile = BACKENDS[name]
        request = urllib.request.Request(profile.base_url + "/models", headers={
            "Authorization": "Bearer " + resolve_llm_api_key(base_url=profile.base_url)})
        with opener.open(request, timeout=15) as response:
            model = next(m for m in json.load(response)["data"] if m["id"] == profile.model)
        if model["max_model_len"] < 32768:
            raise RuntimeError(f"{name} serves a context shorter than 32K")
        services[name] = {"endpoint": profile.base_url, "model": model}
    return services


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_for(args.batch)
    manifest_path = ROOT / "experiments_result" / EXPERIMENT / f"batch_{args.batch}.json"
    if not args.launch:
        print(json.dumps({"manifest": str(manifest_path), "plan": plan,
                          "evaluation_limits": evaluation_limits()}, indent=2))
        return
    if manifest_path.exists():
        raise SystemExit("batch already exists")
    # Held-out evaluators and the training page never call a model.
    others = [c for c in _process_cmdlines()
              if "experiments.infra.evaluate" not in c and "experiments.monitor" not in c]
    if others:
        raise SystemExit(f"existing experiment clients must be inspected first: {others}")
    for item in plan:
        if Path(item["run_dir"]).exists() or is_session_alive(item["session"]):
            raise SystemExit(f"run or session already exists: {item['run_name']}")
    check_backends(list(BACKEND_NAMES))
    services = served_models()
    for name in BACKEND_NAMES:
        services[name]["assigned_runs"] = sum(p["backend"] == name for p in plan)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = {ROOT / f for f in tracked if f and (f.endswith(".py") or f in ("pyproject.toml", "uv.lock"))}
    files |= set((ROOT / "traceaad" / "v10_18").glob("*.py")) | set((ROOT / "experiments" / EXPERIMENT).glob("*.py"))
    files = sorted(p for p in files if p.exists())
    manifest = {"batch": args.batch, "method": "v1018", "protocol_revision": "V10.18",
                "experiment": EXPERIMENT, "created_at": datetime.now().astimezone().isoformat(),
                "execution_host": "local", "host": {"node": platform.node(), "cpus": os.cpu_count()},
                "status": "launching", "repeats": 3, "eval_workers": 4,
                "budget_per_run": 1000, "total_budget": 15000, "search_policy": asdict(Config()),
                "sampling": {**SAMPLING_PROFILES[False], "enable_thinking": False},
                "evaluation_limits": evaluation_limits(), "served_models": services,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
                "implementation_files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in files},
                "compare_with": "traceaad_v10_17/batch_20261003_server3_v1017.json (server3 host)",
                "load_average_at_launch": list(os.getloadavg()),
                "heldout": "separate after independent selection", "plan": plan}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, manifest)
    for item in plan:
        Path(item["startup_log"]).parent.mkdir(parents=True, exist_ok=True)
        shell = ("cd " + shlex.quote(str(ROOT)) + " && exec " + shlex.join(item["command"])
                 + " >> " + shlex.quote(item["startup_log"]) + " 2>&1")
        launch_command(item["session"], ["bash", "-c", shell])
        item["status"] = "started_unverified"
        write_json_atomic(manifest_path, manifest)
        print(item["session"], item["backend"], flush=True)
    manifest["status"] = "started_pending_verification"
    write_json_atomic(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    sys.exit(main())
