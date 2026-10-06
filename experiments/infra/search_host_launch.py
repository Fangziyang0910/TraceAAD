"""Launch a five-task, three-repeat batch on the current host."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import urllib.request

from experiments.infra.base import BACKENDS, SAMPLING_PROFILES, TASKS
from experiments.infra.env import resolve_llm_api_key
from experiments.infra.launcher import check_backends, is_session_alive, launch_command, write_json_atomic
from experiments.infra.search_launch import build_plan, evaluation_limits, implementation_files
from traceaad.common.config import REVISION

ROOT = Path(__file__).resolve().parents[2]


def plan_for(batch, experiment, module, backends=("server3", "server3b"), repeats=3):
    source = {"plan": [{"task": task, "repeat": repeat, "seed": repeat - 1,
                        "backend": backends[0] if (i + repeat) % 2 else backends[1]}
                       for i, task in enumerate(TASKS) for repeat in range(1, repeats + 1)]}
    plan = build_plan(source, batch, experiment, module)
    for item in plan:
        item["command"][:3] = [str(ROOT / ".venv/bin/python")]
        item["startup_log"] = str(ROOT / "experiments_result" / experiment / "launch_logs" / (item["session"] + ".log"))
    return plan


def served_models(backends):
    services = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for name in backends:
        profile = BACKENDS[name]
        request = urllib.request.Request(profile.base_url + "/models", headers={
            "Authorization": "Bearer " + resolve_llm_api_key(base_url=profile.base_url)})
        with opener.open(request, timeout=15) as response:
            model = next(m for m in json.load(response)["data"] if m["id"] == profile.model)
        if model["max_model_len"] < 32768:
            raise RuntimeError(f"{name} serves a context shorter than 32K")
        services[name] = {"endpoint": profile.base_url, "model": model}
    return services


def main(config_class, experiment, module, method, host="local", backends=("server3", "server3b"), argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_for(args.batch, experiment, module, backends)
    manifest_path = ROOT / "experiments_result" / experiment / f"batch_{args.batch}.json"
    if not args.launch:
        print(json.dumps({"manifest": str(manifest_path), "plan": plan,
                          "evaluation_limits": evaluation_limits()}, indent=2))
        return
    if manifest_path.exists():
        raise SystemExit("batch already exists")
    for item in plan:
        if Path(item["run_dir"]).exists() or is_session_alive(item["session"]):
            raise SystemExit(f"run or session already exists: {item['run_name']}")
    check_backends(list(backends))
    services = served_models(backends)
    for name in backends:
        services[name]["assigned_runs"] = sum(p["backend"] == name for p in plan)
    hashes = implementation_files(module)
    manifest = {"batch": args.batch, "method": method, "revision": REVISION,
                "experiment": experiment, "created_at": datetime.now().astimezone().isoformat(),
                "execution_host": host, "host": {"node": platform.node(), "cpus": os.cpu_count()},
                "status": "launching", "repeats": 3, "eval_workers": 4,
                "budget_per_run": 1000, "total_budget": 15000, "search_policy": asdict(config_class()),
                "sampling": {**SAMPLING_PROFILES[False], "enable_thinking": False},
                "evaluation_limits": evaluation_limits(), "served_models": services,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
                "implementation_files": hashes,
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
