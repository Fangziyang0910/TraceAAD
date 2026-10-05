"""Prepare and launch five-task × four-repeat V10.18 runs on prior backend routes."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from experiments.infra.base import REPO_ROOT, RESULTS_ROOT, SAMPLING_PROFILES, TASK_SHORT, build_task
from experiments.infra.launcher import check_backends, is_session_alive, launch_command, write_json_atomic
from traceaad.v10_18 import Config

from .heldout import HELDOUT_TIMEOUT
from .run import TRAIN_TIMEOUT, selection_task


def evaluation_limits():
    """Training, selection and held-out wall-clock limits actually in force."""
    limits = {}
    for task in TASK_SHORT:
        search, _ = build_task(task, 4)
        search.timeout_seconds = TRAIN_TIMEOUT.get(task, search.timeout_seconds)
        limits[task] = {"search": search.timeout_seconds,
                        "selection": selection_task(task, search).timeout_seconds,
                        "heldout": HELDOUT_TIMEOUT[task]}
    return limits


def served_models(backends):
    """What each backend is serving right now (model id, weights, context)."""
    import urllib.request

    from experiments.infra.base import BACKENDS, resolve_llm_api_key

    found = {}
    for name in sorted(set(backends)):
        profile = BACKENDS[name]
        key = resolve_llm_api_key(base_url=profile.base_url)
        headers = {"Authorization": f"Bearer {key}"} if key and key != "EMPTY" else {}
        record = {"endpoint": profile.base_url}
        try:
            request = urllib.request.Request(profile.base_url + "/models", headers=headers)
            with urllib.request.urlopen(request, timeout=20) as response:
                model = json.loads(response.read())["data"][0]
            record.update({k: model.get(k) for k in ("id", "root", "max_model_len", "owned_by")})
            if model.get("owned_by") == "llamacpp":
                with urllib.request.urlopen(profile.base_url[:-3] + "/props", timeout=20) as response:
                    props = json.loads(response.read())
                record.update(model_path=props.get("model_path"), build=props.get("build_info"),
                              slots=props.get("total_slots"))
        except Exception as exc:  # recorded, not fatal: check_backends decides reachability
            record["error"] = f"{type(exc).__name__}: {exc}"
        found[name] = record
    return found


def build_plan(previous, batch, experiment="traceaad_v10_18"):
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
        name = f"{batch}_{short}_{experiment}_rep{repeat}"
        tag = experiment.removeprefix("traceaad_").replace("v10_18", "v1018")
        session = f"{batch}_{short}_{tag}_r{repeat}"
        command = ["uv", "run", "python", "-m", "experiments.traceaad_v10_18.run",
                   "--task", task, "--backend", backend, "--repeat", str(repeat),
                   "--seed", str(seed), "--run-name", name, "--budget=1000",
                   "--eval-workers=4", "--experiment", experiment]
        plan.append({"task": task, "repeat": repeat, "seed": seed, "backend": backend,
                     "session": session, "run_name": name,
                     "run_dir": str(RESULTS_ROOT / experiment / task / name),
                     "command": command, "status": "planned"})
    if len({item["run_name"] for item in plan}) != 20:
        raise ValueError("duplicate run identities")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-batch", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", default="traceaad_v10_18",
                        help="results directory under experiments_result")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.batch.replace("_", "").replace("-", "").isalnum():
        raise ValueError("batch must be alphanumeric")
    previous = json.loads(args.from_batch.read_text(encoding="utf-8"))
    if not args.experiment.replace("_", "").isalnum():
        raise ValueError("experiment must be alphanumeric with underscores")
    plan = build_plan(previous, args.batch, args.experiment)
    sampling = {**SAMPLING_PROFILES[False], "enable_thinking": False}
    if args.dry_run:
        print(json.dumps({"batch": args.batch, "experiment": args.experiment, "plan": plan,
                          "policy": asdict(Config()), "sampling": sampling,
                          "evaluation_limits": evaluation_limits()}, indent=2))
        return
    root = RESULTS_ROOT / args.experiment
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
    files = sorted([*(REPO_ROOT / "traceaad/v10_18").glob("*.py"),
                    *(REPO_ROOT / "experiments/traceaad_v10_18").glob("*.py"),
                    REPO_ROOT / "core/llm.py", REPO_ROOT / "core/evaluate.py",
                    REPO_ROOT / "traceaad/v10_13/storage.py",
                    REPO_ROOT / "traceaad/v10_13/parsing.py"])
    manifest = {"batch": args.batch, "method": "v1018", "created_at": datetime.now().astimezone().isoformat(),
                "status": "launching", "previous_batch": str(args.from_batch),
                "experiment": args.experiment, "search_policy": asdict(Config()),
                # Requests carry every sampling control, so servers' own defaults do not apply.
                "sampling": sampling, "evaluation_limits": evaluation_limits(),
                "services": previous.get("services"),
                "served_models": served_models(item["backend"] for item in plan),
                "repeats": 4, "eval_workers": 4,
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
