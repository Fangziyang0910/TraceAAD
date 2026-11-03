"""Run V19.4 and test its frozen training winner under a whole-task deadline."""

import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

from benchmarks.tasks import ALL_TASKS, PRIMARY_SPLITS, heldout_task, scale_of_split
from traceaad_v19_4 import Config, Evaluator, OpenAIClient, Task, TraceAAD
from traceaad_v19_4.cli import SAMPLING, add_config_arguments, config_from_args, load_task
from traceaad_v19_4.search import METHOD, SCHEMA
from traceaad_v19_4.storage import write_json
from .scheduling import ScheduledEvaluator, ScheduledModel, load_environment


ROOT = Path(__file__).resolve().parents[2]
MEMORY_BYTES = 2 << 30


def configure_time_budget(task, seconds_per_instance):
    if not math.isfinite(seconds_per_instance) or seconds_per_instance <= 0:
        raise ValueError("seconds-per-instance must be finite and positive")
    count = task.metadata.get("n_instance")
    if type(count) is not int or count < 1:
        raise ValueError("A per-instance allowance needs the task instance count")
    task.time_limit = count * seconds_per_instance
    task.memory_limit_bytes = MEMORY_BYTES
    task.metadata["seconds_per_instance"] = seconds_per_instance
    if "settings" in task.metadata:
        task.metadata["settings"] = {**task.metadata["settings"], "timeout_seconds": task.time_limit}


def evaluator_for(task, scheduler, label):
    return ScheduledEvaluator(task, scheduler, label) if scheduler else Evaluator(task)


def metadata_for(args, task, config, model=None):
    return {"method": METHOD, "checkpoint_schema": SCHEMA,
        "experiment": args.experiment, "task": args.task, "run_name": args.run_name,
        "batch": args.batch, "seed": config.seed, "repeat": config.seed + 1,
        "budget": config.evaluation_budget, "budget_axis": "评价次数", "budget_basis": "evaluate_calls",
        "method_params": asdict(config), "final_selection": "training", "eval_workers": 1,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_snapshot": str(ROOT), "task_definition": task.description(),
        "task_eval": task.metadata.get("settings", {}),
        "evaluation_execution": {"timeout_scope": "whole_task", "timeout_seconds": task.time_limit,
            "n_workers": 1, "memory_limit_bytes": task.memory_limit_bytes,
            "evaluation_seed": task.evaluation_seed, "seconds_per_instance": args.seconds_per_instance},
        "llm": {"model": getattr(model, "model", args.model or "scheduled"),
            "temperature": config.temperature, "max_tokens": config.max_tokens,
            "timeout": config.model_timeout,
            "enable_thinking": False},
        "model_routing": {"policy": "direct" if args.base_url else "scheduled",
            "endpoint": args.base_url, "scheduler_socket": args.scheduler_socket},
        "auxiliary_cost": "Reflection calls and tokens are reported separately and in total model cost."}


def evaluate_heldout(directory, name, best, scheduler, *, evaluation_seed, seconds_per_instance=10.):
    path = directory / "heldout.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(r["node_id"] != best.id or r["source_sha256"] != hashlib.sha256(best.code.encode()).hexdigest()
               or r.get("timeout_scope") != "whole_task" for r in previous):
            raise ValueError("Existing held-out results refer to different code or evaluation scope")
        return previous
    results = []
    from contextlib import nullcontext
    for split in PRIMARY_SPLITS[name]:
        external = heldout_task(name, split, workers=1)
        task = Task.from_evaluator(external, use_template_example=False)
        configure_time_budget(task, seconds_per_instance)
        task.evaluation_seed = evaluation_seed
        evaluator = evaluator_for(task, scheduler, directory.name + ":heldout")
        with getattr(evaluator, "evaluation_context", nullcontext)():
            start = time.monotonic()
            evaluation = evaluator.evaluate(best.code)
        valid = evaluation.status == "success"
        results.append({"task": name, "split": split, "scale": str(scale_of_split(name, split)),
            "variant": "", "node_id": best.id, "verification": "verified",
            "source_sha256": hashlib.sha256(best.code.encode()).hexdigest(),
            "fitness": evaluation.fitness, "objective": evaluation.fitness,
            "status": evaluation.status, "failure": evaluation.failure, "seconds": time.monotonic() - start,
            "score_metric": "raw_objective", "search_status": "finished", "condition": METHOD,
            "timeout_scope": "whole_task", "timeout_seconds": task.time_limit, "n_workers": 1,
            "memory_limit_bytes": task.memory_limit_bytes, "evaluation_seeds": [evaluation_seed],
            "evaluations": [{"seed": evaluation_seed, "valid": valid, "score": evaluation.fitness,
                "failure_kind": None if valid else evaluation.status, "error": evaluation.failure}]})
    write_json(path, results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", default=METHOD)
    parser.add_argument("--seconds-per-instance", type=float, default=10.)
    parser.add_argument("--scheduler-socket", default=os.getenv("TRACEAAD_SCHEDULER_SOCKET"))
    parser.add_argument("--base-url", default=os.getenv("TRACEAAD_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("TRACEAAD_MODEL"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--request-options", default=json.dumps(SAMPLING))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    for name in (args.experiment, args.run_name, args.batch):
        if not name.replace("_", "").replace("-", "").isalnum():
            parser.error("Use alphanumeric names with underscores or hyphens")
    load_environment(args.env_file)
    directory = ROOT / "experiments_result" / args.experiment / args.task / args.run_name
    config = config_from_args(args, directory / "native")
    task = load_task(name=args.task)
    configure_time_budget(task, args.seconds_per_instance)
    if args.dry_run:
        print(json.dumps(metadata_for(args, task, config), ensure_ascii=False, indent=2))
        return
    if args.base_url:
        if not args.model:
            parser.error("A direct endpoint needs --model")
        model = OpenAIClient(args.base_url, args.model, api_key=os.getenv(args.api_key_env, ""),
                            request_options=json.loads(args.request_options))
    elif args.scheduler_socket:
        model = ScheduledModel(args.scheduler_socket, args.run_name)
    else:
        parser.error("Provide --base-url and --model, or --scheduler-socket")
    metadata = metadata_for(args, task, config, model)
    if directory.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {directory}; use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    with (directory / ".experiment.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        from .reporting import Reporter
        evaluator = evaluator_for(task, args.scheduler_socket, args.run_name + ":search")
        with TraceAAD(task, model, config=config, run_dir=directory / "native", resume=args.resume,
                      evaluator=evaluator) as search:
            if args.resume and (directory / "run_config.json").exists():
                metadata = json.loads((directory / "run_config.json").read_text())
            reporter = Reporter(directory, metadata)
            reporter.publish(search)
            try:
                result = search.run(on_attempt=lambda _: reporter.publish(search))
                test_pending = bool(result.best and not args.skip_heldout)
                reporter.publish(search, result, heldout_pending=test_pending)
            except BaseException as exc:
                summary = reporter.publish(search)
                write_json(directory / "summary.json", {**summary, "status": "stopped",
                    "error": f"{type(exc).__name__}: {exc}"})
                raise
            if test_pending:
                try:
                    evaluate_heldout(directory, args.task, result.best, args.scheduler_socket,
                        evaluation_seed=task.evaluation_seed, seconds_per_instance=args.seconds_per_instance)
                except BaseException as exc:
                    summary = json.loads((directory / "summary.json").read_text())
                    write_json(directory / "summary.json", {**summary, "status": "stopped",
                        "phase": "heldout_failed", "error": f"{type(exc).__name__}: {exc}"})
                    raise
                reporter.publish(search, result)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
