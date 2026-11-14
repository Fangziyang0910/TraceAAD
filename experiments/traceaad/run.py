"""One versioned TraceAAD search followed by evaluation of its frozen best program."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from benchmarks.tasks import ALL_TASKS, PRIMARY_SPLITS, heldout_task, scale_of_split
import evotrace_v1
import evotrace_v2
import evotrace_v3
from evotrace_v3.cli import load_task as load_evotrace_v3_task
from evotrace_v3.evaluation import Evaluator as EvoTraceV3Evaluator
from evotrace_v2.cli import load_task as load_evotrace_v2_task
from evotrace_v2.evaluation import Evaluator as EvoTraceV2Evaluator
from evotrace_v1.cli import load_task as load_evotrace_task
from evotrace_v1.evaluation import Evaluator as EvoTraceEvaluator
import traceaad_base
from traceaad_base import Config, Evaluator, Task
from traceaad_base.cli import load_task
from traceaad_base.selection import SELECTION_RULE
from experiments.infra.storage import seal_calls
from traceaad_base.storage import write_json
import traceaad_v21_3
import traceaad_v22_2
import traceaad_v23_2
import traceaad_v24_1
from traceaad_v21_3.prompts import CANDIDATE_RESPONSE_FORMAT as V21_RESPONSE_FORMAT
from .scheduling import REQUEST_OPTIONS, ScheduledEvaluator, ScheduledModel, load_environment


ROOT = Path(__file__).resolve().parents[2]
MEMORY_BYTES = 2 << 30
METHODS = {"base": ("traceaad_base", traceaad_base.TraceAAD),
           "evotrace.v1": ("evotrace_v1", evotrace_v1.EvoTrace), "evotrace_v1": ("evotrace_v1", evotrace_v1.EvoTrace),
           "evotrace.v2": ("evotrace_v2", evotrace_v2.EvoTrace), "evotrace_v2": ("evotrace_v2", evotrace_v2.EvoTrace),
           "evotrace.v3": ("evotrace_v3", evotrace_v3.EvoTrace), "evotrace_v3": ("evotrace_v3", evotrace_v3.EvoTrace),
           "v21.3": ("traceaad_v21_3", traceaad_v21_3.TraceAAD), "v21_3": ("traceaad_v21_3", traceaad_v21_3.TraceAAD),
           "v22.2": ("traceaad_v22_2", traceaad_v22_2.TraceAAD), "v22_2": ("traceaad_v22_2", traceaad_v22_2.TraceAAD),
           "v23.2": ("traceaad_v23_2", traceaad_v23_2.TraceAAD), "v23_2": ("traceaad_v23_2", traceaad_v23_2.TraceAAD),
           "v24.1": ("traceaad_v24_1", traceaad_v24_1.TraceAAD), "v24_1": ("traceaad_v24_1", traceaad_v24_1.TraceAAD)}


def configure_time_budget(task, seconds_per_instance):
    if seconds_per_instance is None:
        return
    if not math.isfinite(seconds_per_instance) or seconds_per_instance <= 0:
        raise ValueError("seconds-per-instance must be finite and positive")
    count = task.metadata.get("n_instance")
    if type(count) is not int or count < 1:
        raise ValueError("A per-instance time allowance needs the task instance count")
    task.time_limit = seconds_per_instance * count
    task.metadata["seconds_per_instance"] = seconds_per_instance
    if "settings" in task.metadata:
        task.metadata["settings"] = {**task.metadata["settings"],
                                     "timeout_seconds": task.time_limit}


def selection_rule_for(method):
    return getattr(METHODS[method][1], "selection_rule", SELECTION_RULE)


def request_options(method):
    mechs = METHODS[method][1].mechanisms
    if 2 in mechs or "multiscale_context" in mechs:
        return {**REQUEST_OPTIONS, "response_format": V21_RESPONSE_FORMAT}
    return REQUEST_OPTIONS


def reference_embedding_enabled(method, requested):
    """Mechanism 1 includes trajectory references unless explicitly ablated."""
    has_allocation = 1 in METHODS[method][1].mechanisms
    enabled = has_allocation if requested is None else requested
    if enabled and not has_allocation:
        raise ValueError("trajectory-reference-embedding requires a method with mechanism 1")
    return enabled


def metadata_for(args, task, config, model):
    return {
        "method": METHODS[args.method][0], "experiment": args.experiment, "task": args.task,
        "run_name": args.run_name, "batch": args.batch,
        "seed": args.seed, "repeat": args.seed + 1,
        "budget": config.evaluation_budget, "budget_axis": "评价次数", "budget_basis": "evaluate_calls",
        "method_params": asdict(config), "final_selection": "training", "eval_workers": 1,
        "selection_rule": selection_rule_for(args.method),
        "mechanisms": list(METHODS[args.method][1].mechanisms),
        "generation_protocol": METHODS[args.method][1].generation_protocol
                               + ("+reference_code" if args.reference_code and METHODS[args.method][0] not in {"evotrace_v1", "evotrace_v2", "evotrace_v3"} else ""),
        "trajectory_reference_embedding": args.trajectory_reference_embedding,
        "reference_code": args.reference_code or METHODS[args.method][0] == "evotrace_v3",
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_snapshot": str(ROOT), "task_definition": task.description(),
        "task_eval": task.metadata["settings"],
        "evaluation_execution": {
            "timeout_scope": "whole_task", "timeout_seconds": task.time_limit,
            "n_workers": 1, "memory_limit_bytes": task.memory_limit_bytes,
            "evaluation_seed": task.evaluation_seed,
            "seconds_per_instance": args.seconds_per_instance,
        },
        "llm": {"model": model.model, "temperature": config.temperature,
                "max_tokens": config.max_tokens, "timeout": config.model_timeout,
                "enable_thinking": False, "sampling": {"temperature": config.temperature, **request_options(args.method)}},
        "model_routing": {"policy": "shared", "endpoints": model.scheduler["endpoints"]},
        "scheduler_socket": model.path,
    }


def evaluate_heldout(directory, name, best, scheduler, *, evaluation_seed,
                     seconds_per_instance=None, condition="traceaad_base"):
    path = directory / "heldout.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(row["node_id"] != best.id or row.get("timeout_scope") != "whole_task" for row in previous):
            raise ValueError("Existing held-out results have a different program or evaluation scope")
        return previous
    results = []
    for split in PRIMARY_SPLITS[name]:
        external = heldout_task(name, split, workers=1)
        task_type = {"evotrace_v1": evotrace_v1.Task, "evotrace_v2": evotrace_v2.Task, "evotrace_v3": evotrace_v3.Task}.get(condition, Task)
        task = task_type.from_evaluator(external, use_template_example=False)
        configure_time_budget(task, seconds_per_instance)
        task.evaluation_seed = evaluation_seed
        task.memory_limit_bytes = MEMORY_BYTES
        evaluator_type = {"evotrace_v1": EvoTraceEvaluator, "evotrace_v2": EvoTraceV2Evaluator, "evotrace_v3": EvoTraceV3Evaluator}.get(condition, Evaluator)
        evaluator = (ScheduledEvaluator(task, scheduler, directory.name + ":heldout",
                                        evaluator_factory=evaluator_type)
                     if scheduler else evaluator_type(task))
        with getattr(evaluator, "evaluation_context", nullcontext)():
            started = time.monotonic()
            evaluation = evaluator.evaluate(best.code)
            seconds = time.monotonic() - started
        valid = evaluation.status == "success"
        results.append({
            "task": name, "split": split, "scale": str(scale_of_split(name, split)),
            "variant": "", "node_id": best.id, "verification": "verified",
            "source_sha256": hashlib.sha256(best.code.encode()).hexdigest(),
            "fitness": evaluation.fitness, "objective": evaluation.fitness,
            "status": evaluation.status, "failure": evaluation.failure,
            "seconds": seconds, "score_metric": "raw_objective", "search_status": "finished",
            "condition": condition, "timeout_scope": "whole_task",
            "timeout_seconds": task.time_limit, "n_workers": 1,
            "seconds_per_instance": seconds_per_instance,
            "memory_limit_bytes": task.memory_limit_bytes, "evaluation_seeds": [evaluation_seed],
            "resources": getattr(evaluator, "last_dispatch", {}),
            "evaluations": [{"seed": evaluation_seed, "valid": valid,
                             "score": evaluation.fitness, "seconds": seconds,
                             "failure_kind": None if valid else evaluation.status,
                             "error": evaluation.failure}],
            "failures": [] if valid else [{"kind": evaluation.status, "error": evaluation.failure}],
        })
    write_json(path, results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), default="base")
    parser.add_argument("--backend", choices=("scheduled",), default="scheduled")
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", help="Result directory name; defaults to the selected method")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--initial-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--model-timeout", type=float, default=600)
    parser.add_argument("--selection-temperature", type=float, default=1.0,
                        help="Boltzmann parent temperature in fitness units")
    parser.add_argument("--trajectory-reference-embedding", action=argparse.BooleanOptionalAction,
                        default=None, help="Mechanism 1 uses trajectory references by default; --no-... ablates them")
    parser.add_argument("--reference-code", action="store_true",
                        help="Crossover also shows the reference algorithm source")
    parser.add_argument("--seconds-per-instance", type=float, default=10.,
                        help="Time allowance per instance; their sum is the whole-task deadline")
    parser.add_argument("--scheduler-socket", default=os.getenv("TRACEAAD_SCHEDULER_SOCKET"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    args = parser.parse_args(argv)
    args.experiment = args.experiment or METHODS[args.method][0]
    try:
        args.trajectory_reference_embedding = reference_embedding_enabled(args.method, args.trajectory_reference_embedding)
    except ValueError as error:
        parser.error(str(error))
    if args.reference_code and METHODS[args.method][0] not in {"traceaad_v21_3", "evotrace_v1", "evotrace_v2", "evotrace_v3"}:
        parser.error("--reference-code applies to v21.3 or EvoTrace only")
    if not args.scheduler_socket and args.backend == "scheduled":
        parser.error("A resource scheduler socket is required")
    if not math.isfinite(args.seconds_per_instance) or args.seconds_per_instance <= 0:
        parser.error("seconds-per-instance must be finite and positive")
    if not math.isfinite(args.selection_temperature) or args.selection_temperature <= 0:
        parser.error("selection-temperature must be finite and positive")
    for name in (args.experiment, args.run_name, args.batch):
        if not name.replace("_", "").replace("-", "").isalnum():
            parser.error("Use alphanumeric names with underscores or hyphens")
    load_environment(ROOT / ".env")
    load_environment(args.env_file)
    method_id = METHODS[args.method][0]
    evotrace = {"evotrace_v1": evotrace_v1, "evotrace_v2": evotrace_v2, "evotrace_v3": evotrace_v3}.get(method_id)
    is_evotrace = evotrace is not None
    config_type = evotrace.Config if is_evotrace else Config
    config = config_type(evaluation_budget=args.budget, initial_size=args.initial_size,
                         seed=args.seed, max_tokens=args.max_tokens, model_timeout=args.model_timeout,
                         **({} if is_evotrace else {"selection_temperature": args.selection_temperature}))
    task_loader = {"evotrace_v1": load_evotrace_task, "evotrace_v2": load_evotrace_v2_task, "evotrace_v3": load_evotrace_v3_task}.get(method_id, load_task)
    task = task_loader(name=args.task)
    configure_time_budget(task, args.seconds_per_instance)
    task.memory_limit_bytes = MEMORY_BYTES
    model = ScheduledModel(args.scheduler_socket, args.run_name)
    if is_evotrace:
        model.factory = evotrace.OpenAIClient
    metadata = metadata_for(args, task, config, model)
    reference_embedding = None
    if args.trajectory_reference_embedding:
        from experiments.deltaevolve.embedding import CodeEmbedding
        from traceaad_v22_2.references import settings_for

        reference_embedding = CodeEmbedding()
        metadata["reference_settings"] = settings_for(reference_embedding)
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    directory = ROOT / "experiments_result" / args.experiment / args.task / args.run_name
    if directory.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {directory}; use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    if args.resume and (directory / "run_config.json").exists():
        metadata = json.loads((directory / "run_config.json").read_text())

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)  # Closing the tmux session.
    from .reporting import Reporter

    search = METHODS[args.method][1](task, model, config=config,
                              run_dir=directory / "native", resume=args.resume,
                              **({"reference_embedding": reference_embedding}
                                 if reference_embedding is not None else {}),
                              **({"reference_code": True} if args.reference_code else {}))
    search.evaluator = ScheduledEvaluator(
        task, args.scheduler_socket, args.run_name + ":search",
        **({"evaluator_factory": {"evotrace_v1": EvoTraceEvaluator, "evotrace_v2": EvoTraceV2Evaluator,
                                "evotrace_v3": EvoTraceV3Evaluator}[method_id]}
           if is_evotrace else {}),
    )
    reporter = Reporter(directory, metadata)
    reporter.publish(search)

    def progress(attempt):
        reporter.publish(search)
        best = search.best.evaluation.fitness if search.best else None
        print(json.dumps({"attempt": attempt.id, "operator": attempt.operator,
                          "evaluations": search.evaluations_used,
                          "model_calls": search.model_calls, "best": best}), flush=True)

    try:
        result = search.run(on_attempt=progress)
        reporter.publish(search, result=result)
    except BaseException as exc:
        reporter.publish(search)
        summary = json.loads((directory / "summary.json").read_text())
        summary.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "summary.json", summary)
        raise
    finally:
        if reference_embedding is not None:
            reference_embedding.close()
    seal_calls(directory / "native")
    if result.best is not None and not args.skip_heldout:
        evaluate_heldout(directory, args.task, result.best, args.scheduler_socket,
                         evaluation_seed=task.evaluation_seed,
                         seconds_per_instance=args.seconds_per_instance,
                         condition=search.method)
    print(json.dumps({"run_dir": str(directory), **asdict(result)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
