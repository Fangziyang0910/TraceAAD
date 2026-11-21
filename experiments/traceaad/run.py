"""One versioned TraceAAD search followed by evaluation of its frozen best program."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
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
import evotrace_v4
import evotrace_v5
import evotrace_v5_1
from evotrace_v4.allocation import AllocationConfig as V4AllocationConfig
from evotrace_v4.topology import TopologyConfig as V4TopologyConfig
from evotrace_v4.cli import load_task as load_evotrace_v4_task
from evotrace_v4.evaluation import Evaluator as EvoTraceV4Evaluator
from evotrace_v3.allocation import AllocationConfig as V3AllocationConfig
from evotrace_v3.topology import TopologyConfig as V3TopologyConfig
from evotrace_v3.cli import load_task as load_evotrace_v3_task
from evotrace_v3.evaluation import Evaluator as EvoTraceV3Evaluator
from evotrace_v2.cli import load_task as load_evotrace_v2_task
from evotrace_v2.evaluation import Evaluator as EvoTraceV2Evaluator
from evotrace_v1.cli import load_task as load_evotrace_task
from evotrace_v1.evaluation import Evaluator as EvoTraceEvaluator
from evotrace_v5.cli import load_task as load_evotrace_v5_task
from evotrace_v5.evaluation import Evaluator as EvoTraceV5Evaluator
from evotrace_v5_1.cli import load_task as load_evotrace_v5_1_task
from evotrace_v5_1.evaluation import Evaluator as EvoTraceV51Evaluator
import traceaad_base
from traceaad_base import Config, Evaluator, Task
from traceaad_base.cli import load_task
from traceaad_base.selection import SELECTION_RULE
from experiments.infra.storage import seal_calls
from traceaad_base.storage import write_json
from .scheduling import (
    EVALUATION_WORKERS, REQUEST_OPTIONS, ScheduledEvaluator, ScheduledModel,
    load_environment, scheduler_status,
)


ROOT = Path(__file__).resolve().parents[2]
MEMORY_BYTES = 2 << 30
ECO_BACKEND = "opendesign-eco"
BACKENDS = ("scheduled", ECO_BACKEND)
ECO_URL = "https://amr-link.open-design.ai/v1"
ECO_MODEL = "deepseek-v4.1-flash-eco"
TOPOLOGY_FIELDS = tuple(f.name for f in fields(V3TopologyConfig))
ALLOCATION_FIELDS = tuple(f.name for f in fields(V4AllocationConfig))
EVOTRACE_METHODS = {
    "evotrace_v1", "evotrace_v2", "evotrace_v3", "evotrace_v4", "evotrace_v5", "evotrace_v5_1",
}
EVOTRACE_MODULES = {
    "evotrace_v1": evotrace_v1, "evotrace_v2": evotrace_v2, "evotrace_v3": evotrace_v3,
    "evotrace_v4": evotrace_v4, "evotrace_v5": evotrace_v5, "evotrace_v5_1": evotrace_v5_1,
}
V5_FAMILY = {"evotrace_v5", "evotrace_v5_1"}
V51_PARAM_FIELDS = ("population_capacity", "selection_lambda", "reflection_mask_prob")
TREE_PARAM_METHODS = {"evotrace_v3", "evotrace_v4"}  # V5/V5.1 have their own settings; V3/V4 parameters never apply.
REFLECTION_METHODS = {"evotrace_v4", "evotrace_v5", "evotrace_v5_1"}
READS_REFERENCE_CODE = TREE_PARAM_METHODS | V5_FAMILY
METHODS = {"base": ("traceaad_base", traceaad_base.TraceAAD),
           "evotrace.v1": ("evotrace_v1", evotrace_v1.EvoTrace), "evotrace_v1": ("evotrace_v1", evotrace_v1.EvoTrace),
           "evotrace.v2": ("evotrace_v2", evotrace_v2.EvoTrace), "evotrace_v2": ("evotrace_v2", evotrace_v2.EvoTrace),
           "evotrace.v3": ("evotrace_v3", evotrace_v3.EvoTrace), "evotrace_v3": ("evotrace_v3", evotrace_v3.EvoTrace),
           "evotrace.v4": ("evotrace_v4", evotrace_v4.EvoTrace), "evotrace_v4": ("evotrace_v4", evotrace_v4.EvoTrace),
           "evotrace.v5": ("evotrace_v5", evotrace_v5.EvoTrace), "evotrace_v5": ("evotrace_v5", evotrace_v5.EvoTrace),
           "evotrace.v5.1": ("evotrace_v5_1", evotrace_v5_1.EvoTrace),
           "evotrace_v5_1": ("evotrace_v5_1", evotrace_v5_1.EvoTrace)}


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


def request_options(method, backend="scheduled"):
    module = EVOTRACE_MODULES.get(METHODS[method][0])
    schema = module.models.CANDIDATE_RESPONSE_FORMAT if module is not None else REQUEST_OPTIONS["response_format"]
    sampling = {"reasoning_effort": "none"} if backend == ECO_BACKEND else REQUEST_OPTIONS
    return {**sampling, "response_format": schema}


def eco_routing():
    return {"policy": "direct", "endpoints": [{"base_url": ECO_URL, "model": ECO_MODEL}]}


def metadata_for(args, task, config, model, routing):
    return {
        "method": METHODS[args.method][0], "experiment": args.experiment, "task": args.task,
        "run_name": args.run_name, "batch": args.batch, "backend": args.backend,
        "seed": args.seed, "repeat": args.seed + 1,
        "budget": config.evaluation_budget, "budget_axis": "评价次数", "budget_basis": "evaluate_calls",
        "method_params": asdict(config), "final_selection": "training", "eval_workers": EVALUATION_WORKERS,
        "selection_rule": selection_rule_for(args.method),
        "mechanisms": list(METHODS[args.method][1].mechanisms),
        "generation_protocol": METHODS[args.method][1].generation_protocol
                               + ("+reference_code" if args.reference_code and METHODS[args.method][0] not in EVOTRACE_METHODS else ""),
        "trajectory_reference_embedding": False,
        "reference_code": args.reference_code or METHODS[args.method][0] in READS_REFERENCE_CODE,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_snapshot": str(ROOT), "task_definition": task.description(),
        "task_eval": task.metadata["settings"],
        "evaluation_execution": {
            "timeout_scope": "whole_task", "timeout_seconds": task.time_limit,
            "time_budget": "pooled_instance_seconds", "n_workers": EVALUATION_WORKERS, "memory_limit_bytes": task.memory_limit_bytes,
            "evaluation_seed": task.evaluation_seed,
            "seconds_per_instance": args.seconds_per_instance,
        },
        "llm": {"model": model.model, "temperature": config.temperature,
                "max_tokens": config.max_tokens, "timeout": config.model_timeout,
                "enable_thinking": False,
                "sampling": {"temperature": config.temperature, **request_options(args.method, args.backend)}},
        "model_routing": routing,
        "scheduler_socket": args.scheduler_socket,
    }


def evaluate_heldout(directory, name, best, scheduler, *, evaluation_seed,
                     seconds_per_instance=None, condition="traceaad_base"):
    path = directory / "heldout.json"
    source_hash = hashlib.sha256(best.code.encode()).hexdigest()
    expected_splits = list(PRIMARY_SPLITS[name])
    if path.exists():
        previous = json.loads(path.read_text())
        if (
            [row.get("split") for row in previous] != expected_splits
            or any(
                row.get("task") != name
                or row.get("node_id") != best.id
                or row.get("source_sha256") != source_hash
                or row.get("condition") != condition
                or row.get("timeout_scope") != "whole_task"
                or row.get("seconds_per_instance") != seconds_per_instance
                or row.get("evaluation_seeds") != [evaluation_seed]
                for row in previous
            )
        ):
            raise ValueError("Existing held-out results have a different program or evaluation scope")
        return previous
    results = []
    for split in PRIMARY_SPLITS[name]:
        external = heldout_task(name, split, workers=1)
        task_type = {key: module.Task for key, module in EVOTRACE_MODULES.items()}.get(condition, Task)
        task = task_type.from_evaluator(external, use_template_example=False)
        configure_time_budget(task, seconds_per_instance)
        task.evaluation_seed = evaluation_seed
        task.memory_limit_bytes = MEMORY_BYTES
        evaluator_type = {"evotrace_v1": EvoTraceEvaluator, "evotrace_v2": EvoTraceV2Evaluator,
                          "evotrace_v3": EvoTraceV3Evaluator, "evotrace_v4": EvoTraceV4Evaluator,
                          "evotrace_v5": EvoTraceV5Evaluator,
                          "evotrace_v5_1": EvoTraceV51Evaluator}.get(condition, Evaluator)
        evaluator = (ScheduledEvaluator(task, scheduler, directory.name + ":heldout",
                                        evaluator_factory=evaluator_type)
                     if scheduler else evaluator_type(task))
        with getattr(evaluator, "evaluation_context", nullcontext)():
            started = time.monotonic()
            # The held-out score is the objective of this one evaluation, nothing more.
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
            "timeout_seconds": task.time_limit, "n_workers": getattr(evaluator, "workers", 1),
            "time_budget": "pooled_instance_seconds",
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
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", help="Result directory name; defaults to the selected method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--initial-size", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--model-timeout", type=float)
    parser.add_argument("--selection-temperature", type=float,
                        help="Base only: Boltzmann parent temperature in fitness units (default: 1.0)")
    parser.add_argument("--rank-half-life", type=float, help="V3/V4 quality weight half-life in ranks")
    parser.add_argument("--explore-references", type=int, help="V3/V4/V5.1 maximum diverse Explore overview cards")
    parser.add_argument("--max-representatives", type=int, help="V3/V4 maximum mutually uncovered core representatives")
    parser.add_argument("--cover-distance", type=float, help="V3/V4 AST shingle distance covered by a better representative")
    parser.add_argument("--region-size", type=int, help="V3/V4 minimum nodes per simplified path or subtree region")
    parser.add_argument("--population-capacity", type=int, help="V5.1 only: bounded lineage forest capacity K (default: 24)")
    parser.add_argument("--selection-lambda", type=float, help="V5.1 only: median/MAD sigmoid steepness (default: 10.0)")
    parser.add_argument("--reflection-mask-prob", type=float, help="V5.1 only: within-run Refine memory masking probability in [0, 1]")
    parser.add_argument("--reflection", action=argparse.BooleanOptionalAction, default=None,
                        help="V4/V5 only: enable or disable reflection during tree simplification (default: enabled)")
    parser.add_argument("--reference-code", action="store_true",
                        help="Crossover also shows the reference algorithm source")
    parser.add_argument("--seconds-per-instance", type=float,
                        help="Time allowance per instance; their sum is the whole-task deadline")
    parser.add_argument("--scheduler-socket", default=os.getenv("TRACEAAD_SCHEDULER_SOCKET"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    args = parser.parse_args(argv)
    args.experiment = args.experiment or METHODS[args.method][0]
    for name in (args.experiment, args.run_name, args.batch):
        if not name.replace("_", "").replace("-", "").isalnum():
            parser.error("Use alphanumeric names with underscores or hyphens")
    directory = ROOT / "experiments_result" / args.experiment / args.task / args.run_name
    defaults = dict(seed=0, budget=1000, initial_size=5, max_tokens=16384, model_timeout=600,
                    seconds_per_instance=10., backend="scheduled")
    native_v5 = None
    method_id_early = METHODS[args.method][0]
    if args.resume and method_id_early in V5_FAMILY:
        native_path = directory / "native" / "run_config.json"
        if not native_path.is_file():
            parser.error(f"No saved {method_id_early} run to resume: {directory}")
        native_v5 = json.loads(native_path.read_text())
        saved = native_v5["config"]
        defaults.update({k: saved[k] for k in ("seed", "initial_size", "max_tokens", "model_timeout")})
        defaults["budget"] = saved["evaluation_budget"]
        if method_id_early == "evotrace_v5_1":
            for k in (*V51_PARAM_FIELDS, "explore_references"):
                if k in saved:
                    defaults[k] = saved[k]
        public_path = directory / "run_config.json"
        if public_path.is_file():
            public = json.loads(public_path.read_text())
            defaults["backend"] = public.get("backend", "scheduled")
            defaults["seconds_per_instance"] = public.get("evaluation_execution", {}).get("seconds_per_instance", 10.)
        for key, value in defaults.items():
            requested = getattr(args, key)
            if key == "budget":
                if requested is not None and requested < value:
                    parser.error("Resume can only extend the evaluation budget")
            elif requested is not None and requested != value:
                parser.error(f"Resume requires the original {key.replace('_', '-')} ({value!r})")
        saved_reflection = native_v5.get("reflection", True)
        if args.reflection is not None and args.reflection != saved_reflection:
            parser.error("Resume requires the original reflection setting")
        args.reflection = saved_reflection
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    for name in (*ALLOCATION_FIELDS, *TOPOLOGY_FIELDS):
        if name == "explore_references" and method_id_early == "evotrace_v5_1":
            continue
        if getattr(args, name) is not None and method_id_early not in TREE_PARAM_METHODS:
            parser.error(f"{name.replace('_', '-')} applies to EvoTrace V3/V4 only")
    for name in V51_PARAM_FIELDS:
        if getattr(args, name) is not None and method_id_early != "evotrace_v5_1":
            parser.error(f"--{name.replace('_', '-')} applies to EvoTrace V5.1 only")
    if args.reflection is not None and method_id_early not in REFLECTION_METHODS:
        parser.error("--reflection/--no-reflection applies to EvoTrace V4/V5 only")
    if args.reference_code and method_id_early not in EVOTRACE_METHODS - V5_FAMILY:
        parser.error("--reference-code applies to EvoTrace V1-V4 only")
    if args.selection_temperature is not None and method_id_early in V5_FAMILY:
        parser.error("--selection-temperature applies to Base only")
    if args.selection_temperature is None:
        args.selection_temperature = 1.0
    if not args.scheduler_socket:
        parser.error("A resource scheduler socket is required for evaluation")
    if not math.isfinite(args.seconds_per_instance) or args.seconds_per_instance <= 0:
        parser.error("seconds-per-instance must be finite and positive")
    if not math.isfinite(args.selection_temperature) or args.selection_temperature <= 0:
        parser.error("selection-temperature must be finite and positive")
    load_environment(ROOT / ".env")
    load_environment(args.env_file)
    method_id = METHODS[args.method][0]
    allocation_config = None
    topology_config = None
    if method_id in TREE_PARAM_METHODS:
        alloc_cls = V4AllocationConfig if method_id == "evotrace_v4" else V3AllocationConfig
        topo_cls = V4TopologyConfig if method_id == "evotrace_v4" else V3TopologyConfig
        allocation_config = alloc_cls(**{
            key: getattr(args, key) for key in ALLOCATION_FIELDS if getattr(args, key) is not None})
        topology_config = topo_cls(**{
            key: getattr(args, key) for key in TOPOLOGY_FIELDS if getattr(args, key) is not None})
    evotrace = EVOTRACE_MODULES.get(method_id)
    is_evotrace = evotrace is not None
    config_type = evotrace.Config if is_evotrace else Config
    v51_extra = (
        {
            k: getattr(args, k)
            for k in (*V51_PARAM_FIELDS, "explore_references")
            if getattr(args, k) is not None
        }
        if method_id == "evotrace_v5_1"
        else {}
    )
    config = config_type(evaluation_budget=args.budget, initial_size=args.initial_size,
                         seed=args.seed, max_tokens=args.max_tokens, model_timeout=args.model_timeout,
                         **v51_extra,
                         **({} if is_evotrace else {"selection_temperature": args.selection_temperature}))
    if native_v5:
        config = config_type(**{**native_v5["config"], "evaluation_budget": args.budget})
    task_loader = {"evotrace_v1": load_evotrace_task, "evotrace_v2": load_evotrace_v2_task,
                   "evotrace_v3": load_evotrace_v3_task, "evotrace_v4": load_evotrace_v4_task,
                   "evotrace_v5": load_evotrace_v5_task,
                   "evotrace_v5_1": load_evotrace_v5_1_task}.get(method_id, load_task)
    task = task_loader(name=args.task)
    configure_time_budget(task, args.seconds_per_instance)
    task.memory_limit_bytes = MEMORY_BYTES
    if args.backend == ECO_BACKEND:
        key = os.environ.pop("OPENDESIGN_API_KEY", "")
        if not args.dry_run and not key:
            parser.error("Set OPENDESIGN_API_KEY in the environment or --env-file")
        scheduler_status(args.scheduler_socket)
        client = (evotrace or traceaad_base).OpenAIClient
        model = client(ECO_URL, ECO_MODEL, api_key=key,
                       request_options=request_options(args.method, args.backend))
        routing = eco_routing()
    else:
        model = ScheduledModel(args.scheduler_socket, args.run_name)
        if is_evotrace:
            model.factory = evotrace.OpenAIClient
        routing = {"policy": "shared", "endpoints": model.scheduler["endpoints"]}
    metadata = metadata_for(args, task, config, model, routing)
    reflection = (True if args.reflection is None else args.reflection) if method_id in REFLECTION_METHODS else None
    include_experiences = native_v5.get("include_experiences", reflection) if native_v5 else reflection
    if allocation_config is not None:
        metadata["allocation"] = asdict(allocation_config)
        metadata["topology"] = asdict(topology_config)
    if reflection is not None:
        metadata["reflection"] = reflection
        metadata["include_experiences"] = include_experiences
        metadata["mechanisms"] = list(METHODS[args.method][1].mechanisms_for(reflection))
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if directory.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {directory}; use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    if args.resume and (directory / "run_config.json").exists():
        metadata = json.loads((directory / "run_config.json").read_text())
        if metadata.get("backend", "scheduled") != args.backend:
            parser.error("Resume requires the original model backend")
    if args.resume and allocation_config is not None:
        alloc_cls = V4AllocationConfig if method_id == "evotrace_v4" else V3AllocationConfig
        topo_cls = V4TopologyConfig if method_id == "evotrace_v4" else V3TopologyConfig
        native = json.loads((directory / "native" / "run_config.json").read_text())
        allocation_config = alloc_cls(**{**native["allocation"], **{
            key: getattr(args, key) for key in ALLOCATION_FIELDS if getattr(args, key) is not None}})
        topology_config = topo_cls(**{**native["topology"], **{
            key: getattr(args, key) for key in TOPOLOGY_FIELDS if getattr(args, key) is not None}})
        if method_id == "evotrace_v4":
            saved_reflection = native.get("reflection", True)
            if args.reflection is not None and args.reflection != saved_reflection:
                parser.error("Resume requires the original reflection setting")
            reflection = saved_reflection
    if native_v5:
        previous_budget = native_v5["config"]["evaluation_budget"]
        if args.budget > previous_budget:
            metadata["budget_extended_from"] = previous_budget
        metadata["method_params"] = asdict(config)
    if allocation_config is not None:
        metadata["allocation"] = asdict(allocation_config)
        metadata["topology"] = asdict(topology_config)
    if reflection is not None:
        metadata["reflection"] = reflection
        metadata["include_experiences"] = include_experiences
        metadata["mechanisms"] = list(METHODS[args.method][1].mechanisms_for(reflection))

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)  # Closing the tmux session.
    from .reporting import Reporter

    search = METHODS[args.method][1](task, model, config=config,
                              run_dir=directory / "native", resume=args.resume,
                              **({"allocation_config": allocation_config} if allocation_config is not None else {}),
                              **({"topology_config": topology_config} if topology_config is not None else {}),
                              **({"reflection": reflection} if reflection is not None else {}),
                              **({"include_experiences": include_experiences} if method_id in V5_FAMILY else {}),
                              **({"reference_code": True}
                                 if args.reference_code and method_id not in TREE_PARAM_METHODS else {}))
    search.evaluator = ScheduledEvaluator(
        task, args.scheduler_socket, args.run_name + ":search",
        **({"evaluator_factory": {"evotrace_v1": EvoTraceEvaluator, "evotrace_v2": EvoTraceV2Evaluator,
                                "evotrace_v3": EvoTraceV3Evaluator, "evotrace_v4": EvoTraceV4Evaluator,
                                "evotrace_v5": EvoTraceV5Evaluator,
                                "evotrace_v5_1": EvoTraceV51Evaluator}[method_id]}
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
    if result.status == "finished" and result.evaluations < config.evaluation_budget:
        summary = json.loads((directory / "summary.json").read_text())
        summary.update(
            status="stopped",
            error=f"Stopped early ({result.evaluations}/{config.evaluation_budget} evaluations): {result.stop_reason}",
        )
        write_json(directory / "summary.json", summary)
        print(json.dumps({"run_dir": str(directory), **asdict(result)}, ensure_ascii=False), flush=True)
        return
    seal_calls(directory / "native")
    if result.best is not None and not args.skip_heldout:
        evaluate_heldout(directory, args.task, result.best, args.scheduler_socket,
                         evaluation_seed=task.evaluation_seed,
                         seconds_per_instance=args.seconds_per_instance,
                         condition=search.method)
    print(json.dumps({"run_dir": str(directory), **asdict(result)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
