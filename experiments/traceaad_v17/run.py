"""Search on training data, freeze its best program, then evaluate held-out separately."""

import argparse
from dataclasses import asdict
import json
from traceaad.v17.limits import FUNCTION_SECONDS, FUNCTION_LIMITS, INSTANCE_LIMITS, INSTANCE_SECONDS
from benchmarks.tasks import training_task
from .base import HOSTED_BACKENDS, RESULTS_ROOT
from .runner import add_common_run_args, setup_experiment_run
from .storage import read_json


def build_parser(experiment="traceaad_v17", config_type=None):
    if config_type is None:
        from traceaad.v17.config import Config as config_type
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=config_type.output_tokens)
    parser.add_argument("--scheduler-socket", help="shared CPU/GPU scheduler on this host")
    parser.add_argument(
        "--eval-timeout-seconds",
        type=float,
        default=None,
        help="wall-clock limit for each instance, including the fixed solver",
    )
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=list(config_type.evaluation_seeds)
    )
    parser.add_argument(
        "--experiment", default=experiment, help="results directory under experiments_result"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _check_resume_configuration(run_dir, method, config):
    previous = read_json(run_dir / "run_config.json")
    params = json.loads(json.dumps(asdict(config)))
    if previous and (
        previous.get("method") != method
        or any(
            (previous.get("method_params", {}).get(key) != value for key, value in params.items())
        )
    ):
        raise ValueError("resume configuration differs from the stored run")


def main(argv=None):
    extra = {}
    from traceaad.v17 import Config as config_type, TraceAADV17 as method_type
    from . import options

    experiment = "traceaad_v17"
    parser = build_parser(experiment, config_type)
    options.add_arguments(parser)
    args = parser.parse_args(argv)
    extra = options.values(args)
    if args.repeat is not None and args.seed != args.repeat - 1:
        raise ValueError("run seed must equal repeat - 1")
    if args.eval_timeout_seconds is None:
        args.eval_timeout_seconds = INSTANCE_LIMITS.get(args.task, INSTANCE_SECONDS)
    execution = {
        "eval_timeout_seconds": args.eval_timeout_seconds,
        "eval_workers": args.eval_workers,
    }
    scheduler, llm_factory, profile = (None, None, None)
    if getattr(args, "scheduler_socket", None):
        from pathlib import Path
        from traceaad.v17.runtime.scheduling import scheduler_status
        from .scheduled_llm import ScheduledLLM

        args.scheduler_socket = str(Path(args.scheduler_socket).expanduser().resolve())
        scheduler = scheduler_status(args.scheduler_socket)
        execution["scheduler_socket"] = args.scheduler_socket
        if args.backend not in HOSTED_BACKENDS:
            llm_factory = lambda **kw: ScheduledLLM(
                args.scheduler_socket,
                max_tokens=kw["max_tokens"],
                label=args.run_name or args.task,
                enable_thinking=kw["enable_thinking"],
            )
    if execution["eval_workers"] is None:
        execution["eval_workers"] = scheduler["cpu"]["capacity"] if scheduler else 1
        args.eval_workers = execution["eval_workers"]
    config = config_type(
        budget=args.budget,
        output_tokens=args.output_tokens,
        evaluation_seeds=tuple(args.evaluation_seeds),
        seed=args.seed,
        **execution,
        **extra,
    )
    metadata = {
        "evaluation_execution": {
            "timeout_scope": "instance",
            "function_budget_scope": "mean_instance",
            "timeout_seconds": config.eval_timeout_seconds,
            "function_seconds": FUNCTION_LIMITS.get(args.task, FUNCTION_SECONDS),
            "n_workers": config.eval_workers,
        }
    }
    if scheduler and llm_factory:
        metadata["model_routing"] = {"policy": "shared", "endpoints": scheduler["endpoints"]}
    if args.dry_run:
        evaluation, task_config = training_task(args.task, args.eval_workers, condition="traceaad")
        print(
            json.dumps(
                {
                    "method": method_type.METHOD,
                    "task": args.task,
                    "config": asdict(config),
                    "task_eval": task_config,
                    "search_timeout": config.eval_timeout_seconds,
                    **metadata,
                    "final_selection": "training",
                    "test": "separate held-out evaluation of the frozen program",
                },
                indent=2,
            )
        )
        return
    if not args.experiment.replace("_", "").isalnum():
        raise ValueError("experiment must be alphanumeric with underscores")
    root = RESULTS_ROOT / args.experiment
    if args.run_name:
        existing = root / args.task / args.run_name
        if (
            existing.is_dir()
            and any(existing.iterdir())
            and (not (existing / "events.jsonl").exists())
        ):
            raise ValueError(f"refusing to overwrite a non-resumable run directory: {existing}")
    if args.run_name:
        _check_resume_configuration(root / args.task / args.run_name, method_type.METHOD, config)
    ctx = setup_experiment_run(
        args,
        method=method_type.METHOD,
        results_root=root,
        resume_file="events.jsonl",
        method_params=asdict(config),
        condition="traceaad",
        extra_config={"budget_axis": "候选尝试", "final_selection": "training", **metadata},
        llm_factory=llm_factory,
        backend_profile=profile,
        budget_basis="Program-generation replies including initialization, parse failures and repair; every parsed program is a distinct individual.",
    )
    try:
        method = method_type(
            evaluation=ctx.evaluation,
            llm=ctx.llm,
            run_dir=ctx.run_dir,
            config=config,
            task=args.task,
        )
        ctx.run(method.run, [method_type.METHOD])
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()
