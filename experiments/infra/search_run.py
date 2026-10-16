"""Search on training data, freeze its best program, then evaluate held-out separately."""

import argparse
from dataclasses import asdict
import json

from benchmarks.tasks import (
    FUNCTION_SECONDS,
    FUNCTION_LIMITS,
    INSTANCE_LIMITS,
    INSTANCE_SECONDS,
    training_task,
)
from experiments.infra.base import HOSTED_BACKENDS, RESULTS_ROOT
from experiments.infra.runner import add_common_run_args, setup_experiment_run
from traceaad.config import Config, EXPERIMENT
from traceaad import TraceAAD
from .diagnose_search import diagnose


def build_parser(experiment=EXPERIMENT):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=Config.output_tokens)
    parser.add_argument("--scheduler-socket", help="shared CPU/GPU scheduler on this host")
    parser.add_argument(
        "--eval-timeout-seconds",
        type=float,
        default=None,
        help="wall-clock limit for each instance, including the fixed solver",
    )
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=list(Config.evaluation_seeds)
    )
    parser.add_argument(
        "--experiment", default=experiment, help="results directory under experiments_result"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.repeat is not None and args.seed != args.repeat - 1:
        raise ValueError("run seed must equal repeat - 1")
    if args.eval_timeout_seconds is None:
        args.eval_timeout_seconds = INSTANCE_LIMITS.get(args.task, INSTANCE_SECONDS)
    execution = {
        "eval_timeout_seconds": args.eval_timeout_seconds,
        "eval_workers": args.eval_workers,
    }
    scheduler, llm_factory, profile = None, None, None
    if getattr(args, "scheduler_socket", None):
        from pathlib import Path
        from core.scheduling import scheduler_status
        from .base import BackendProfile
        from .scheduled_llm import ScheduledLLM

        args.scheduler_socket = str(Path(args.scheduler_socket).expanduser().resolve())
        scheduler = scheduler_status(args.scheduler_socket)
        execution["scheduler_socket"] = args.scheduler_socket
        if args.backend not in HOSTED_BACKENDS:
            endpoint = scheduler["endpoints"][0]
            profile = BackendProfile(
                endpoint["base_url"], endpoint["model"], endpoint.get("no_proxy", "")
            )
            llm_factory = lambda **kw: ScheduledLLM(
                args.scheduler_socket,
                max_tokens=kw["max_tokens"],
                label=args.run_name or args.task,
                enable_thinking=kw["enable_thinking"],
            )
    if execution["eval_workers"] is None:
        execution["eval_workers"] = scheduler["cpu"]["capacity"] if scheduler else 1
        args.eval_workers = execution["eval_workers"]
    config = Config(
        budget=args.budget,
        output_tokens=args.output_tokens,
        evaluation_seeds=tuple(args.evaluation_seeds),
        seed=args.seed,
        **execution,
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
    if scheduler:
        metadata["resource_scheduler"] = scheduler
    if args.dry_run:
        evaluation, task_config = training_task(args.task, args.eval_workers, condition="traceaad")
        print(
            json.dumps(
                {
                    "method": TraceAAD.METHOD,
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
            and not (existing / "events.jsonl").exists()
        ):
            raise ValueError(f"refusing to overwrite a non-resumable run directory: {existing}")
    ctx = setup_experiment_run(
        args,
        method=TraceAAD.METHOD,
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
        method = TraceAAD(
            evaluation=ctx.evaluation,
            llm=ctx.llm,
            run_dir=ctx.run_dir,
            config=config,
            task=args.task,
        )
        ctx.run(method.run, ["TraceAAD: Refine, Explore and Crossover"])
        diagnose(ctx.run_dir)
    finally:
        ctx.llm.close()
