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


def build_parser(experiment=EXPERIMENT, config_type=Config):
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
        "--evaluation-seeds", type=int, nargs="+", default=list(Config.evaluation_seeds)
    )
    parser.add_argument(
        "--experiment", default=experiment, help="results directory under experiments_result"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None, *, version=None):
    method_type, config_type, experiment = TraceAAD, Config, EXPERIMENT
    extra = {}
    if version == 'v13':
        from traceaad.v13 import Config as config_type, TraceAADV13 as method_type
        from experiments.traceaad_v13 import options
        experiment = 'traceaad_v13'
    elif version == 'v13_1':
        from traceaad.v13_1 import Config as config_type, TraceAADV131 as method_type
        from experiments.traceaad_v13_1 import options
        experiment = 'traceaad_v13_1'
    elif version == 'v14':
        from traceaad.v14 import Config as config_type, TraceAADV14 as method_type
        from experiments.traceaad_v14 import options
        experiment = 'traceaad_v14'
    elif version == 'v15':
        from traceaad.v15 import Config as config_type, TraceAADV15 as method_type
        from experiments.traceaad_v15 import options
        experiment = 'traceaad_v15'
    elif version == 'v16':
        from traceaad.v16 import Config as config_type, TraceAADV16 as method_type
        from experiments.traceaad_v16 import options
        experiment = 'traceaad_v16'
    elif version == 'v17':
        from traceaad.v17 import Config as config_type, TraceAADV17 as method_type
        from experiments.traceaad_v17 import options
        experiment = 'traceaad_v17'
    parser = build_parser(experiment, config_type)
    if version in ('v13', 'v13_1', 'v14', 'v15', 'v16', 'v17'):
        options.add_arguments(parser)
    args = parser.parse_args(argv)
    if version in ('v13', 'v13_1', 'v14', 'v15', 'v16', 'v17'):
        extra = options.values(args)
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
        metadata['model_routing'] = {'policy': 'shared', 'endpoints': scheduler['endpoints']}
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
            and not (existing / "events.jsonl").exists()
        ):
            raise ValueError(f"refusing to overwrite a non-resumable run directory: {existing}")
    if version in ('v13', 'v13_1', 'v14', 'v15', 'v16', 'v17') and args.run_name:
        from traceaad.storage import read_json
        previous = read_json(root / args.task / args.run_name / 'run_config.json')
        if previous and (previous.get('method') != method_type.METHOD or any(
                previous.get('method_params', {}).get(k) != v for k, v in json.loads(json.dumps(asdict(config))).items())):
            raise ValueError('V13 resume configuration differs from the stored run')
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
        budget_basis=("Candidate opportunities including initialization, nonmatches and failed edits; all auxiliary/failed/interrupted model requests consume max_model_calls."
                      if version in ('v13', 'v13_1') else "Program-generation replies including initialization, parse failures and repair; every parsed program is a distinct individual."),
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
