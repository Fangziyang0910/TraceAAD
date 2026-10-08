"""Search on training data, freeze its best program, then evaluate held-out separately."""

import argparse
from dataclasses import asdict
import json

from benchmarks.tasks import FUNCTION_SECONDS, SELECTION_SEED, selection_task, training_task
from experiments.infra.base import RESULTS_ROOT
from experiments.infra.runner import add_common_run_args, setup_experiment_run
from traceaad.common.config import REVISION
from .diagnose_search import diagnose


def build_parser(experiment, config_class=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192)
    if config_class is not None and hasattr(config_class, "eval_timeout_seconds"):
        parser.add_argument('--scheduler-socket', help='shared CPU/GPU scheduler on this host')
        from benchmarks.tasks import INSTANCE_SECONDS
        parser.add_argument("--eval-timeout-seconds", type=float, default=float(INSTANCE_SECONDS),
                            help="wall-clock limit for each instance, including the fixed solver")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[730241])
    parser.add_argument("--experiment", default=experiment,
                        help="results directory under experiments_result (one per protocol batch series)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument('--final-selection', choices=('training', 'validation'), default='training',
                        help='default: training-best; validation reproduces historical selection conditions')
    return parser


def main(method_class, config_class, experiment, description, argv=None):
    args = build_parser(experiment, config_class).parse_args(argv)
    if args.thinking or args.approx_chars_per_token is not None:
        raise ValueError("search requires thinking disabled and exact serving token counts")
    if args.repeat is not None and args.seed != args.repeat - 1:
        raise ValueError("run seed must equal repeat - 1")
    execution = ({"eval_timeout_seconds": args.eval_timeout_seconds, "eval_workers": args.eval_workers}
                 if hasattr(args, "eval_timeout_seconds") else {})
    scheduler, llm_factory, profile = None, None, None
    if getattr(args, 'scheduler_socket', None):
        from pathlib import Path
        from core.scheduling import scheduler_status
        from .base import BackendProfile
        from .scheduled_llm import ScheduledLLM
        args.scheduler_socket = str(Path(args.scheduler_socket).expanduser().resolve())
        scheduler = scheduler_status(args.scheduler_socket)
        execution['scheduler_socket'] = args.scheduler_socket
        endpoint = scheduler['endpoints'][0]
        profile = BackendProfile(endpoint['base_url'], endpoint['model'], endpoint.get('no_proxy', ''))
        llm_factory = lambda **kw: ScheduledLLM(args.scheduler_socket, max_tokens=kw['max_tokens'],
                                               label=args.run_name or args.task)
    if execution and execution['eval_workers'] is None:
        execution['eval_workers'] = scheduler['cpu']['capacity'] if scheduler else 1
        args.eval_workers = execution['eval_workers']
    config = config_class(budget=args.budget, output_tokens=args.output_tokens,
                    evaluation_seeds=tuple(args.evaluation_seeds), seed=args.seed, **execution)
    metadata = ({"evaluation_execution": {"timeout_scope": "instance",
                 "timeout_seconds": config.eval_timeout_seconds, "function_seconds": FUNCTION_SECONDS,
                 "n_workers": config.eval_workers,
                 "protocol": "isolated-instances-v1"}} if execution else {})
    if scheduler:
        metadata['resource_scheduler'] = scheduler
    if args.dry_run:
        evaluation, _ = training_task(args.task, args.eval_workers, condition="traceaad")
        selection = selection_task(args.task, evaluation) if args.final_selection == 'validation' else None
        print(json.dumps({"method": method_class.METHOD, "revision": REVISION, "task": args.task, "config": asdict(config),
            "search_timeout": config.eval_timeout_seconds if execution else evaluation.timeout_seconds,
            **metadata,
            "final_selection": args.final_selection,
            "selection": (getattr(selection, "instance_description", "val_50" if args.task in {"cvrp_aco", "op_aco"} else {"seed": SELECTION_SEED}) if selection else None),
            "test": "separate held-out evaluation of the frozen program"}, indent=2))
        return
    if not args.experiment.replace("_", "").isalnum():
        raise ValueError("experiment must be alphanumeric with underscores")
    root = RESULTS_ROOT / args.experiment
    if args.run_name:
        existing = root / args.task / args.run_name
        if existing.is_dir() and any(existing.iterdir()) and not (existing / "events.jsonl").exists():
            raise ValueError(f"refusing to overwrite a non-resumable run directory: {existing}")
    ctx = setup_experiment_run(args, method=method_class.METHOD, results_root=root,
        resume_file="events.jsonl", method_params=asdict(config), condition="traceaad",
        extra_config={"revision": REVISION, "budget_axis": "候选尝试", "final_selection": args.final_selection, **metadata},
        llm_factory=llm_factory, backend_profile=profile,
        budget_basis="Completed model-generated candidates including initialization, failures, duplicates and repair.")
    try:
        method = method_class(evaluation=ctx.evaluation, llm=ctx.llm, run_dir=ctx.run_dir,
                               config=config, task=args.task,
                               selection_evaluation=(selection_task(args.task, ctx.evaluation) if args.final_selection == 'validation' else None))
        ctx.run(method.run, [description])
        diagnose(ctx.run_dir)
    finally:
        ctx.llm.close()
