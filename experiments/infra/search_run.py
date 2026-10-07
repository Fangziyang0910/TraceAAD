"""Search on training data, freeze its best program, then evaluate held-out separately."""

import argparse
from dataclasses import asdict
import json

from benchmarks.tasks import SELECTION_SEED, selection_task, training_task
from experiments.infra.base import RESULTS_ROOT
from experiments.infra.runner import add_common_run_args, setup_experiment_run
from traceaad.common.config import REVISION
from .diagnose_search import diagnose


def build_parser(experiment):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[730241])
    parser.add_argument("--experiment", default=experiment,
                        help="results directory under experiments_result (one per protocol batch series)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument('--final-selection', choices=('training', 'validation'), default='training',
                        help='default: training-best; validation reproduces historical selection conditions')
    return parser


def main(method_class, config_class, experiment, description, argv=None):
    args = build_parser(experiment).parse_args(argv)
    if args.thinking or args.approx_chars_per_token is not None:
        raise ValueError("search requires thinking disabled and exact serving token counts")
    if args.repeat is not None and args.seed != args.repeat - 1:
        raise ValueError("run seed must equal repeat - 1")
    config = config_class(budget=args.budget, output_tokens=args.output_tokens,
                    evaluation_seeds=tuple(args.evaluation_seeds), seed=args.seed)
    if args.dry_run:
        evaluation, _ = training_task(args.task, args.eval_workers, condition="traceaad")
        selection = selection_task(args.task, evaluation) if args.final_selection == 'validation' else None
        print(json.dumps({"method": method_class.METHOD, "revision": REVISION, "task": args.task, "config": asdict(config),
            "search_timeout": evaluation.timeout_seconds,
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
        extra_config={"revision": REVISION, "budget_axis": "候选尝试", "final_selection": args.final_selection},
        budget_basis="Completed model-generated candidates including initialization, failures, duplicates and repair.")
    try:
        method = method_class(evaluation=ctx.evaluation, llm=ctx.llm, run_dir=ctx.run_dir,
                               config=config, task=args.task,
                               selection_evaluation=(selection_task(args.task, ctx.evaluation) if args.final_selection == 'validation' else None))
        ctx.run(method.run, [description])
        diagnose(ctx.run_dir)
    finally:
        ctx.llm.close()
