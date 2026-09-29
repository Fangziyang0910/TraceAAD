"""Run V10.14-4 with candidate accounting and a separate, frozen selection set."""

import argparse
from dataclasses import asdict
import json

from benchmarks.cvrp_aco import CVRPACOEvaluation
from benchmarks.generated_data_config import get_generated_task_kwargs
from benchmarks.online_bin_packing import OBPEvaluation
from benchmarks.op_aco import OPACOEvaluation
from benchmarks.tsp_construct import TSPEvaluation
from benchmarks.vrptw_construct import VRPTWEvaluation
from experiments.infra.base import RESULTS_ROOT
from experiments.infra.runner import add_common_run_args, setup_experiment_run
from traceaad.v10_14_4 import Config, TraceAADV10144


SELECTION_SEED = 20260927  # distinct from training=2024 and final test=2025


def selection_task(task, search):
    if task in {"op_aco", "cvrp_aco"}:
        cls = OPACOEvaluation if task == "op_aco" else CVRPACOEvaluation
        selection = cls(split="val_50", timeout_seconds=search.timeout_seconds,
                        n_ants=search.n_ants, n_iterations=search.n_iterations,
                        aco_seed=search.aco_seed, n_workers=search.n_workers)
        if search.timeout_seconds is not None:
            selection.timeout_seconds *= max(1., selection.n_instance / search.n_instance)
        return selection
    kwargs = get_generated_task_kwargs(task, "train")
    kwargs["seed"] = SELECTION_SEED
    cls = {"tsp_construct": TSPEvaluation, "vrptw_construct": VRPTWEvaluation,
           "online_bin_packing": OBPEvaluation}[task]
    return cls(**kwargs)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192)
    parser.add_argument("--max-evaluations", type=int, help="search evaluator calls; defaults to budget × seed count")
    parser.add_argument("--init-proposals", type=int, default=8)
    parser.add_argument("--init-mode", choices=("independent", "hybrid", "sequential"), default="hybrid")
    parser.add_argument("--regions", type=int, default=8)
    parser.add_argument("--initial-regions", type=int, default=4)
    parser.add_argument("--exploration-constant", type=float, default=.35)
    parser.add_argument("--discovery-fraction", type=float, default=.12)
    parser.add_argument("--parent-policy", choices=("rank_count", "raw_count"), default="rank_count")
    parser.add_argument("--pivot-context", choices=("independent", "anchored"), default="independent")
    parser.add_argument("--idea-tokens", type=int, default=320)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--min-behavior-distance", type=float, default=.01)
    parser.add_argument("--max-input-tokens", type=int, default=24320)
    parser.add_argument("--evidence-tokens", type=int, default=3000)
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--history-depth", type=int, default=4)
    parser.add_argument("--output-mode", choices=("full", "edit"), default="full")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[730241])
    parser.add_argument("--final-candidates", type=int, default=5)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--evidence-policy", choices=("none", "trajectory", "bag", "conditional"), default="conditional")
    parser.add_argument("--dry-run", action="store_true", help="print policy without creating a run or calling a model")
    return parser


def config_from_args(args):
    values = {name: getattr(args, name) for name in Config.__dataclass_fields__
              if hasattr(args, name) and name != "max_evaluations"}
    values["evaluation_seeds"] = tuple(args.evaluation_seeds)
    values["max_evaluations"] = args.max_evaluations if args.max_evaluations is not None else args.budget * len(args.evaluation_seeds)
    return Config(**values)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if args.dry_run:
        print(json.dumps({"method": "v1014_4", "task": args.task, "config": asdict(config),
            "selection": "val_50" if args.task in {"cvrp_aco", "op_aco"} else {"seed": SELECTION_SEED},
            "test": "only through the held-out evaluator after selection"}, indent=2))
        return
    if args.run_name:
        existing = RESULTS_ROOT / "traceaad_v10_14_4" / args.task / args.run_name
        if existing.is_dir() and any(existing.iterdir()) and not (existing / "search.jsonl").exists():
            raise ValueError(f"refusing to overwrite a non-resumable run directory: {existing}")
    ctx = setup_experiment_run(args, method="v1014_4", results_root=RESULTS_ROOT / "traceaad_v10_14_4",
        resume_file="search.jsonl", method_params=asdict(config),
        budget_basis="Attempted candidates; model calls, evaluations, probes and validation costs recorded separately.")
    try:
        # No hidden SDK retries outside the recorded generation ledger.
        ctx.llm._client = ctx.llm._client.with_options(max_retries=0)
        method = TraceAADV10144(evaluation=ctx.evaluation, llm=ctx.llm, run_dir=ctx.run_dir,
            config=config, task=args.task, selection_evaluation=selection_task(args.task, ctx.evaluation))
        ctx.run(method.run, ["V10.14-4: bounded discovery; region development; measured transition feedback"])
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()
