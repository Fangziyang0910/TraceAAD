"""Run V10.18 with one paid candidate per completed model generation."""

import argparse
from dataclasses import asdict
import json

from benchmarks.cvrp_aco import CVRPACOEvaluation
from benchmarks.generated_data_config import get_generated_task_kwargs
from benchmarks.online_bin_packing import OBPEvaluation
from benchmarks.op_aco import OPACOEvaluation
from benchmarks.tsp_construct import TSPEvaluation
from benchmarks.vrptw_construct import VRPTWEvaluation
from experiments.infra.base import RESULTS_ROOT, write_run_config
from experiments.infra.runner import add_common_run_args, setup_experiment_run
from traceaad.v10_18 import Config, TraceAADV1018


SELECTION_SEED = 20260927
TRAIN_TIMEOUT = {"online_bin_packing": 30, "vrptw_construct": 30}


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
    # Five one-off evaluations: no efficiency pressure is needed here, and a
    # finalist that met the search limit should not be lost to host load.
    kwargs["timeout_seconds"] = None if search.timeout_seconds is None else 2 * search.timeout_seconds
    cls = {"tsp_construct": TSPEvaluation, "vrptw_construct": VRPTWEvaluation,
           "online_bin_packing": OBPEvaluation}[task]
    return cls(**kwargs)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[730241])
    parser.add_argument("--experiment", default="traceaad_v10_18",
                        help="results directory under experiments_result (one per protocol batch series)")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.thinking or args.approx_chars_per_token is not None:
        raise ValueError("V10.18 requires thinking disabled and exact serving token counts")
    if args.repeat is not None and args.seed != args.repeat - 1:
        raise ValueError("run seed must equal repeat - 1")
    config = Config(budget=args.budget, output_tokens=args.output_tokens,
                    evaluation_seeds=tuple(args.evaluation_seeds), seed=args.seed)
    if args.dry_run:
        print(json.dumps({"method": "v1018", "task": args.task, "config": asdict(config),
            "search_timeout": TRAIN_TIMEOUT.get(args.task, 30 if args.task == "tsp_construct" else 120 if args.task == "cvrp_aco" else 60),
            "selection": "val_50" if args.task in {"cvrp_aco", "op_aco"} else {"seed": SELECTION_SEED},
            "test": "separate heldout.py after selection"}, indent=2))
        return
    if not args.experiment.replace("_", "").isalnum():
        raise ValueError("experiment must be alphanumeric with underscores")
    root = RESULTS_ROOT / args.experiment
    if args.run_name:
        existing = root / args.task / args.run_name
        if existing.is_dir() and any(existing.iterdir()) and not (existing / "search.jsonl").exists():
            raise ValueError(f"refusing to overwrite a non-resumable run directory: {existing}")
    ctx = setup_experiment_run(args, method="v1018", results_root=root,
        resume_file="search.jsonl", method_params=asdict(config),
        budget_basis="Completed model-generated candidates including initialization, failures, duplicates and repair.")
    try:
        ctx.llm._client = ctx.llm._client.with_options(max_retries=0)
        if args.task in TRAIN_TIMEOUT:
            ctx.evaluation.timeout_seconds = TRAIN_TIMEOUT[args.task]
            config_path = ctx.run_dir / "run_config.json"
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if ctx.resumed and saved["task_eval"]["timeout_seconds"] != ctx.evaluation.timeout_seconds:
                raise ValueError("resume timeout does not match the frozen task protocol")
            saved["task_eval"]["timeout_seconds"] = ctx.evaluation.timeout_seconds
            if not ctx.resumed:
                write_run_config(ctx.run_dir, saved)
        method = TraceAADV1018(evaluation=ctx.evaluation, llm=ctx.llm, run_dir=ctx.run_dir,
                               config=config, task=args.task,
                               selection_evaluation=selection_task(args.task, ctx.evaluation))
        ctx.run(method.run, ["V10.18: every new Explore design is developed from its best version until it shows whether it works (at least 2 steps, while improving, at most 8, until it ranks among the five best); Explore draws develop the open design before proposing; Refine/Explore/Crossover 0.35/0.40/0.25"])
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()
