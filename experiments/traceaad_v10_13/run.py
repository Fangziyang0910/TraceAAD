"""Run one TraceAAD V10.13 search."""

import argparse
import json

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from experiments.infra.base import RESULTS_ROOT
from traceaad.v10_13 import TraceAADV1013


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192, default_budget=FORMAL_BUDGET)
    parser.add_argument("--n-roots", type=int, default=8)
    parser.add_argument("--init-mode", choices=("independent", "sequential", "hybrid"),
                        default="hybrid")
    parser.add_argument("--history-depth", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=24320)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.history_depth < 0:
        raise SystemExit("--history-depth must be nonnegative")
    if args.n_roots < 1 or args.n_roots > args.budget:
        raise SystemExit("--n-roots must be between 1 and --budget")
    params = {
        "budget": args.budget,
        "n_roots": args.n_roots,
        "init_mode": args.init_mode,
        "history_depth": args.history_depth,
        "max_input_tokens": args.max_input_tokens,
        "output_tokens": args.output_tokens,
    }
    ctx = setup_experiment_run(
        args, method="v1013", results_root=RESULTS_ROOT / "traceaad_v10_13",
        resume_file="tree_state.json", method_params=params,
        budget_basis="Each evaluator call consumes one budget unit.",
    )
    try:
        if ctx.resumed:
            config = json.loads((ctx.run_dir / "run_config.json").read_text(encoding="utf-8"))
            previous_params = config.get("method_params", {})
            previous_mode = previous_params.get("init_mode", "sequential")
            if previous_mode != args.init_mode:
                raise ValueError(
                    f"run was initialized with {previous_mode}; "
                    f"resume with --init-mode {previous_mode}")
            previous_roots = previous_params.get("n_roots", 8)
            if previous_roots != args.n_roots:
                raise ValueError(
                    f"run was initialized with {previous_roots} roots; "
                    f"resume with --n-roots {previous_roots}")
        method = TraceAADV1013(
            evaluation=ctx.evaluation,
            llm=ctx.llm,
            run_dir=ctx.run_dir,
            seed=args.seed,
            **params,
        )
        ctx.run(method.run, header=["V10.13: compact single-pass evolutionary search"])
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()
