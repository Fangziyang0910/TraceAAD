"""Run one TraceAAD V11.1 search."""

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from llm4ad.method.traceaad_v11_1 import TraceAADV111


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192, default_budget=FORMAL_BUDGET)
    parser.add_argument("--n-roots", type=int, default=8)
    parser.add_argument("--history-depth", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=24576)
    parser.add_argument("--n-references", type=int, default=8)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_references < 1:
        raise SystemExit("--n-references must be positive")
    if args.history_depth < 0:
        raise SystemExit("--history-depth must be nonnegative")
    params = {
        "budget": args.budget,
        "n_roots": args.n_roots,
        "history_depth": args.history_depth,
        "max_input_tokens": args.max_input_tokens,
        "output_tokens": args.output_tokens,
        "n_references": args.n_references,
    }
    ctx = setup_experiment_run(
        args, method="v111", method_dir=Path(__file__).resolve().parent,
        resume_file="tree_state.json", method_params=params,
        budget_basis=(
            "Actual evaluator calls, including failures, repairs, and duplicate-code "
            "generations; LLM-only failures consume no evaluator slot."
        ),
    )
    try:
        method = TraceAADV111(
            evaluation=ctx.evaluation,
            llm=ctx.llm,
            run_dir=ctx.run_dir,
            seed=args.seed,
            **params,
        )
        ctx.run(method.run, header=[
            "v111: fixed-temperature node scheduling with operator-conditional context"
        ])
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()
