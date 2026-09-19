"""Run one TraceAAD B/C scheduler-context ablation arm."""

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from llm4ad.method.traceaad_bc import (
    TraceAADV10BudgetV11Context,
    TraceAADV11BudgetV10Context,
)

ARM_CLASSES = {
    "B": TraceAADV11BudgetV10Context,
    "C": TraceAADV10BudgetV11Context,
}
ARM_METHODS = {
    "B": "bc_v11budget_v10ctx",
    "C": "bc_v10budget_v11ctx",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192, default_budget=FORMAL_BUDGET)
    parser.add_argument("--arm", choices=tuple(ARM_CLASSES), required=True)
    parser.add_argument("--n-roots", type=int, default=8)
    parser.add_argument("--traj-gens", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=24576)
    parser.add_argument("--n-references", type=int, default=8)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_references < 1:
        raise SystemExit("--n-references must be positive")
    if args.traj_gens < 0:
        raise SystemExit("--traj-gens must be nonnegative")
    method_cls = ARM_CLASSES[args.arm]
    params = {
        "budget": args.budget,
        "n_roots": args.n_roots,
        "traj_gens": args.traj_gens,
        "max_input_tokens": args.max_input_tokens,
        "output_tokens": args.output_tokens,
        "n_references": args.n_references,
        "arm": args.arm,
    }
    ctx = setup_experiment_run(
        args,
        method=ARM_METHODS[args.arm],
        method_dir=Path(__file__).resolve().parent,
        resume_file="tree_state.json",
        method_params=params,
        budget_basis=(
            "Actual evaluator calls, including failures, repairs, and duplicate-code "
            "generations; LLM-only failures consume no evaluator slot."
        ),
    )
    try:
        method = method_cls(
            evaluation=ctx.evaluation,
            llm=ctx.llm,
            run_dir=ctx.run_dir,
            seed=args.seed,
            **{key: value for key, value in params.items() if key != "arm"},
        )
        ctx.run(method.run, header=[
            f"TraceAAD B/C arm {args.arm}: "
            f"{method.scheduler if hasattr(method, 'scheduler') else method.mechanism['scheduler']} + "
            f"{method.mechanism['context']}"
        ])
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()
