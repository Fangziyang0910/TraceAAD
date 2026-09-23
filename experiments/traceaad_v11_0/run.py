"""Run TraceAAD V11.0."""

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from traceaad.v11_0 import TraceAADV110


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192, default_budget=FORMAL_BUDGET)
    parser.add_argument('--n-roots', type=int, default=8)
    parser.add_argument('--traj-gens', type=int, default=8)
    parser.add_argument('--max-input-tokens', type=int, default=24576)
    return parser


def main():
    args = build_parser().parse_args()
    params = {key: getattr(args, key) for key in (
        'budget', 'n_roots', 'traj_gens', 'max_input_tokens', 'output_tokens')}
    ctx = setup_experiment_run(
        args, method='v110', method_dir=Path(__file__).resolve().parent,
        resume_file='tree_state.json', method_params=params,
        budget_basis='Actual evaluator calls, including failures, repairs, and '
                     'duplicate-code generations; LLM-only failures consume no evaluator slot.')
    try:
        method = TraceAADV110(evaluation=ctx.evaluation, llm=ctx.llm,
                              run_dir=ctx.run_dir, seed=args.seed, **params)
        ctx.run(method.run, header=[
            'v110: percentile-plus-bonus scheduling with operator-conditional reference context'])
    finally:
        ctx.llm.close()


if __name__ == '__main__':
    main()
