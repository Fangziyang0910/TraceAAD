"""Run TraceAAD V10.12."""

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from llm4ad.method.traceaad_v10_12 import TraceAADV1012
from llm4ad.method.traceaad_v10_12_rand_ctx import TraceAADV1012RandCtx


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192, default_budget=FORMAL_BUDGET)
    parser.add_argument('--n-roots', type=int, default=8)
    parser.add_argument('--traj-gens', type=int, default=8)
    parser.add_argument('--max-input-tokens', type=int, default=24576)
    parser.add_argument('--history-code', action='store_true',
                        help='include each historical trajectory program in generation context')
    parser.add_argument('--rand-context', action='store_true',
                        help='replace formation history with rank-sampled archive references')
    parser.add_argument('--n-references', type=int, default=8,
                        help='number of archive reference cards in random-context mode')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.n_references < 1:
        parser.error('n-references must be positive')
    if args.rand_context:
        if args.history_code or args.traj_gens != 8:
            parser.error('--rand-context excludes --history-code and non-default --traj-gens')
        params = {key: getattr(args, key) for key in (
            'budget', 'n_roots', 'max_input_tokens', 'output_tokens', 'n_references')}
        method_tag, method_cls = 'v1012rc', TraceAADV1012RandCtx
    else:
        if args.n_references != 8:
            parser.error('--n-references requires --rand-context')
        params = {key: getattr(args, key) for key in (
            'budget', 'n_roots', 'traj_gens', 'max_input_tokens',
            'output_tokens', 'history_code')}
        method_tag, method_cls = 'v1012', TraceAADV1012
    ctx = setup_experiment_run(
        args, method=method_tag, method_dir=Path(__file__).resolve().parent,
        resume_file='tree_state.json', method_params=params,
        budget_basis='Actual evaluator calls, including failures and repairs; '
                     'LLM-only failures and parent/donor copies consume no evaluator slot.')
    try:
        method = method_cls(evaluation=ctx.evaluation, llm=ctx.llm,
                            run_dir=ctx.run_dir, seed=args.seed, **params)
        ctx.run(method.run, header=['v1012: compact function-level search'])
    finally:
        ctx.llm.close()


if __name__ == '__main__':
    main()
