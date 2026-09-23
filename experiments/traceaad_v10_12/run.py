"""Run TraceAAD V10.12."""

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from experiments.traceaad_v10_12.freeze import verify_runtime
from traceaad.v10_12 import TraceAADV1012


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192, default_budget=FORMAL_BUDGET)
    parser.add_argument('--n-roots', type=int, default=8)
    parser.add_argument('--traj-gens', type=int, default=8)
    parser.add_argument('--max-input-tokens', type=int, default=24576)
    parser.add_argument('--history-code', action='store_true',
                        help='include each historical trajectory program in generation context')
    parser.add_argument('--n-profile-cards', type=int, default=2,
                        help='number of archive profile cards to include in generation context')
    parser.add_argument('--profile-card-tau', type=float, default=8.0,
                        help='temperature for rank-softmax archive card sampling')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[2]
    if not (runtime / 'runtime_manifest.json').is_file():
        parser.error('search must run from a verified frozen runtime; use manual_launch')
    verify_runtime(runtime, preflight=False)
    if args.n_profile_cards < 0:
        parser.error('n-profile-cards must be non-negative')
    params = {key: getattr(args, key) for key in (
        'budget', 'n_roots', 'traj_gens', 'n_profile_cards', 'profile_card_tau',
        'max_input_tokens', 'output_tokens', 'history_code')}
    ctx = setup_experiment_run(
        args, method='v1012', method_dir=Path(__file__).resolve().parent,
        resume_file='tree_state.json', method_params=params,
        budget_basis='Actual evaluator calls, including failures and repairs; '
                     'LLM-only failures and parent/donor copies consume no evaluator slot.')
    try:
        method = TraceAADV1012(evaluation=ctx.evaluation, llm=ctx.llm,
                            run_dir=ctx.run_dir, seed=args.seed, **params)
        ctx.run(method.run, header=['v1012: compact function-level search'])
    finally:
        ctx.llm.close()


if __name__ == '__main__':
    main()
