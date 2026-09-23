"""Run one TraceAAD V10.13 search."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from traceaad.v10_13 import TraceAADV1013


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192, default_budget=FORMAL_BUDGET)
    parser.add_argument("--n-roots", type=int, default=8)
    parser.add_argument("--history-depth", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=24320)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    runtime = Path(__file__).resolve().parents[2]
    manifest = runtime / "runtime_manifest.json"
    if not manifest.exists():
        raise SystemExit("search must run from a verified frozen runtime")
    payload = json.loads(manifest.read_text())
    for relative, expected in payload["files"].items():
        path = runtime / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"frozen source changed or missing: {relative}")
    if args.history_depth < 0:
        raise SystemExit("--history-depth must be nonnegative")
    params = {
        "budget": args.budget,
        "n_roots": args.n_roots,
        "history_depth": args.history_depth,
        "max_input_tokens": args.max_input_tokens,
        "output_tokens": args.output_tokens,
    }
    ctx = setup_experiment_run(
        args, method="v1013", method_dir=Path(__file__).resolve().parent,
        resume_file="tree_state.json", method_params={**params, "revision": TraceAADV1013.REVISION},
        budget_basis=(
            "Actual evaluator calls, including failures, repairs, and duplicate-code "
            "generations; LLM-only failures consume no evaluator slot, "
            "but all LLM calls and available token usage are recorded separately. "
            "A reserved evaluation without a durable receipt blocks automatic recovery."
        ),
    )
    try:
        method = TraceAADV1013(
            evaluation=ctx.evaluation,
            llm=ctx.llm,
            run_dir=ctx.run_dir,
            seed=args.seed,
            **params,
        )
        ctx.run(method.run, header=[
            "v10.13-r3: ESS allocation, single-pass evidence, Python-first optional edits"
        ])
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()
