"""Run one TraceAAD V10.13 search."""

import argparse
import json

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from experiments.infra.base import RESULTS_ROOT, resolve_backend
from traceaad.v10_13 import TraceAADV1013
from traceaad.v10_13.storage import JOURNAL_NAME, RunStorage


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
    if args.run_name:
        existing = RESULTS_ROOT / "traceaad_v10_13" / args.task / args.run_name
        if existing.is_dir() and any(existing.iterdir()) and not (existing / JOURNAL_NAME).exists():
            raise SystemExit(f"existing run has no V10.13 journal: {existing}")
    ctx = setup_experiment_run(
        args, method="v1013", results_root=RESULTS_ROOT / "traceaad_v10_13",
        resume_file=JOURNAL_NAME, method_params=params,
        budget_basis="Each evaluator call consumes one budget unit.",
    )
    try:
        if ctx.resumed:
            storage = RunStorage(ctx.run_dir)
            completed = storage.load_summary()
            if not (completed and completed.get("status") == "finished" and completed.get("budget", 0) >= args.budget):
                config = json.loads((ctx.run_dir / "run_config.json").read_text(encoding="utf-8"))
                previous_params = config.get("method_params", {})
                for key in ("budget", "history_depth", "max_input_tokens", "output_tokens"):
                    if key in previous_params and previous_params[key] != params[key]:
                        raise ValueError(f"run was initialized with {key}={previous_params[key]}")
                if "seed" in config and config["seed"] != args.seed:
                    raise ValueError(f"run was initialized with seed={config['seed']}")
                previous_model = (config.get("llm") or {}).get("model")
                current_model = resolve_backend(args.backend, args.base_url, args.model, args.no_proxy).model
                if previous_model and previous_model != current_model:
                    raise ValueError(f"run was initialized with model={previous_model}")
                previous_thinking = (config.get("llm") or {}).get("enable_thinking")
                if previous_thinking is not None and previous_thinking != args.thinking:
                    raise ValueError(f"run was initialized with enable_thinking={previous_thinking}")
                previous_mode = previous_params.get("init_mode")
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
        method.run()
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()
