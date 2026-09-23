"""Standard experiment execution harness.

Unifies CLI argument parsing, backend profile resolution, run-directory setup,
config persistence, task and LLM client initialization, and tmux logging across
all experiment entries.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.infra.base import (
    ALL_TASKS,
    BACKENDS,
    BackendProfile,
    build_llm_client,
    build_task,
    llm_payload,
    resolve_backend,
    run_in_tmux_log,
    set_random_seed,
    write_run_config,
)
from core.llm import OpenAIAPI

FORMAL_BUDGET = 1000


def add_common_run_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_tokens: int = 16384,
    default_budget: int = FORMAL_BUDGET,
) -> None:
    """Add standard arguments required by all experiment runners."""
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--output-tokens", type=int, default=default_output_tokens)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--budget", type=int, default=default_budget)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--thinking", action="store_true",
                        help="enable model thinking mode for generation requests")
    parser.add_argument("--approx-chars-per-token", type=float,
                        help="count tokens locally by characters (for gateways without /tokenize)")


@dataclass(frozen=True, slots=True)
class RunContext:
    task: str
    run_dir: Path
    run_name: str
    resumed: bool
    seed: int
    llm: OpenAIAPI
    evaluation: Any
    profile: BackendProfile
    log_dir: Path
    args: argparse.Namespace

    def run(self, body: Callable[[], None], header: list[str] | None = None) -> None:
        """Run the experiment body within the tmux run log context."""
        meta_header = [
            f"log_dir={self.log_dir}",
            f"llm={self.profile.model} @ {self.profile.base_url}",
            f"resumed={self.resumed}",
        ]
        full_header = (header or []) + meta_header if header else meta_header
        run_in_tmux_log(self.run_dir, self.log_dir, full_header, body)


def resolve_resumable_run_dir(
    task_root: Path,
    run_name: str | None,
    resume_file: str | None = None,
) -> tuple[Path, str, bool]:
    """Resolve a new run dir or existing one for resume."""
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = task_root / run_name
    if run_dir.exists():
        if resume_file and (run_dir / resume_file).exists():
            return run_dir, run_name, True
        print(f"run dir exists without resumable state, starting fresh: {run_dir}")
        return run_dir, run_name, False
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name, False


def setup_experiment_run(
    args: argparse.Namespace,
    *,
    method: str,
    method_dir: Path,
    resume_file: str | None = None,
    method_params: dict[str, Any] | None = None,
    budget_basis: str | None = None,
) -> RunContext:
    """Set up the standard environment, LLM, task evaluation, and configuration."""
    profile = resolve_backend(args.backend, args.base_url, args.model, args.no_proxy)
    task_root = method_dir / "results" / args.task
    run_dir, run_name, resumed = resolve_resumable_run_dir(task_root, args.run_name, resume_file)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    params = dict(method_params or {})
    if "budget" not in params and hasattr(args, "budget"):
        params["budget"] = args.budget
    if budget_basis:
        params["budget_basis"] = budget_basis

    if not resumed:
        evaluation, task_config = build_task(args.task, getattr(args, "eval_workers", None))
        write_run_config(
            run_dir,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "run_name": run_name,
                "task": args.task,
                "method": method,
                "repeat": args.repeat,
                "backend": args.backend,
                "seed": args.seed,
                "llm": llm_payload(
                    base_url=profile.base_url,
                    model=profile.model,
                    no_proxy=profile.no_proxy,
                    max_tokens=args.output_tokens,
                    enable_thinking=getattr(args, "thinking", False),
                    chars_per_token=getattr(args, "approx_chars_per_token", None),
                ),
                "task_eval": task_config,
                "method_params": params,
            },
        )
    else:
        evaluation, _ = build_task(args.task, getattr(args, "eval_workers", None))

    set_random_seed(args.seed)
    llm = build_llm_client(
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_tokens=args.output_tokens,
        enable_thinking=getattr(args, "thinking", False),
        chars_per_token=getattr(args, "approx_chars_per_token", None),
    )

    return RunContext(
        task=args.task,
        run_dir=run_dir,
        run_name=run_name,
        resumed=resumed,
        seed=args.seed,
        llm=llm,
        evaluation=evaluation,
        profile=profile,
        log_dir=log_dir,
        args=args,
    )

