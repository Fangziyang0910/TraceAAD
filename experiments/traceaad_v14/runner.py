"""Initialize and resume v14 experiments with local run tools."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.tasks import ALL_TASKS
from .base import (
    BACKENDS,
    DEFAULT_BACKEND,
    RESULTS_ROOT,
    BackendProfile,
    build_llm_client,
    build_task,
    llm_payload,
    resolve_backend,
    run_in_tmux_log,
    set_random_seed,
)
from benchmarks.tasks import NATIVE_MINIMIZE
from .storage import read_json, write_json
from traceaad.v14.runtime.llm import OpenAIAPI

FORMAL_BUDGET = 1000


def add_common_run_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_tokens: int = 16384,
    default_budget: int = FORMAL_BUDGET,
) -> None:
    """Add standard arguments required by all experiment runners."""
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--output-tokens", type=int, default=default_output_tokens)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--budget", type=int, default=default_budget)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override model thinking mode (DeepSeek defaults to disabled; --thinking enables high)",
    )


@dataclass(frozen=True, slots=True)
class RunContext:
    run_dir: Path
    resumed: bool
    llm: OpenAIAPI
    evaluation: Any
    profile: BackendProfile
    log_dir: Path

    def run(self, body: Callable[[], None], header: list[str] | None = None) -> None:
        """Run the experiment body within the tmux run log context."""
        meta_header = [
            f"log_dir={self.log_dir}",
            f"llm={self.profile.model} @ {self.profile.base_url}",
            f"resumed={self.resumed}",
        ]
        full_header = (header or []) + meta_header
        run_in_tmux_log(self.run_dir, full_header, body)


def resolve_resumable_run_dir(
    task_root: Path, run_name: str | None, resume_file: str | None = None
) -> tuple[Path, str, bool]:
    """Resolve a new run dir or existing one for resume."""
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = task_root / run_name
    if run_dir.exists():
        if resume_file and (run_dir / resume_file).exists():
            return run_dir, run_name, True
        raise FileExistsError(f"run directory has no resumable state: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name, False


def setup_experiment_run(
    args: argparse.Namespace,
    *,
    method: str,
    results_root: Path | None = None,
    resume_file: str | None = None,
    method_params: dict[str, Any] | None = None,
    budget_basis: str | None = None,
    condition: str = "shared",
    extra_config: dict | None = None,
    llm_factory=None,
    backend_profile=None,
) -> RunContext:
    """Set up the standard environment, LLM, task evaluation, and configuration."""
    profile = backend_profile or resolve_backend(
        args.backend, args.base_url, args.model, args.no_proxy
    )
    task_root = (results_root or RESULTS_ROOT / method) / args.task
    run_dir, run_name, resumed = resolve_resumable_run_dir(task_root, args.run_name, resume_file)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    params = dict(method_params or {})
    if "budget" not in params and hasattr(args, "budget"):
        params["budget"] = args.budget
    if budget_basis:
        params["budget_basis"] = budget_basis

    evaluation, task_config = build_task(
        args.task, getattr(args, "eval_workers", None), condition=condition
    )
    saved = read_json(run_dir / "run_config.json", {})
    if saved:
        from benchmarks.tasks import validate_task_interface

        validate_task_interface(args.task, saved)
    phase = read_json(run_dir / "resume.json", {}).get("state", {}).get("phase")
    if phase not in {"finished", "no_valid_root"}:
        write_json(
            run_dir / "run_config.json",
            {
                "created_at": saved.get("created_at", datetime.now().isoformat(timespec="seconds")),
                "budget": getattr(args, "budget", params.get("max_sample_nums", 0)),
                "objective": "min",
                "score_metric": "raw_objective",
                "native_objective": "min" if args.task in NATIVE_MINIMIZE else "max",
                "budget_axis": "样本次数",
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
                    enable_thinking=getattr(args, "thinking", None),
                ),
                "task_eval": task_config,
                "method_params": params,
                "evaluation_condition": condition,
                **(extra_config or {}),
            },
        )
    set_random_seed(args.seed)
    llm = (llm_factory or build_llm_client)(
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_tokens=args.output_tokens,
        enable_thinking=getattr(args, "thinking", None),
    )

    return RunContext(
        run_dir=run_dir,
        resumed=resumed,
        llm=llm,
        evaluation=evaluation,
        profile=profile,
        log_dir=log_dir,
    )
