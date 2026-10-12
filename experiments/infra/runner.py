"""Standard experiment execution harness.

Unifies CLI argument parsing, backend profile resolution, run-directory setup,
config persistence, task and LLM client initialization, and tmux logging across
all experiment entries.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from collections.abc import Callable
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.infra.base import (
    scheduler_socket,
    ALL_TASKS,
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
    write_run_config,
)
from benchmarks.tasks import CO_TASKS, FUNCTION_LIMITS, INSTANCE_LIMITS, NATIVE_MINIMIZE
from core.evaluate import EVALUATION_SEED
from traceaad.common.storage import read_json, write_json, repair_jsonl_tail
from core.llm import OpenAIAPI
from traceaad.common.config import REVISION

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
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="override model thinking mode (DeepSeek defaults to disabled; --thinking enables high)")
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
    profile = backend_profile or resolve_backend(args.backend, args.base_url, args.model, args.no_proxy)
    task_root = (results_root or RESULTS_ROOT / method) / args.task
    run_dir, run_name, resumed = resolve_resumable_run_dir(task_root, args.run_name, resume_file)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    params = dict(method_params or {})
    if "budget" not in params and hasattr(args, "budget"):
        params["budget"] = args.budget
    if budget_basis:
        params["budget_basis"] = budget_basis

    evaluation, task_config = build_task(args.task, getattr(args, "eval_workers", None), condition=condition)
    saved = read_json(run_dir / "run_config.json", {})
    if saved:
        from benchmarks.tasks import validate_task_interface
        validate_task_interface(args.task, saved)
        if args.task in CO_TASKS and saved.get('revision') != REVISION:
            raise ValueError('evaluation budget changed; use a new batch instead of rewriting the saved configuration')
    phase = read_json(run_dir / "resume.json", {}).get("state", {}).get("phase")
    if phase not in {"finished", "search_complete", "selection_failed", "no_valid_root"}:
        write_run_config(
            run_dir,
            {
                "created_at": saved.get("created_at", datetime.now().isoformat(timespec="seconds")),
                "budget": getattr(args, "budget", params.get("max_sample_nums", 0)),
                "objective": "min",
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
                    chars_per_token=getattr(args, "approx_chars_per_token", None),
                ),
                "task_eval": task_config,
                "method_params": params,
                "evaluation_condition": condition,
                "revision": REVISION,
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


@contextmanager
def baseline_run_lock(run_dir):
    """One writer per run; a dead writer releases its lock even after SIGKILL."""
    with (Path(run_dir) / '.recovery.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Run has an active writer: {run_dir}') from exc
        summary_path = Path(run_dir) / 'summary.json'
        summary = read_json(summary_path, {})
        if summary.get('status') == 'running' and not summary.get('writer_lock'):
            raise ValueError('Legacy run marked running; verify its process has stopped before updating its status')
        for path in Path(run_dir).rglob('*.jsonl'):
            repair_jsonl_tail(path, Path(run_dir) / 'logs/interrupted_tails')
        try:
            yield
        finally:
            summary = read_json(summary_path, {})
            if summary.get('status') == 'running' and summary.get('writer_lock'):
                write_json(summary_path, {**summary, 'status': 'stopped', 'phase': 'stopped'})


def run_baseline_experiment(spec, *, write_config, build_method, description: str) -> Path:
    """Run a baseline with the shared directory, config and log lifecycle."""
    from experiments.infra.base import resolve_run_dir

    run_dir, run_name = resolve_run_dir(spec.experiment_root, spec.run_name)
    log_dir = run_dir / "logs"
    # Each evaluation's outcome, failure kind included, whatever the method records.
    os.environ["TRACEAAD_EVALUATION_LOG"] = str(run_dir / "evaluations.jsonl")
    with baseline_run_lock(run_dir):
        write_config(spec, run_dir, run_name)
        write_json(run_dir / 'summary.json', {'status': 'running', 'writer_lock': True})
        print(f"run_dir={run_dir}")
        run_in_tmux_log(
            run_dir, log_dir,
            [f"log_dir={log_dir}", f"llm={spec.model} @ {spec.base_url}", description],
            lambda: build_method(spec, log_dir).run(),
        )
    return run_dir


def baseline_run_config(spec, run_dir: Path, run_name: str, method: str,
                        method_params: dict[str, Any], *,
                        task_config: dict[str, Any] | None = None,
                        llm_options: dict[str, Any] | None = None,
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
    if task_config is None:
        _, task_config = build_task(spec.task, spec.eval_workers)
    if spec.task in CO_TASKS:
        method_params = {**method_params, "evaluation_seeds": [EVALUATION_SEED],
                         "eval_timeout_seconds": INSTANCE_LIMITS[spec.task], "function_seconds": FUNCTION_LIMITS[spec.task],
                         "function_budget_scope": "mean_instance",
                         "eval_workers": 1 if spec.eval_workers is None else spec.eval_workers,
                         "scheduler_socket": scheduler_socket()}
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "budget": method_params["max_sample_nums"],
        "objective": "min",
        "native_objective": "min" if spec.task in NATIVE_MINIMIZE else "max",
        "budget_axis": "样本次数",
        "run_dir": str(run_dir), "run_name": run_name, "task": spec.task,
        "method": method, "repeat": spec.repeat, "backend": spec.backend,
        "seed": spec.seed,
        "llm": llm_payload(
            base_url=spec.base_url, model=spec.model, no_proxy=spec.no_proxy,
            max_tokens=spec.output_tokens,
            **(llm_options or {}),
        ),
        "task_eval": task_config, "method_params": method_params,
        "evaluation_condition": "shared", "revision": REVISION,
        **(extra or {}),
    }
