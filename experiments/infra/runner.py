"""Standard experiment execution harness.

Unifies CLI argument parsing, backend profile resolution, run-directory setup,
config persistence, task and LLM client initialization, and tmux logging across
all experiment entries.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.tasks import ALL_TASKS
from experiments.infra.base import (
    scheduler_socket,
    BACKENDS,
    HOSTED_BACKENDS,
    DEFAULT_BACKEND,
    build_task,
    llm_payload,
    run_in_tmux_log,
)
from benchmarks.tasks import CO_TASKS, FUNCTION_LIMITS, INSTANCE_LIMITS, NATIVE_MINIMIZE
from core.evaluate import EVALUATION_SEED
from .storage import read_json, write_json, repair_jsonl_tail

FORMAL_BUDGET = 1000


def add_common_run_args(
    parser: argparse.ArgumentParser, *, default_output_tokens: int = 16384, default_budget: int = FORMAL_BUDGET
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


@contextmanager
def baseline_run_lock(run_dir):
    """One writer per run; a dead writer releases its lock even after SIGKILL."""
    with (Path(run_dir) / ".recovery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Run has an active writer: {run_dir}") from exc
        summary_path = Path(run_dir) / "summary.json"
        for path in Path(run_dir).rglob("*.jsonl"):
            repair_jsonl_tail(path, Path(run_dir) / "logs/interrupted_tails")
        try:
            yield
        finally:
            summary = read_json(summary_path, {})
            if summary.get("status") == "running" and summary.get("writer_lock"):
                write_json(summary_path, {**summary, "status": "stopped", "phase": "stopped"})


def run_baseline_experiment(spec, *, write_config, build_method, description: str) -> Path:
    """Run a baseline with the shared directory, config and log lifecycle."""
    from experiments.infra.base import resolve_run_dir

    run_dir, run_name = resolve_run_dir(spec.experiment_root, spec.run_name)
    log_dir = run_dir / "logs"
    # Each evaluation's outcome, failure kind included, whatever the method records.
    os.environ["TRACEAAD_EVALUATION_LOG"] = str(run_dir / "evaluations.jsonl")
    with baseline_run_lock(run_dir):
        write_config(spec, run_dir, run_name)
        write_json(run_dir / "summary.json", {"status": "running", "writer_lock": True})
        print(f"run_dir={run_dir}")
        run_in_tmux_log(
            run_dir,
            [f"log_dir={log_dir}", f"llm={spec.model} @ {spec.base_url}", description],
            lambda: build_method(spec, log_dir).run(),
        )
    return run_dir


def baseline_run_config(
    spec,
    run_dir: Path,
    run_name: str,
    method: str,
    method_params: dict[str, Any],
    *,
    task_config: dict[str, Any] | None = None,
    llm_options: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if task_config is None:
        _, task_config = build_task(spec.task, spec.eval_workers)
    if spec.task in CO_TASKS:
        method_params = {
            **method_params,
            "evaluation_seeds": [EVALUATION_SEED],
            "eval_timeout_seconds": INSTANCE_LIMITS[spec.task],
            "function_seconds": FUNCTION_LIMITS[spec.task],
            "function_budget_scope": "mean_instance",
            "eval_workers": 1 if spec.eval_workers is None else spec.eval_workers,
            "scheduler_socket": scheduler_socket(),
        }
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "budget": method_params["max_sample_nums"],
        "objective": "min",
        "score_metric": "raw_objective",
        "native_objective": "min" if spec.task in NATIVE_MINIMIZE else "max",
        "budget_axis": "样本次数",
        "run_dir": str(run_dir),
        "run_name": run_name,
        "task": spec.task,
        "method": method,
        "repeat": spec.repeat,
        "backend": spec.backend,
        "seed": spec.seed,
        "llm": llm_payload(
            base_url=spec.base_url,
            model=spec.model,
            no_proxy=spec.no_proxy,
            max_tokens=spec.output_tokens,
            **(llm_options or {}),
        ),
        "task_eval": task_config,
        "method_params": method_params,
        "evaluation_condition": "shared",
        **(extra or {}),
    }
    socket = os.environ.get('TRACEAAD_MODEL_SCHEDULER_SOCKET')
    if socket and spec.base_url.rstrip('/') not in {BACKENDS[k].base_url.rstrip('/') for k in HOSTED_BACKENDS}:
        from core.scheduling import scheduler_status

        payload['model_routing'] = {'policy': 'shared', 'scheduler_socket': socket,
                                    'endpoints': scheduler_status(socket)['endpoints']}
    return payload
