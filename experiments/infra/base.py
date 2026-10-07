"""Shared plumbing for ``experiments/runners/*`` entries.

Mechanical infrastructure common to every method entry: backend profiles,
LLM client construction, task evaluation construction, run-directory layout,
tmux log redirection, and the launch-side slot scheduler. Method-specific
parameters and the search method itself stay in each runner package.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np

from benchmarks.tasks import ALL_TASKS, TASKS, TASK_SHORT, SUITES, DEFAULT_WORKERS, training_task
from .env import resolve_llm_api_key
from core.llm import OpenAIAPI

TaskName = Literal[
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
    "vrptw_construct",
    "fssp_gls",
    "mdmkp_search",
    "graph_colouring",
    "set_cover_construct",
    "jssp_construct",
]
BackendName = Literal["local", "server1", "server3", "server3b"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "experiments_result"


@dataclass(frozen=True, slots=True)
class BackendProfile:
    base_url: str
    model: str
    no_proxy: str
    capacity: int = 3
    label: str | None = None
    marker: str | None = None


BACKENDS: dict[BackendName, BackendProfile] = {
    "local": BackendProfile(
        base_url="http://127.0.0.1:8001/v1",
        model="Qwen3.8-27B",
        no_proxy="127.0.0.1,localhost,::1",
    ),
    "server1": BackendProfile(
        base_url="http://222.201.145.8:8080/v1",
        model="qwen3.8-27b-awq",
        no_proxy="222.201.145.8,localhost,127.0.0.1,::1",
        capacity=6, marker="222.201.145.8",
    ),
    "server3": BackendProfile(
        base_url="http://222.201.145.6:8000/v1",
        model="qwen3.8-27b-awq",
        no_proxy="222.201.145.6,localhost,127.0.0.1,::1", capacity=9, label="server3-1",
    ),
    "server3b": BackendProfile(
        base_url="http://222.201.145.6:8001/v1",
        model="qwen3.8-27b-awq",
        no_proxy="222.201.145.6,localhost,127.0.0.1,::1", capacity=9, label="server3-2",
    ),
}

BACKEND_CAPACITY = {name: profile.capacity for name, profile in BACKENDS.items()}
BACKEND_DISPLAY_NAMES = {name: profile.label or name for name, profile in BACKENDS.items()}
BACKEND_MARKERS = {name: (profile.marker or urlparse(profile.base_url).netloc,) for name, profile in BACKENDS.items()}
PRIMARY_BACKENDS = ('server3', 'server3b', 'server1', 'local')

# Qwen3.8-27B's official sampling settings per mode (model card). Every
# control is sent explicitly: left unset, vLLM fills it from the model's
# generation_config.json and llama.cpp from its CLI defaults, so the "same"
# request could sample differently per server. vLLM reads
# repetition_penalty and llama.cpp repeat_penalty; each ignores the other.
# Searches run with thinking disabled, so the non-thinking profile is the
# default. Until 2026-09-30 all runs used the thinking profile (temperature
# 1.0, top_p 0.95, no presence penalty) with thinking disabled.
SAMPLING_PROFILES: dict[bool, dict[str, float]] = {
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "frequency_penalty": 0.0,
            "repetition_penalty": 1.0, "repeat_penalty": 1.0},
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
           "presence_penalty": 0.0, "frequency_penalty": 0.0,
           "repetition_penalty": 1.0, "repeat_penalty": 1.0},
}
_EXPLICIT = ("temperature", "top_p", "top_k")


def sampling_controls(enable_thinking: bool, **overrides: float | None) -> dict[str, float]:
    """The mode's profile with any method-specified temperature/top_p/top_k."""
    controls = dict(SAMPLING_PROFILES[bool(enable_thinking)])
    controls.update({k: v for k, v in overrides.items() if v is not None})
    return controls


LLM_TIMEOUT_SECONDS = 600
# Local ACO parallelism only; seeded scores do not depend on this count.
# Sized for 18 concurrent searches on a 32-core host: 4 workers cover
# CVRP's 10 train instances in three rounds without the old 10-worker pileup.
DEFAULT_ACO_EVAL_WORKERS = DEFAULT_WORKERS


# ---------------------------------------------------------------------------
# run side
# ---------------------------------------------------------------------------


def resolve_backend(
    backend: BackendName,
    base_url: str | None,
    model: str | None,
    no_proxy: str | None,
) -> BackendProfile:
    profile = BACKENDS[backend]
    return replace(profile, base_url=base_url or profile.base_url,
                   model=model or profile.model, no_proxy=no_proxy or profile.no_proxy)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def set_no_proxy(no_proxy: str) -> None:
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy


def build_llm_client(
    *,
    base_url: str,
    model: str,
    no_proxy: str,
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool = False,
    chars_per_token: float | None = None,
) -> OpenAIAPI:
    set_no_proxy(no_proxy)
    controls = sampling_controls(enable_thinking, temperature=temperature, top_p=top_p, top_k=top_k)
    temperature, top_p = controls.pop("temperature"), controls.pop("top_p")
    extra_body = controls
    return OpenAIAPI(
        base_url=base_url,
        api_key=resolve_llm_api_key(base_url=base_url),
        model=model,
        timeout=LLM_TIMEOUT_SECONDS,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
        enable_thinking=enable_thinking,
        chars_per_token=chars_per_token,
    )


def build_task(task, eval_workers=None, *, condition='shared'):
    return training_task(task, eval_workers, condition=condition)


def llm_payload(
    *,
    base_url: str,
    model: str,
    no_proxy: str,
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool = False,
    chars_per_token: float | None = None,
) -> dict[str, Any]:
    controls = sampling_controls(enable_thinking, temperature=temperature, top_p=top_p, top_k=top_k)
    payload: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "timeout": LLM_TIMEOUT_SECONDS,
        "max_tokens": max_tokens,
        "temperature": controls["temperature"],
        "top_p": controls["top_p"],
        "top_k": controls["top_k"],
        "enable_thinking": enable_thinking,
        "sampling": controls,
        "no_proxy": no_proxy,
        "api_key_configured": resolve_llm_api_key(base_url=base_url) != "EMPTY",
    }
    if chars_per_token is not None:
        payload["chars_per_token"] = chars_per_token
    return payload


def write_run_config(run_dir: Path, payload: dict[str, Any]) -> None:
    from traceaad.common.storage import RESULT_FORMAT, write_json
    payload = {"result_format": RESULT_FORMAT, **payload}
    write_json(run_dir / "run_config.json", payload)


def resolve_run_dir(experiments_root: Path, run_name: str | None) -> tuple[Path, str]:
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = experiments_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name


def run_in_tmux_log(
    run_dir: Path,
    log_dir: Path,
    header: list[str],
    body: Callable[[], None],
) -> None:
    with (run_dir / "tmux_run.log").open(
        "a", encoding="utf-8", buffering=1
    ) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"run_dir={run_dir}", flush=True)
            for line in header:
                print(line, flush=True)
            body()


# ---------------------------------------------------------------------------
# launch side
# ---------------------------------------------------------------------------

# Cap automatic relaunch after backend outages to avoid retry storms.
MAX_RETRIES = 5


@dataclass(frozen=True, slots=True)
class LaunchItem:
    task: TaskName
    repeat: int
    backend: BackendName | None
    session: str
    run_name: str
    run_dir: Path
    seed: int
    module: str
    extra_args: tuple[str, ...] = ()

    def with_backend(self, backend: BackendName) -> LaunchItem:
        return replace(self, backend=backend)

    def command(self) -> tuple[str, ...]:
        if self.backend is None:
            raise ValueError(f"backend not assigned for {self.run_name}")
        return (
            "uv",
            "run",
            "python",
            "-m",
            self.module,
            "--task",
            self.task,
            "--backend",
            self.backend,
            "--repeat",
            str(self.repeat),
            "--seed",
            str(self.seed),
            "--run-name",
            self.run_name,
            *self.extra_args,
        )


def build_launch_plan(
    args: argparse.Namespace,
    *,
    module: str,
    method: str,
) -> list[LaunchItem]:
    plan = []
    for repeat in range(1, args.repeats + 1):
        for task in (getattr(args, "tasks", None) or SUITES[getattr(args, "suite", "legacy")]):
            short = TASK_SHORT[task]
            run_name = f"{args.batch}_{short}_{method}_rep{repeat}"
            session = f"{args.session_prefix}_{short}_r{repeat}"
            run_dir = RESULTS_ROOT / method / task / run_name
            plan.append(
                LaunchItem(
                    task=task,
                    repeat=repeat,
                    backend=None,
                    session=session,
                    run_name=run_name,
                    run_dir=run_dir,
                    seed=repeat - 1,
                    module=module,
                )
            )
    return plan


def _process_cmdlines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "args="],
        check=True,
        capture_output=True,
        text=True,
    )
    lines: set[str] = set()
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or "uv run" in text or "python" not in text:
            continue
        if "experiments." not in text and "run_experiment" not in text:
            continue
        if ".launch" in text:
            # Capacity watchers orchestrate jobs but never call an LLM backend.
            continue
        # SecureEvaluator workers inherit the parent's complete command line.
        # Count one logical LLM client command, not every forked evaluator
        # process. Distinct runs and shards retain distinct CLI arguments.
        lines.add(text)
    return sorted(lines)


def detect_backend(cmdline: str) -> BackendName | None:
    """Identify backend from a process cmdline.

    Prefer the ``--backend`` token (exact name). Fall back to host:port markers
    for older processes that embed the service URL.
    """
    try:
        tokens = shlex.split(cmdline)
    except ValueError:
        tokens = cmdline.split()
    try:
        name = tokens[tokens.index("--backend") + 1]
    except (ValueError, IndexError):
        name = None
    if name in BACKENDS:
        return name  # type: ignore[return-value]
    for backend, markers in BACKEND_MARKERS.items():
        if any(marker in cmdline for marker in markers):
            return backend
    return None


def count_backend_usage() -> dict[BackendName, int]:
    counts: dict[BackendName, int] = {name: 0 for name in BACKEND_CAPACITY}
    for cmdline in _process_cmdlines():
        matched = detect_backend(cmdline)
        if matched is not None:
            counts[matched] += 1
    return counts


def free_slots() -> dict[BackendName, int]:
    usage = count_backend_usage()
    return {
        backend: max(0, BACKEND_CAPACITY[backend] - usage[backend])
        for backend in BACKEND_CAPACITY
    }


def _summary_status(item: LaunchItem) -> str | None:
    summary = item.run_dir / "summary.json"
    if not summary.exists():
        return None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def item_is_done(item: LaunchItem) -> bool:
    return _summary_status(item) == "finished"


def item_is_failed(item: LaunchItem) -> bool:
    return _summary_status(item) in {"error", "aborted", "interrupted"}


def item_is_running(item: LaunchItem) -> bool:
    from .launcher import is_session_alive
    return is_session_alive(item.session)


def _retry_candidate(item: LaunchItem, retry: int) -> LaunchItem:
    return replace(item, backend=None, session=f"{item.session}_retry{retry}",
                   run_name=f"{item.run_name}_retry{retry}",
                   run_dir=item.run_dir.parent / f"{item.run_name}_retry{retry}")


def item_active_attempt(item: LaunchItem) -> LaunchItem | None:
    """Return the base item or latest retry that is currently running."""
    if item_is_running(item):
        return item
    for retry in range(2, MAX_RETRIES + 1):
        candidate = _retry_candidate(item, retry)
        if item_is_running(candidate):
            return candidate
    # Also accept already-launched high retries from the previous storm.
    retry = MAX_RETRIES + 1
    while True:
        candidate = _retry_candidate(item, retry)
        if item_is_running(candidate):
            return candidate
        if not candidate.run_dir.exists():
            return None
        retry += 1
        if retry > 50:
            return None


def select_backend(remaining: dict[BackendName, int]) -> BackendName | None:
    """Pick the primary backend with the most free slots; ties keep listed order."""
    candidates = [name for name in PRIMARY_BACKENDS if remaining.get(name, 0) > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda name: remaining[name])


def assign_backends(pending: list[LaunchItem]) -> list[LaunchItem]:
    remaining = dict(free_slots())
    assigned: list[LaunchItem] = []
    for item in pending:
        chosen = select_backend(remaining)
        if chosen is None:
            break
        remaining[chosen] -= 1
        assigned.append(item.with_backend(chosen))
    return assigned


def validate_new_items(items: list[LaunchItem]) -> None:
    sessions = [item.session for item in items]
    run_dirs = [item.run_dir for item in items]
    if len(sessions) != len(set(sessions)):
        raise ValueError("tmux session names must be unique")
    if len(run_dirs) != len(set(run_dirs)):
        raise ValueError("run directories must be unique")
    collisions = [str(path) for path in run_dirs if path.exists()]
    if collisions:
        raise FileExistsError(f"run directories already exist: {collisions}")
    active = [item.session for item in items if item_is_running(item)]
    if active:
        raise RuntimeError(f"tmux sessions already exist: {active}")


def launch_items(items: list[LaunchItem], *, dry_run: bool) -> None:
    for item in items:
        printable = shlex.join(item.command())
        print(
            f"{item.backend:7s} {item.task:20s} rep={item.repeat} "
            f"session={item.session} command={printable}",
            flush=True,
        )
        if dry_run:
            continue
        from .launcher import launch_command
        launch_command(item.session, item.command())


def item_has_successful_result(item: LaunchItem) -> bool:
    if item_is_done(item):
        return True
    if not item_is_failed(item):
        return False
    retry = 2
    while True:
        candidate = _retry_candidate(item, retry)
        if item_is_done(candidate):
            return True
        if item_is_failed(candidate):
            retry += 1
            if retry > 50:
                return False
            continue
        return False


def pending_items(plan: list[LaunchItem]) -> list[LaunchItem]:
    pending = []
    for item in plan:
        if item_has_successful_result(item):
            continue
        if item_active_attempt(item) is not None:
            continue
        if item.run_dir.exists() and not item_is_failed(item):
            print(f"skip incomplete run_dir={item.run_dir}", flush=True)
            continue
        if item_is_failed(item):
            retry = 2
            while True:
                if retry > MAX_RETRIES:
                    print(
                        f"skip {item.run_name}: exceeded MAX_RETRIES={MAX_RETRIES}",
                        flush=True,
                    )
                    break
                candidate = _retry_candidate(item, retry)
                if item_is_done(candidate):
                    retry += 1
                    continue
                if item_is_running(candidate):
                    break
                if candidate.run_dir.exists() and not item_is_failed(candidate):
                    print(f"skip incomplete run_dir={candidate.run_dir}", flush=True)
                    break
                if item_is_failed(candidate):
                    retry += 1
                    continue
                pending.append(candidate)
                break
            continue
        pending.append(item)
    return pending


def fill_once(plan: list[LaunchItem], *, dry_run: bool) -> list[LaunchItem]:
    pending = pending_items(plan)
    assigned = assign_backends(pending)
    if not assigned:
        print(
            "no free slots or no pending runs; "
            f"free={free_slots()} pending={len(pending)}",
            flush=True,
        )
        return []
    if not dry_run:
        validate_new_items(assigned)
    launch_items(assigned, dry_run=dry_run)
    return assigned


def watch_and_fill(
    plan: list[LaunchItem],
    *,
    interval_sec: int,
    dry_run: bool,
    method_label: str,
) -> None:
    print(
        f"watching {method_label} batch; interval={interval_sec}s; "
        f"capacity={BACKEND_CAPACITY}",
        flush=True,
    )
    while True:
        remaining = pending_items(plan)
        running = sum(1 for item in plan if item_active_attempt(item) is not None)
        done = sum(1 for item in plan if item_has_successful_result(item))
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"done={done} running={running} pending={len(remaining)} "
            f"free={free_slots()}",
            flush=True,
        )
        if done == len(plan):
            print(f"all {method_label} runs finished", flush=True)
            return
        fill_once(plan, dry_run=dry_run)
        if dry_run:
            return
        time.sleep(interval_sec)


def add_launch_parser_args(
    parser: argparse.ArgumentParser,
    *,
    watch: bool = True,
    session_prefix: str = "batch",
) -> None:
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default=session_prefix)
    parser.add_argument("--dry-run", action="store_true")
    if watch:
        parser.add_argument(
            "--watch",
            action="store_true",
            help="keep filling free slots until the whole batch finishes",
        )
        parser.add_argument("--watch-interval", type=int, default=60)
