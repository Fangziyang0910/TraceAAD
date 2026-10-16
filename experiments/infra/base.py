"""Shared plumbing for ``experiments/runners/*`` entries.

Mechanical infrastructure common to every method entry: backend profiles,
LLM client construction, task evaluation construction, run-directory layout,
tmux log redirection, and queue bookkeeping. Method-specific
parameters and the search method itself stay in each runner package.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from benchmarks.tasks import ALL_TASKS, CO_TASKS, TASKS, TASK_SHORT, SUITES, DEFAULT_WORKERS, INSTANCE_LIMITS, training_task
from .env import resolve_llm_api_key
from core.llm import OpenAIAPI

TaskName = Literal[
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
    "vrptw_construct",
    "fssp_gls",
    "graph_colouring",
    "jssp_construct",
]
BackendName = Literal["deepseek", "local", "server1", "server3", "server3b"]

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
    # Hosted gateway; key DEEPSEEK_API_KEY in .env. 24 concurrent requests tested.
    "deepseek": BackendProfile(
        base_url="http://183.36.243.124:33333/v1",
        model="deepseek-v4.1-flash",
        no_proxy="183.36.243.124,localhost,127.0.0.1,::1",
        capacity=24,
    ),
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

# Hosted models are called directly; a host scheduler serves only its own GPU endpoints.
HOSTED_BACKENDS = ("deepseek",)
DEFAULT_BACKEND: BackendName = 'deepseek'

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

# Models whose own settings replace the Qwen profiles above. The gateway has no
# /tokenize, so prompts are counted with the released tokenizer; the format is
# the published chat format, including roles and the thinking effort prefix.
# Sampling follows the model card.
MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "deepseek-v4.1-flash": {
        "sampling": {"temperature": 1.0, "top_p": 0.95},
        "enable_thinking": False,
        "reasoning_effort": "high",
        "tokenizer": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt_format": "<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜></think>",
        # The gateway's /models omits it; model card 1M, 78k-token prompt verified.
        "context": 1048576,
    },
}


def sampling_controls(enable_thinking: bool, model: str | None = None, **overrides: float | None) -> dict[str, float]:
    """The model's or mode's profile with any method-specified temperature/top_p/top_k."""
    settings = MODEL_SETTINGS.get(model or "")
    controls = dict(settings["sampling"] if settings else SAMPLING_PROFILES[bool(enable_thinking)])
    controls.update({k: v for k, v in overrides.items() if v is not None})
    return controls


LLM_TIMEOUT_SECONDS = 600
# ACO parallelism for cross-scale held-out only; training and the primary test
# run instances in turn because the time limit is a per-instance budget.
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
    enable_thinking: bool | None = None,
) -> OpenAIAPI:
    set_no_proxy(no_proxy)
    settings = MODEL_SETTINGS.get(model, {})
    if enable_thinking is None:
        enable_thinking = settings.get("enable_thinking", False)
    controls = sampling_controls(enable_thinking, model, temperature=temperature, top_p=top_p, top_k=top_k)
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
        reasoning_effort=settings.get("reasoning_effort") if enable_thinking else None,
        tokenizer=settings.get("tokenizer"),
        prompt_format=settings.get("prompt_format", "{}"),
    )


def scheduler_socket():
    """The host CPU scheduler a baseline's instances request cores from, if any."""
    return os.environ.get("TRACEAAD_SCHEDULER_SOCKET") or None


def build_task(task, eval_workers=None, *, condition='shared'):
    evaluation, settings = training_task(task, eval_workers, condition=condition)
    if task in CO_TASKS:
        evaluation._instance_execution = {"timeout_seconds": INSTANCE_LIMITS[task],
                                          "function_budget_scope": "mean_instance",
                                          "n_workers": 1 if eval_workers is None else eval_workers,
                                          "scheduler_socket": scheduler_socket(),
                                          "record_path": os.environ.get("TRACEAAD_EVALUATION_LOG") or None}
    return evaluation, settings


def llm_payload(
    *,
    base_url: str,
    model: str,
    no_proxy: str,
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    settings = MODEL_SETTINGS.get(model, {})
    if enable_thinking is None:
        enable_thinking = settings.get("enable_thinking", False)
    controls = sampling_controls(enable_thinking, model, temperature=temperature, top_p=top_p, top_k=top_k)
    payload: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "timeout": LLM_TIMEOUT_SECONDS,
        "max_tokens": max_tokens,
        "temperature": controls["temperature"],
        "top_p": controls["top_p"],
        "top_k": controls.get("top_k"),
        "enable_thinking": enable_thinking,
        "reasoning_effort": settings.get("reasoning_effort") if enable_thinking else None,
        "sampling": controls,
        "no_proxy": no_proxy,
        "api_key_configured": resolve_llm_api_key(base_url=base_url) != "EMPTY",
    }
    return payload


def write_run_config(run_dir: Path, payload: dict[str, Any]) -> None:
    from traceaad.storage import write_json
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
