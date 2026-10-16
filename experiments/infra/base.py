"""Model settings, task construction and experiment file handling."""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.tasks import (
    CO_TASKS,
    INSTANCE_LIMITS,
    training_task,
)
from .env import resolve_llm_api_key
from core.llm import OpenAIAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "experiments_result"


@dataclass(frozen=True, slots=True)
class BackendProfile:
    base_url: str
    model: str
    no_proxy: str
    capacity: int = 3


BACKENDS: dict[str, BackendProfile] = {
    # Hosted gateway; key DEEPSEEK_API_KEY in .env. 24 concurrent requests tested.
    "deepseek": BackendProfile(
        base_url="http://183.36.243.124:33333/v1",
        model="deepseek-v4.1-flash",
        no_proxy="183.36.243.124,localhost,127.0.0.1,::1",
        capacity=24,
    ),
    "local": BackendProfile(
        base_url="http://127.0.0.1:8001/v1", model="Qwen3.8-27B", no_proxy="127.0.0.1,localhost,::1"
    ),
    "server1": BackendProfile(
        base_url="http://222.201.145.8:8080/v1",
        model="qwen3.8-27b-awq",
        no_proxy="222.201.145.8,localhost,127.0.0.1,::1",
        capacity=6,
    ),
    "server3": BackendProfile(
        base_url="http://222.201.145.6:8000/v1",
        model="qwen3.8-27b-awq",
        no_proxy="222.201.145.6,localhost,127.0.0.1,::1",
        capacity=9,
    ),
    "server3b": BackendProfile(
        base_url="http://222.201.145.6:8001/v1",
        model="qwen3.8-27b-awq",
        no_proxy="222.201.145.6,localhost,127.0.0.1,::1",
        capacity=9,
    ),
}

# Hosted models are called directly; a host scheduler serves only its own GPU endpoints.
HOSTED_BACKENDS = ("deepseek",)
DEFAULT_BACKEND: str = "deepseek"

# Qwen3.8-27B's official sampling settings per mode (model card). Every
# control is sent explicitly: left unset, vLLM fills it from the model's
# generation_config.json and llama.cpp from its CLI defaults, so the "same"
# request could sample differently per server. vLLM reads
# repetition_penalty and llama.cpp repeat_penalty; each ignores the other.
# Searches run with thinking disabled, so the non-thinking profile is the
# default. Until 2026-09-30 all runs used the thinking profile (temperature
# 1.0, top_p 0.95, no presence penalty) with thinking disabled.
SAMPLING_PROFILES: dict[bool, dict[str, float]] = {
    False: {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "repeat_penalty": 1.0,
    },
    True: {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "repeat_penalty": 1.0,
    },
}

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
    }
}


def sampling_controls(enable_thinking: bool, model: str | None = None, **overrides: float | None) -> dict[str, float]:
    """The model's or mode's profile with any method-specified temperature/top_p/top_k."""
    settings = MODEL_SETTINGS.get(model or "")
    controls = dict(settings["sampling"] if settings else SAMPLING_PROFILES[bool(enable_thinking)])
    controls.update({k: v for k, v in overrides.items() if v is not None})
    return controls


LLM_TIMEOUT_SECONDS = 600


def resolve_backend(backend: str, base_url: str | None, model: str | None, no_proxy: str | None) -> BackendProfile:
    profile = BACKENDS[backend]
    return replace(
        profile,
        base_url=base_url or profile.base_url,
        model=model or profile.model,
        no_proxy=no_proxy or profile.no_proxy,
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def model_options(model, max_tokens, *, temperature=None, top_p=None, top_k=None, enable_thinking=None):
    """One set of options for requests and saved experiment settings."""
    settings = MODEL_SETTINGS.get(model, {})
    if enable_thinking is None:
        enable_thinking = settings.get("enable_thinking", False)
    controls = sampling_controls(enable_thinking, model, temperature=temperature, top_p=top_p, top_k=top_k)
    return {
        "model": model,
        "timeout": LLM_TIMEOUT_SECONDS,
        "max_tokens": max_tokens,
        "temperature": controls.pop("temperature"),
        "top_p": controls.pop("top_p"),
        "extra_body": controls,
        "enable_thinking": enable_thinking,
        "reasoning_effort": settings.get("reasoning_effort") if enable_thinking else None,
        "tokenizer": settings.get("tokenizer"),
        "prompt_format": settings.get("prompt_format", "{}"),
    }


def build_llm_client(*, base_url, model, no_proxy, max_tokens, **overrides):
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = no_proxy
    return OpenAIAPI(
        base_url=base_url,
        api_key=resolve_llm_api_key(base_url=base_url),
        **model_options(model, max_tokens, **overrides),
    )


def scheduler_socket():
    """The host CPU scheduler a baseline's instances request cores from, if any."""
    return os.environ.get("TRACEAAD_SCHEDULER_SOCKET") or None


def build_task(task, eval_workers=None, *, condition="shared"):
    evaluation, settings = training_task(task, eval_workers, condition=condition)
    if task in CO_TASKS:
        evaluation._instance_execution = {
            "timeout_seconds": INSTANCE_LIMITS[task],
            "n_workers": 1 if eval_workers is None else eval_workers,
            "scheduler_socket": scheduler_socket(),
            "record_path": os.environ.get("TRACEAAD_EVALUATION_LOG") or None,
        }
    return evaluation, settings


def llm_payload(*, base_url, model, no_proxy, max_tokens, **overrides):
    options = model_options(model, max_tokens, **overrides)
    controls = options.pop("extra_body")
    options.pop("tokenizer")
    options.pop("prompt_format")
    return {
        **options,
        "base_url": base_url,
        "no_proxy": no_proxy,
        "sampling": {"temperature": options["temperature"], "top_p": options["top_p"], **controls},
        "top_k": controls.get("top_k"),
        "api_key_configured": resolve_llm_api_key(base_url=base_url) != "EMPTY",
    }


def resolve_run_dir(experiments_root: Path, run_name: str | None) -> tuple[Path, str]:
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = experiments_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name


def run_in_tmux_log(run_dir: Path, header: list[str], body: Callable[[], None]) -> None:
    with (run_dir / "tmux_run.log").open("a", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"run_dir={run_dir}", flush=True)
            for line in header:
                print(line, flush=True)
            body()
