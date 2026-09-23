from __future__ import annotations

import inspect
from typing import Any


def init_observability(target: Any, max_consecutive_sample_failures: int = 20) -> None:
    target._consecutive_sample_failures = 0
    target._max_consecutive_sample_failures = max(
        1, int(max_consecutive_sample_failures)
    )
    target._search_aborted = False


def reset_sample_failures(target: Any) -> None:
    target._consecutive_sample_failures = 0


def is_search_aborted(target: Any) -> bool:
    return bool(getattr(target, "_search_aborted", False))


def _profiler(target: Any):
    return getattr(target, "_profiler", None)


def log_llm_call(target: Any, **payload) -> None:
    logger = getattr(_profiler(target), "log_llm_call", None)
    if callable(logger):
        logger(**payload)


def log_event(target: Any, event: str | None = None, **payload) -> None:
    logger = getattr(_profiler(target), "log_method_event", None)
    if callable(logger):
        logger(event=event, **payload)


def log_state(target: Any, phase: str | None = None, **payload) -> None:
    logger = getattr(_profiler(target), "log_method_state", None)
    if callable(logger):
        logger(phase=phase, **payload)


def log_error(
    target: Any, stage_name: str, exc: Exception | None = None, **payload
) -> None:
    logger = getattr(_profiler(target), "log_error", None)
    if callable(logger):
        stage = payload.pop("stage", stage_name) or stage_name
        logger(stage, exc, **payload)


def record_sample_failure(
    method: Any,
    exc: Exception,
    *,
    stage: str = "sample",
    operator: str | None = None,
    sample_order: int | None = None,
    prompt: Any = None,
    messages: Any = None,
    counts_budget: bool = False,
    **payload,
) -> bool:
    if getattr(method, "_debug_mode", False):
        raise exc

    if sample_order is None:
        sample_order = getattr(method, "_tot_sample_nums", 0) + 1
    method._consecutive_sample_failures = (
        getattr(method, "_consecutive_sample_failures", 0) + 1
    )
    max_failures = getattr(method, "_max_consecutive_sample_failures", 20)
    error_message = str(exc)
    if len(error_message) > 1000:
        error_message = error_message[:997] + "..."

    common = {
        "stage": stage,
        "operator": operator,
        "sample_order": sample_order,
        "counts_budget": counts_budget,
        "error_type": type(exc).__name__,
        "error": error_message,
        "consecutive_failures": method._consecutive_sample_failures,
        "max_consecutive_failures": max_failures,
    }
    common.update(payload)
    event_payload = dict(common)
    if prompt is not None:
        common["prompt"] = prompt
    if messages is not None:
        common["messages"] = messages

    error_payload = dict(common)
    error_payload.pop("stage", None)
    log_error(method, stage, exc, **error_payload)
    log_event(method, event=f"{stage}_error", status="error", **event_payload)

    if method._consecutive_sample_failures >= max_failures:
        method._search_aborted = True
        log_event(
            method,
            event="search_aborted",
            status="aborted",
            reason="max_consecutive_sample_failures",
            consecutive_failures=method._consecutive_sample_failures,
            max_consecutive_failures=max_failures,
        )
    return bool(getattr(method, "_search_aborted", False))


def call_sampler_get_thought_and_function(sampler: Any, prompt: Any, **kwargs):
    method = getattr(sampler, "get_thought_and_function")
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method(prompt)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    usable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_kwargs or key in parameters
    }
    return method(prompt, **usable_kwargs)


def close_llm(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def close_sampler_llm(sampler: Any) -> None:
    llm = getattr(sampler, "llm", None)
    if llm is None:
        llm = getattr(sampler, "_sampler", None)
    close_llm(llm)


def shutdown_executor(executor: Any) -> None:
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(cancel_futures=True)
        except TypeError:
            shutdown()
        except Exception:
            pass


def finish_profiler(method: Any, *, status: str = "finished", **payload) -> None:
    profiler = _profiler(method)
    writer = getattr(profiler, "write_run_summary", None)
    if callable(writer):
        writer(
            status=status,
            method_sample_count=getattr(method, "_tot_sample_nums", None),
            search_aborted=getattr(method, "_search_aborted", False),
            consecutive_sample_failures=getattr(
                method, "_consecutive_sample_failures", None
            ),
            **payload,
        )
    finish = getattr(profiler, "finish", None)
    if callable(finish):
        finish()
