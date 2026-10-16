from __future__ import annotations

import json
import multiprocessing
import os
import queue
import resource
import signal
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal

from .code import Program, parse_program
from .scheduling import SchedulerError


@dataclass(frozen=True)
class EvaluationOutcome:
    """Result and a compact reason when generated code cannot be evaluated."""

    result: Any | None
    failure_kind: str | None = None
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None
    cpu_seconds: float | None = None
    instance_scores: tuple[float, ...] = ()


class InvalidEvaluationResult(Exception):
    """The task rejected the generated solution."""


# Every method evaluates a candidate from the same Python/NumPy random state.
EVALUATION_SEED = 730241


@contextmanager
def seeded_random_state(seed=EVALUATION_SEED):
    import random

    import numpy as np

    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class Evaluation(ABC):
    @contextmanager
    def program_context(self, source, function_name, **kwargs):
        """Run the source as supplied, from the fixed evaluation random state."""
        with seeded_random_state():
            yield source, self, kwargs

    def __init__(
        self,
        template_program: str | Program,
        task_description: str = "",
        timeout_seconds: int | float = None,
        *,
        safe_evaluate: bool = True,
        daemon_eval_process: bool = False,
        fork_proc: Literal["auto"] | bool = "auto",
    ):
        """Configure the task template and isolated execution."""
        self.template_program = template_program
        self.task_description = task_description
        self.timeout_seconds = timeout_seconds
        self.safe_evaluate = safe_evaluate
        self.daemon_eval_process = daemon_eval_process
        self.fork_proc = fork_proc

    @abstractmethod
    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        """Return minimized fitness; use a negative value for maximization objectives."""
        raise NotImplementedError("Must provide a evaluator for a function.")


def set_kill_with_parent() -> None:
    """Terminate orphan evaluation workers on Linux."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, int(signal.SIGKILL))
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except OSError:
        return


def _enter_eval_session() -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    set_kill_with_parent()


def _descendant_pids(pid: int) -> list[int]:
    import psutil

    try:
        return [child.pid for child in psutil.Process(pid).children(recursive=True)]
    except psutil.NoSuchProcess, psutil.AccessDenied:
        return []


def _own_cpu_seconds() -> float:
    usage = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(u.ru_utime + u.ru_stime for u in usage)


def _signal_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except OSError:
        return


def _stop_eval_process(process: multiprocessing.Process) -> None:
    """Terminate the evaluator and every nested ACO / spawn worker."""
    pid = process.pid
    if pid is None:
        return
    pgid = None
    if hasattr(os, "getpgid"):
        try:
            candidate = os.getpgid(pid)
        except OSError:
            candidate = None
        else:
            if candidate == pid:
                pgid = candidate
    descendants = [] if pgid is not None else _descendant_pids(pid)

    def _blast(sig: int) -> None:
        if pgid is not None and hasattr(os, "killpg"):
            try:
                os.killpg(pgid, sig)
                return
            except OSError:
                pass
        for child in descendants:
            _signal_pid(child, sig)
        _signal_pid(pid, sig)

    _blast(signal.SIGTERM)
    process.join(timeout=5)
    if pgid is None:
        descendants = _descendant_pids(pid)
    _blast(signal.SIGKILL)
    process.join(timeout=5)


_RECORD_LOCK = threading.Lock()


def _append_evaluation(path, program, record):
    """Keep baseline scores, failures and timing in the evaluation log."""
    failed = next(
        (r for r in record["instances"] if r["failure_kind"] not in (None, "cancelled")), None
    )
    row = {
        "code": program,
        "valid": record["valid"],
        "score": record["score"],
        "failure_kind": record["failure_kind"],
        "error_type": record["error_type"],
        "error": (record["error"] or "")[:300] or None,
        "instance_index": failed["instance_index"] if failed else None,
        "instance_function_seconds": failed["function_seconds"] if failed else None,
        "instance_function_cpu_seconds": failed["function_cpu_seconds"] if failed else None,
        "instance_seconds": failed["seconds"] if failed else None,
        "cpu_id": failed.get("cpu_id") if failed else None,
        "seconds": record["seconds"],
        "function_seconds": record["function_seconds"],
        "function_cpu_seconds": record["function_cpu_seconds"],
        "mean_function_cpu_seconds": record["mean_function_cpu_seconds"],
        "function_budget_scope": record["function_budget_scope"],
        "n_instances": record["n_instances"],
        "timeout_seconds": record["timeout_seconds"],
        "function_seconds_limit": record["function_seconds_limit"],
        "instance_scores": [r["score"] for r in record["instances"]] if record["valid"] else [],
    }
    with _RECORD_LOCK, open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(row) + "\n")


class SecureEvaluator:
    """Run a generated program and retain its score or failure reason."""

    def __init__(self, evaluator: Evaluation, debug_mode=False):
        self._evaluator = evaluator
        self._debug_mode = debug_mode
        self.scheduler_error = None
        self._context = multiprocessing.get_context(
            "spawn" if evaluator.fork_proc is False else "fork"
        )

    def evaluate_program(self, program: str | Program, **kwargs):
        return self.evaluate_program_with_details(program, **kwargs).result

    def _wait(self, result_queue) -> EvaluationOutcome:
        timeout = self._evaluator.timeout_seconds
        try:
            return result_queue.get(timeout=timeout)
        except queue.Empty:
            if self._debug_mode:
                print(f"DEBUG: the evaluation time exceeds {timeout}s.")
            return EvaluationOutcome(
                result=None,
                failure_kind="timeout",
                error_type="TimeoutError",
                error=f"evaluation exceeded {timeout}s",
            )

    def evaluate_program_with_details(self, program: str | Program, **kwargs) -> EvaluationOutcome:
        try:
            program_str = str(program)
            if execution := getattr(self._evaluator, "_instance_execution", None):
                from traceaad.instance_evaluation import InstanceProgramEvaluator

                if execution.get("function_budget_scope", "mean_instance") != "mean_instance":
                    raise ValueError(
                        "historical instance CPU budgets require the saved source snapshot"
                    )
                options = {
                    name: value
                    for name, value in execution.items()
                    if name not in {"record_path", "function_budget_scope"}
                }
                evaluator = InstanceProgramEvaluator(
                    self._evaluator, seeds=(kwargs.get("seed", EVALUATION_SEED),), **options
                )
                result = evaluator.evaluate(program_str, "")
                record = result["evaluations"][-1]
                if execution.get("record_path"):
                    _append_evaluation(execution["record_path"], program_str, record)
                return EvaluationOutcome(
                    result["fitness"],
                    **{
                        name: record[name]
                        for name in ("failure_kind", "error_type", "error", "traceback")
                    },
                    cpu_seconds=sum(r["cpu_seconds"] for r in result["evaluations"]),
                    instance_scores=tuple(r["score"] for r in record["instances"])
                    if result["fitness"] is not None
                    else (),
                )
            function_name = self._target_function_name()

            if self._debug_mode:
                print(f"DEBUG: evaluated program:\n{program_str}\n")

            if self._evaluator.safe_evaluate:
                result_queue = self._context.Queue()
                process = self._context.Process(
                    target=self._evaluate_in_safe_process_with_details,
                    args=(program_str, function_name, result_queue),
                    kwargs=kwargs,
                    daemon=self._evaluator.daemon_eval_process,
                )
                process.start()
                try:
                    outcome = self._wait(result_queue)
                finally:
                    _stop_eval_process(process)
                return outcome
            else:
                return self._evaluate_with_details(program_str, function_name, **kwargs)
        except SchedulerError as exc:
            self.scheduler_error = exc
            raise
        except Exception as e:
            if self._debug_mode:
                print("DEBUG: Exception occurred in evaluate_program:")
                traceback.print_exc()
            return self._failure("prepare_error", e)

    def evaluate_program_record_time(self, program: str | Program, **kwargs):
        outcome, seconds = self.evaluate_program_record_time_with_details(program, **kwargs)
        return outcome.result, seconds

    def evaluate_program_record_time_with_details(
        self, program: str | Program, **kwargs
    ) -> tuple[EvaluationOutcome, float]:
        evaluate_start = time.monotonic()
        outcome = self.evaluate_program_with_details(program, **kwargs)
        return outcome, time.monotonic() - evaluate_start

    def _target_function_name(self) -> str:
        source = self._evaluator.template_program
        template = source if isinstance(source, Program) else parse_program(source)
        if template is None or len(template.functions) != 1:
            raise ValueError("evaluation template must define one target function")
        return template.functions[0].name

    def _evaluate_in_safe_process_with_details(
        self, program_str: str, function_name: str, result_queue: multiprocessing.Queue, **kwargs
    ) -> None:
        _enter_eval_session()
        try:
            outcome = self._evaluate_with_details(program_str, function_name, **kwargs)
        except Exception as exc:
            outcome = self._failure("prepare_error", exc)
        result_queue.put(replace(outcome, cpu_seconds=_own_cpu_seconds()))

    def _evaluate_with_details(self, program_str: str, function_name, **kwargs):
        with self._evaluator.program_context(program_str, function_name, **kwargs) as (
            code,
            task,
            task_kwargs,
        ):
            function = None
            if not getattr(task, "executes_source", False):
                try:
                    namespace = {}
                    exec(code, namespace)
                    function = namespace[function_name]
                except Exception as exc:
                    return self._failure("exec_error", exc)
            try:
                result = task.evaluate_program(code, function, **task_kwargs)
                if result is None:
                    raise InvalidEvaluationResult("evaluator returned None")
                return EvaluationOutcome(result=result)
            except InvalidEvaluationResult as exc:
                return EvaluationOutcome(
                    result=None,
                    failure_kind="invalid_result",
                    error_type="InvalidEvaluationResult",
                    error=str(exc),
                )
            except Exception as exc:
                if self._debug_mode:
                    traceback.print_exc()
                return self._failure("runtime_error", exc)

    @staticmethod
    def _failure(kind: str, exc: Exception) -> EvaluationOutcome:
        message = str(exc)
        if len(message) > 20000:
            message = message[:19997] + "..."
        return EvaluationOutcome(
            result=None,
            failure_kind=kind,
            error_type=type(exc).__name__,
            error=message,
            traceback=traceback.format_exc(),
        )
