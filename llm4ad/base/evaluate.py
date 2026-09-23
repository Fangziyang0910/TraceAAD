# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .code import TextFunctionProgramConverter, Program


@dataclass(frozen=True)
class EvaluationOutcome:
    """Result and a compact reason when generated code cannot be evaluated."""

    result: Any | None
    failure_kind: str | None = None
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None


class InvalidEvaluationResult(Exception):
    """Evaluator-detected reason why a candidate construction yields no result.

    Task evaluations raise this instead of returning None when they can name
    the exact violated condition; SecureEvaluator maps it to an
    ``invalid_result`` outcome that carries the reason.
    """


class Evaluation(ABC):
    def __init__(
            self,
            template_program: str | Program,
            task_description: str = '',
            timeout_seconds: int | float = None,
            *,
            safe_evaluate: bool = True,
            daemon_eval_process: bool = False,
            fork_proc: Literal['auto'] | bool = 'auto'
    ):
        """Evaluation interface for executing generated code.
        Args:
            timeout_seconds     : Terminate the evaluation after timeout seconds.
            safe_evaluate       : Evaluate in safe mode using a new process. If is set to False,
                the evaluation will not be terminated after timeout seconds. The user should consider how to
                terminate evaluating in time.
            daemon_eval_process : Set the evaluate process as a daemon process. If set to True,
                you can not set new processes in the evaluator. Which means in self.evaluate_program(),
                you can not create new processes.
            fork_proc           : This arg is valid when safe_evaluate=True, which determines to 'fork' process or 'spawn' a safe process.
                If set to 'auto', the process creating method will depend on OS. Set to 'True' to use 'fork', 'False' to use 'spawn'.
        """
        self.template_program = template_program
        self.task_description = task_description
        self.timeout_seconds = timeout_seconds
        self.safe_evaluate = safe_evaluate
        self.daemon_eval_process = daemon_eval_process
        self.fork_proc = fork_proc

    @abstractmethod
    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        r"""Evaluate a given function. You can use compiled function (function_callable),
        as well as the original function strings for evaluation.
        Args:
            program_str: The function in string. You can _ignore this argument when implementation. (See below).
            callable_func: The callable heuristic function. You can call it using `callable_func(args, kwargs)`.
        Return:
            Returns the fitness value.
        """
        raise NotImplementedError('Must provide a evaluator for a function.')


def set_kill_with_parent() -> None:
    """Kill this process if its parent dies.

    Secure-eval subprocesses and ACO pool workers call this so a timeout or
    a killed search process cannot leave grandchildren attached to init.
    """
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
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    children_by_ppid: dict[int, list[int]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        child_pid = int(entry.name)
        try:
            text = (entry / "status").read_text()
        except OSError:
            continue
        ppid = None
        for line in text.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
                break
        if ppid is None:
            continue
        children_by_ppid.setdefault(ppid, []).append(child_pid)
    found: list[int] = []
    stack = list(children_by_ppid.get(pid, []))
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(children_by_ppid.get(current, []))
    return found


def _signal_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
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
        except (ProcessLookupError, PermissionError, OSError):
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
            except (ProcessLookupError, PermissionError, OSError):
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


class SecureEvaluator:
    def __init__(self,
                 evaluator: Evaluation,
                 debug_mode=False,
                 **kwargs):
        self._evaluator = evaluator
        self._debug_mode = debug_mode
        fork_proc = self._evaluator.fork_proc

        if self._evaluator.safe_evaluate:
            if fork_proc == 'auto':
                # force MacOS and Linux use 'fork' to generate new process
                if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
                    multiprocessing.set_start_method('fork', force=True)
            elif fork_proc is True:
                multiprocessing.set_start_method('fork', force=True)
            elif fork_proc is False:
                multiprocessing.set_start_method('spawn', force=True)

    def evaluate_program(self, program: str | Program, **kwargs):
        return self.evaluate_program_with_details(program, **kwargs).result

    def evaluate_program_with_details(
            self, program: str | Program, **kwargs
    ) -> EvaluationOutcome:
        try:
            program_str = str(program)
            function_name = self._target_function_name()

            if self._debug_mode:
                print(f'DEBUG: evaluated program:\n{program_str}\n')

            # safe evaluate
            if self._evaluator.safe_evaluate:
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=self._evaluate_in_safe_process_with_details,
                    args=(program_str, function_name, result_queue),
                    kwargs=kwargs,
                    daemon=self._evaluator.daemon_eval_process
                )
                process.start()

                try:
                    if self._evaluator.timeout_seconds is None:
                        outcome = result_queue.get()
                    else:
                        outcome = result_queue.get(
                            timeout=self._evaluator.timeout_seconds
                        )
                except queue.Empty:
                    if self._debug_mode:
                        print(
                            'DEBUG: the evaluation time exceeds '
                            f'{self._evaluator.timeout_seconds}s.'
                        )
                    outcome = EvaluationOutcome(
                        result=None,
                        failure_kind='timeout',
                        error_type='TimeoutError',
                        error=(
                            'evaluation exceeded '
                            f'{self._evaluator.timeout_seconds}s'
                        ),
                    )
                finally:
                    _stop_eval_process(process)
                return outcome
            else:
                return self._evaluate_with_details(
                    program_str, function_name, **kwargs
                )
        except Exception as e:
            if self._debug_mode:
                print("DEBUG: Exception occurred in evaluate_program:")
                traceback.print_exc()  # 这将打印完整红色报错信息
            return self._failure('prepare_error', e)

    def evaluate_program_record_time(self, program: str | Program, **kwargs):
        evaluate_start = time.time()
        result = self.evaluate_program(program, **kwargs)
        return result, time.time() - evaluate_start

    def evaluate_program_record_time_with_details(
            self, program: str | Program, **kwargs
    ) -> tuple[EvaluationOutcome, float]:
        evaluate_start = time.time()
        outcome = self.evaluate_program_with_details(program, **kwargs)
        return outcome, time.time() - evaluate_start

    def _target_function_name(self) -> str:
        source = self._evaluator.template_program
        template = (
            source
            if isinstance(source, Program)
            else TextFunctionProgramConverter.text_to_program(source)
        )
        if template is None or len(template.functions) != 1:
            raise ValueError('evaluation template must define one target function')
        return template.functions[0].name

    def _evaluate_in_safe_process_with_details(
            self,
            program_str: str,
            function_name: str,
            result_queue: multiprocessing.Queue,
            **kwargs,
    ) -> None:
        _enter_eval_session()
        try:
            all_globals_namespace = {}
            exec(program_str, all_globals_namespace)
            program_callable = all_globals_namespace[function_name]
        except Exception as exc:
            result_queue.put(self._failure('exec_error', exc))
            return

        try:
            res = self._evaluator.evaluate_program(program_str, program_callable, **kwargs)
            result_queue.put(self._outcome(res))
        except InvalidEvaluationResult as exc:
            result_queue.put(EvaluationOutcome(
                result=None,
                failure_kind='invalid_result',
                error_type='InvalidEvaluationResult',
                error=str(exc),
            ))
        except Exception as exc:
            if self._debug_mode:
                print("DEBUG: Exception occurred in evaluate_program:")
                traceback.print_exc()  # 这将打印完整红色报错信息
            result_queue.put(self._failure('runtime_error', exc))

    def _evaluate_with_details(self, program_str: str, function_name, **kwargs):
        try:
            all_globals_namespace = {}
            exec(program_str, all_globals_namespace)
            program_callable = all_globals_namespace[function_name]
        except Exception as exc:
            return self._failure('exec_error', exc)

        try:
            res = self._evaluator.evaluate_program(program_str, program_callable, **kwargs)
            return self._outcome(res)
        except InvalidEvaluationResult as exc:
            return EvaluationOutcome(
                result=None,
                failure_kind='invalid_result',
                error_type='InvalidEvaluationResult',
                error=str(exc),
            )
        except Exception as exc:
            if self._debug_mode:
                print("DEBUG: Exception occurred in evaluate_program:")
                traceback.print_exc()  # 这将打印完整红色报错信息
            return self._failure('runtime_error', exc)

    @staticmethod
    def _outcome(result: Any | None) -> EvaluationOutcome:
        if result is None:
            return EvaluationOutcome(
                result=None,
                failure_kind='invalid_result',
                error_type='InvalidEvaluationResult',
                error='evaluator returned None',
            )
        return EvaluationOutcome(result=result)

    @staticmethod
    def _failure(kind: str, exc: Exception) -> EvaluationOutcome:
        message = str(exc)
        if len(message) > 20000:
            message = message[:19997] + '...'
        return EvaluationOutcome(
            result=None,
            failure_kind=kind,
            error_type=type(exc).__name__,
            error=message,
            traceback=traceback.format_exc(),
        )
