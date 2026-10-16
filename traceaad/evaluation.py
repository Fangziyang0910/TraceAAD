"""Seeded, isolated evaluation and one uniform measured result."""

import copy
from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
import re
import time

import numpy as np
import psutil

from core import Evaluation
from core.evaluate import EVALUATION_SEED, InvalidEvaluationResult, seeded_random_state

from . import probe


FRAME = re.compile(r'^\s*File "([^"]*)", line (\d+)')
HARNESS = tuple(
    str(Path(__file__).resolve().parents[1] / name) + os.sep for name in ("core", "traceaad")
)


def clean_traceback(text, code=None):
    """Drop the evaluation's own frames, with their source and caret lines.

    These are the core and TraceAAD layers and the call counter: its module
    and the wrapper appended after the program (``<string>`` lines past the
    program's last line). Task solver frames stay: they show how the function
    was called.
    """
    limit = len(code.splitlines()) if code is not None else None
    lines, kept, skip = (text or "").splitlines(), [], False
    for index, line in enumerate(lines):
        frame = FRAME.match(line)
        if frame:
            path, number = frame.group(1), int(frame.group(2))
            following = lines[index + 1] if index + 1 < len(lines) else ""
            skip = (
                path.startswith(HARNESS)
                or "probe.py" in path
                or "_traceaad_probe_" in following
                or path == "<string>"
                and limit is not None
                and number > limit
            )
            if not skip:
                kept.append(line)
            continue
        if skip and line.startswith("    "):
            continue
        skip = False
        kept.append(line)
    return "\n".join(kept)


def failing_line(error, code):
    """The program line where a runtime error was raised: the innermost frame in the program.

    The evaluated text is the program followed by the call counter, so frames
    past the program's last line belong to the counter.
    """
    lines = (code or "").splitlines()
    numbers = [
        int(n)
        for n in re.findall(r'File "<string>", line (\d+)', error or "")
        if 0 < int(n) <= len(lines)
    ]
    return lines[numbers[-1] - 1].strip()[:160] if numbers else None


def process_cpu_seconds(pid):
    """Cumulative process CPU time, including threads that have already exited."""
    try:
        usage = psutil.Process(pid).cpu_times()
        return usage.user + usage.system
    except psutil.NoSuchProcess, psutil.AccessDenied:
        return None


class SeededEvaluation(Evaluation):
    """Seeds every evaluation and measures the calls made to the candidate.

    The counters live in shared memory created here, in the search process;
    ``reset`` before an evaluation and ``measured`` after it (also after a
    timeout) read what the evaluation process recorded.
    """

    def __init__(self, inner):
        super().__init__(
            template_program=inner.template_program,
            task_description=inner.task_description,
            timeout_seconds=inner.timeout_seconds,
            safe_evaluate=inner.safe_evaluate,
            daemon_eval_process=inner.daemon_eval_process,
            fork_proc=inner.fork_proc,
        )
        self.inner = inner
        self.calls = multiprocessing.RawValue("q", 0)
        self.function_seconds = multiprocessing.RawValue("d", 0.0)
        self.call_started = multiprocessing.RawValue("d", 0.0)
        self.function_cpu_seconds = multiprocessing.RawValue("d", 0.0)
        self.call_cpu_started = multiprocessing.RawValue("d", 0.0)

    def reset(self):
        self.calls.value = 0
        self.function_seconds.value = 0.0
        self.call_started.value = 0.0
        self.function_cpu_seconds.value = 0.0
        self.call_cpu_started.value = 0.0

    def measured(self, until=None, pid=None):
        """Completed calls, wall and CPU time inside the function. A call still
        running (the evaluation was stopped inside it) counts its wall time up
        to ``until`` and, given the evaluation process ``pid``, its CPU time so far."""
        inside = float(self.function_seconds.value)
        cpu = float(self.function_cpu_seconds.value)
        started = float(self.call_started.value)
        if started:
            inside += max(0.0, (until if until is not None else time.monotonic()) - started)
            now = process_cpu_seconds(pid) if pid is not None else None
            if now is not None:
                cpu += max(0.0, now - float(self.call_cpu_started.value))
        return {
            "calls": int(self.calls.value),
            "function_seconds": inside,
            "function_cpu_seconds": cpu,
            "call_running": bool(started),
        }

    @contextmanager
    def program_context(self, source, function_name, *, seed=EVALUATION_SEED):
        # The task's own solver randomness (the ACO seed) is part of the task and
        # identical for every method; the seed sets only the candidate's random state.
        try:
            source = probe.instrument(source, function_name)
            probe.arm(
                self.calls,
                self.function_seconds,
                self.call_started,
                self.function_cpu_seconds,
                self.call_cpu_started,
            )
            with seeded_random_state(EVALUATION_SEED if seed is None else seed):
                yield source, _ScoredTask(copy.copy(self.inner)), {}
        finally:
            probe.arm(None, None, None)

    def evaluate_program(self, source, function):
        return _ScoredTask(self.inner).evaluate_program(source, function)


class _ScoredTask:
    def __init__(self, inner):
        self.inner = inner
        self.executes_source = getattr(inner, "executes_source", False)

    def evaluate_program(self, source, function):
        try:
            score = self.inner.evaluate_program(source, function)
        except ValueError as exc:
            if "heuristics must return a finite" in str(exc):
                raise InvalidEvaluationResult(str(exc)) from exc
            raise
        if score is None or not np.isfinite(float(score)):
            raise InvalidEvaluationResult("task returned no finite score")
        return {"score": float(score)}
