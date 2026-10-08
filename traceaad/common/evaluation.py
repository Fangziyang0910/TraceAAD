"""Seeded, isolated evaluation and one uniform measured result."""

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import statistics
import time

import numpy as np

from core import Evaluation, SecureEvaluator
from core.evaluate import EVALUATION_SEED, InvalidEvaluationResult, seeded_random_state

from . import probe
from .config import REVISION

def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"sequence:{len(item)}".encode())
            for entry in item:
                visit(entry)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps(item, allow_nan=False).encode())
            digest.update(b"\x00")
        else:
            raise TypeError(f"unfingerprintable protocol value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def protocol_identity(evaluation, seeds, role, measure_calls):
    """Identify the fixed task data/settings, without freezing Python source files."""
    settings = {key: value for key, value in vars(evaluation).items()
                if key == "_datasets" or not key.startswith("_")}
    environment = fingerprint({"task": type(evaluation).__name__, "settings": settings})
    return fingerprint({"environment": environment, "seeds": seeds, "role": role, "execution": REVISION, "measure_calls": measure_calls}), environment


FRAME = re.compile(r'^\s*File "([^"]*)", line (\d+)')
HARNESS = tuple(str(Path(__file__).resolve().parents[2] / name) + os.sep for name in ("core", "traceaad"))


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
            skip = (path.startswith(HARNESS) or "probe.py" in path or "_traceaad_probe_" in following
                    or path == "<string>" and limit is not None and number > limit)
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
    numbers = [int(n) for n in re.findall(r'File "<string>", line (\d+)', error or "") if 0 < int(n) <= len(lines)]
    return lines[numbers[-1] - 1].strip()[:160] if numbers else None


def process_cpu_seconds(pid):
    """CPU time of a running process over all its threads (time.process_time()
    inside it), or None once it has exited."""
    try:
        return sum(int(Path(path).read_text().split()[0])
                   for path in Path(f"/proc/{pid}/task").glob("*/schedstat")) / 1e9
    except (OSError, ValueError, IndexError):
        return None


class SeededEvaluation(Evaluation):
    """Seeds every evaluation and measures the calls made to the candidate.

    The counters live in shared memory created here, in the search process;
    ``reset`` before an evaluation and ``measured`` after it (also after a
    timeout) read what the evaluation process recorded.
    """

    def __init__(self, inner, measure_calls=True):
        super().__init__(template_program=inner.template_program,
                         task_description=inner.task_description,
                         timeout_seconds=inner.timeout_seconds,
                         safe_evaluate=inner.safe_evaluate,
                         daemon_eval_process=inner.daemon_eval_process, fork_proc=inner.fork_proc)
        self.inner, self.measure_calls = inner, measure_calls
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
        if not self.measure_calls:
            return {"calls": None, "function_seconds": None, "function_cpu_seconds": None, "call_running": False}
        inside = float(self.function_seconds.value)
        cpu = float(self.function_cpu_seconds.value)
        started = float(self.call_started.value)
        if started:
            inside += max(0.0, (until if until is not None else time.monotonic()) - started)
            now = process_cpu_seconds(pid) if pid is not None else None
            if now is not None:
                cpu += max(0.0, now - float(self.call_cpu_started.value))
        return {"calls": int(self.calls.value), "function_seconds": inside,
                "function_cpu_seconds": cpu, "call_running": bool(started)}

    @contextmanager
    def program_context(self, source, function_name, *, seed=EVALUATION_SEED):
        # The task's own solver randomness (the ACO seed) is part of the task and
        # identical for every method; the seed sets only the candidate's random state.
        try:
            if self.measure_calls:
                source = probe.instrument(source, function_name)
                probe.arm(self.calls, self.function_seconds, self.call_started,
                          self.function_cpu_seconds, self.call_cpu_started)
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


class ProgramEvaluator:
    """Run all configured seeds and return fitness, failure and measurements.

    Search, selection and held-out use the same implementation. Per-seed
    records are returned to the caller and committed with the completed result.
    """

    def __init__(self, evaluation, seeds=(730241,), role="search", *, measure_calls=True):
        self.seeded = SeededEvaluation(evaluation, measure_calls)
        self.evaluator = SecureEvaluator(self.seeded)
        self.seeds, self.role = tuple(seeds), role
        self.protocol, self.environment = protocol_identity(evaluation, self.seeds, role, measure_calls)

    def evaluate(self, code, source_key):
        records = []
        for seed in self.seeds:
            started = time.monotonic()
            self.seeded.reset()
            outcome = self.evaluator.evaluate_program_with_details(
                code, seed=seed)
            elapsed = time.monotonic() - started
            measured = self.seeded.measured(until=started + elapsed)
            if outcome.failure_kind == "timeout" and self.seeded.timeout_seconds is not None and measured["function_seconds"] is not None:
                measured["function_seconds"] = min(measured["function_seconds"], self.seeded.timeout_seconds)
            value = outcome.result
            valid = (isinstance(value, dict) and type(value.get("score")) in (int, float)
                     and math.isfinite(value["score"]))
            records.append({"key": source_key, "role": self.role, "protocol": self.protocol,
                            "seed": seed, "valid": valid, "score": value["score"] if valid else None,
                            "failure_kind": outcome.failure_kind, "error_type": outcome.error_type,
                            "error": outcome.error, "traceback": outcome.traceback,
                            "seconds": elapsed, "cpu_seconds": outcome.cpu_seconds, **measured})
            if not valid:
                kind = ("timeout" if outcome.failure_kind == "timeout" else
                        "invalid_output" if outcome.failure_kind == "invalid_result" else "runtime_error")
                error = clean_traceback(outcome.traceback or outcome.error or kind, code)
                return {"fitness": None, "failure": {"kind": kind, "error": error,
                        "line": failing_line(error, code), "seconds": elapsed, **measured},
                        "seconds": elapsed, **measured, "evaluations": records}
        return {"fitness": statistics.fmean(r["score"] for r in records), "failure": None,
                "seconds": statistics.fmean(r["seconds"] for r in records),
                "calls": round(statistics.fmean(r["calls"] for r in records)) if self.seeded.measure_calls else None,
                "function_seconds": statistics.fmean(r["function_seconds"] for r in records) if self.seeded.measure_calls else None,
                "call_running": False, "evaluations": records}
