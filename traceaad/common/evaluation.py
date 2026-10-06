"""Seeded, isolated evaluation and one uniform measured result."""

import copy
import hashlib
import json
import math
import multiprocessing
import random
import re
import statistics
import time

import numpy as np

from core import Evaluation, SecureEvaluator
from core.evaluate import InvalidEvaluationResult

from . import probe

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


def protocol_identity(evaluation, seeds, role):
    """Identify the fixed task data/settings, without freezing Python source files."""
    settings = {key: value for key, value in vars(evaluation).items()
                if key == "_datasets" or not key.startswith("_")}
    environment = fingerprint({"task": type(evaluation).__name__, "settings": settings})
    return fingerprint({"environment": environment, "seeds": seeds, "role": role}), environment


def clean_traceback(text):
    """Drop the frames of the call counter, which wraps the candidate during evaluation."""
    lines, kept, skip = (text or "").splitlines(), [], False
    for index, line in enumerate(lines):
        if skip:
            skip = False
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if line.lstrip().startswith("File ") and ("probe.py" in line or "_traceaad_probe_" in following):
            skip = True
            continue
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

    def reset(self):
        self.calls.value = 0
        self.function_seconds.value = 0.0
        self.call_started.value = 0.0

    def measured(self, until=None):
        """Completed calls and time inside the function; a call still running
        (the evaluation was stopped inside it) counts its time up to ``until``."""
        if not self.measure_calls:
            return {"calls": None, "function_seconds": None, "call_running": False}
        inside = float(self.function_seconds.value)
        started = float(self.call_started.value)
        if started:
            inside += max(0.0, (until if until is not None else time.monotonic()) - started)
        return {"calls": int(self.calls.value), "function_seconds": inside, "call_running": bool(started)}

    def evaluate_program(self, program_str, callable_func, *, seed=730241, source=None):
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            program_str = source if source is not None else program_str
            if self.measure_calls:
                program_str = probe.instrument(program_str, callable_func.__name__)
                probe.arm(self.calls, self.function_seconds, self.call_started)
            random.seed(seed)
            np.random.seed(seed)
            namespace = {}
            exec(program_str, namespace)
            function = namespace[callable_func.__name__]
            evaluator = copy.copy(self.inner)
            if hasattr(evaluator, "aco_seed"):
                evaluator.aco_seed += seed
            try:
                score = evaluator.evaluate_program(program_str, function)
            except ValueError as exc:
                if "heuristics must return a finite" in str(exc):
                    raise InvalidEvaluationResult(str(exc)) from exc
                raise
            if score is None or not np.isfinite(float(score)):
                raise InvalidEvaluationResult("task returned no finite score")
            return {"score": float(score)}
        finally:
            probe.arm(None, None, None)
            random.setstate(py_state)
            np.random.set_state(np_state)


class ProgramEvaluator:
    """Run all configured seeds and return fitness, failure and measurements.

    Search, selection and held-out use the same implementation. Per-seed
    records are returned to the caller and committed with the completed result.
    """

    def __init__(self, evaluation, seeds=(730241,), role="search", *, measure_calls=True):
        self.seeded = SeededEvaluation(evaluation, measure_calls)
        self.evaluator = SecureEvaluator(self.seeded)
        self.seeds, self.role = tuple(seeds), role
        self.protocol, self.environment = protocol_identity(evaluation, self.seeds, role)

    def evaluate(self, code, source_key):
        records = []
        for seed in self.seeds:
            started = time.monotonic()
            self.seeded.reset()
            outcome = self.evaluator.evaluate_program_with_details(
                self.seeded.template_program, source=code, seed=seed)
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
                error = clean_traceback(outcome.traceback or outcome.error or kind)
                return {"fitness": None, "failure": {"kind": kind, "error": error,
                        "line": failing_line(error, code), "seconds": elapsed, **measured},
                        "seconds": elapsed, **measured, "evaluations": records}
        return {"fitness": statistics.fmean(r["score"] for r in records), "failure": None,
                "seconds": statistics.fmean(r["seconds"] for r in records),
                "calls": round(statistics.fmean(r["calls"] for r in records)) if self.seeded.measure_calls else None,
                "function_seconds": statistics.fmean(r["function_seconds"] for r in records) if self.seeded.measure_calls else None,
                "call_running": False, "evaluations": records}
