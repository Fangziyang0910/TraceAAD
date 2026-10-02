"""Frozen-data protocol identity and seeded, isolated score evaluation with call measurement."""

import copy
import hashlib
import inspect
import json
import multiprocessing
import random
import sys
import time
from pathlib import Path

import numpy as np

from core import Evaluation
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
    module = inspect.getmodule(type(evaluation))
    source_file = getattr(module, "__file__", None)
    settings = {key: value for key, value in vars(evaluation).items()
                if key == "_datasets" or not key.startswith("_")}
    repo = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(probe.__file__), repo / "core" / "evaluate.py"]
    if source_file:
        sources.append(Path(source_file))
        if "/benchmarks/" in source_file:
            sources.extend(Path(source_file).parent.glob("*.py"))
    hashes = {str(p.resolve().relative_to(repo) if p.resolve().is_relative_to(repo) else p.resolve()):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))}
    environment = fingerprint({"python": sys.version, "numpy": np.__version__,
                               "sources": hashes, "settings": settings})
    return fingerprint({"environment": environment, "seeds": seeds, "role": role}), environment


class SeededEvaluation(Evaluation):
    """Seeds every evaluation and measures the calls made to the candidate.

    The counters live in shared memory created here, in the search process;
    ``reset`` before an evaluation and ``measured`` after it (also after a
    timeout) read what the evaluation process recorded.
    """

    def __init__(self, inner):
        super().__init__(template_program=inner.template_program,
                         task_description=inner.task_description,
                         timeout_seconds=inner.timeout_seconds,
                         safe_evaluate=inner.safe_evaluate,
                         daemon_eval_process=inner.daemon_eval_process, fork_proc=inner.fork_proc)
        self.inner = inner
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
        inside = float(self.function_seconds.value)
        started = float(self.call_started.value)
        if started:
            inside += max(0.0, (until if until is not None else time.monotonic()) - started)
        return {"calls": int(self.calls.value), "function_seconds": inside, "call_running": bool(started)}

    def evaluate_program(self, program_str, callable_func, *, seed=730241, source=None):
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            program_str = source if source is not None else program_str
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
