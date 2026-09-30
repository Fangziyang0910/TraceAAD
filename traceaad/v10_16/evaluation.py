"""Frozen-data protocol identity and seeded, isolated score evaluation."""

import copy
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np

from core import Evaluation
from core.evaluate import InvalidEvaluationResult


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
    sources = [Path(__file__), repo / "core" / "evaluate.py"]
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
    def __init__(self, inner):
        super().__init__(template_program=inner.template_program,
                         task_description=inner.task_description,
                         timeout_seconds=inner.timeout_seconds,
                         safe_evaluate=inner.safe_evaluate,
                         daemon_eval_process=inner.daemon_eval_process, fork_proc=inner.fork_proc)
        self.inner = inner
        # SecureEvaluator reads the timing policy from this wrapper.
        self.timeout_mode = getattr(inner, "timeout_mode", "wall")
        self.cpu_parallelism = getattr(inner, "cpu_parallelism", None)

    def evaluate_program(self, program_str, callable_func, *, seed=730241, source=None):
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            program_str = source if source is not None else program_str
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
            random.setstate(py_state)
            np.random.set_state(np_state)
