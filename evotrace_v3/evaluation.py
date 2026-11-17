"""Whole-task evaluation: assembly, task execution, score validation, timeout, and failure formatting."""

import importlib.util
import linecache
import math
import numbers
from pathlib import Path
import random
import re
import sys

from .models import Evaluation
from .runtime import ProcessFailure, ProcessTimeout, isolated_call
from .task import Task

TIMEOUT_PREFIX = "Evaluation exceeded"


def _parse_failure(failure: str):
    """The exception message and the traceback blocks that lie in the candidate's own code."""
    lines = failure.strip().splitlines()
    start = len(lines) - 1
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and not line.startswith("Traceback"):
            start = i
    message = re.sub(r"^(?:\w+\.)+(?=\w+(?::|$))", "", "\n".join(lines[start:]))
    blocks, keep = [], False
    for line in lines[:start]:
        if line.lstrip().startswith('File "'):
            keep = "<candidate>" in line or "<string>" in line
            if keep:
                blocks.append([line.strip()])
        elif keep and line.startswith("  "):
            blocks[-1].append(line)
        else:
            keep = False
    return re.sub(r"\d+(?:\.\d+)?", "N", message), blocks


def failure_feedback(failure: str) -> str:
    """Candidate traceback lines plus normalized exception message for repair."""
    message, blocks = _parse_failure(failure)
    return "\n".join([*(line for block in blocks for line in block), message])


def compact_failure(failure_text: str | None) -> str:
    """Short one-line failure reason with offending candidate source line."""
    text = (failure_text or "").strip()
    if not text:
        return "infeasible (unknown error)"
    if text.startswith(TIMEOUT_PREFIX):
        return "timeout (exceeded evaluation time limit)"
    message, blocks = _parse_failure(text)
    if blocks:
        source = next(
            (line.strip() for line in blocks[-1][1:] if line.strip() and set(line.strip()) - set("~^ ")),
            None,
        )
        if source:
            return f"{message} at `{source[:80]}`"
    return message[:120]


def _execute(task: Task, code: str):
    if task.memory_limit_bytes is not None and sys.platform.startswith("linux"):
        import resource

        data = next(line for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmData:"))
        limit = int(data.split()[1]) * 1024 + task.memory_limit_bytes
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        limits = [limit, *[n for n in (soft, hard) if n != resource.RLIM_INFINITY]]
        resource.setrlimit(resource.RLIMIT_DATA, (min(limits), hard))
        try:
            Path("/proc/self/oom_score_adj").write_text("1000")
        except OSError:
            pass
    random.seed(task.evaluation_seed)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(task.evaluation_seed % (2**32))
    linecache.cache["<candidate>"] = (len(code), None, code.splitlines(keepends=True), "<candidate>")
    task.validate_source(code)
    function = None
    if not task.executes_source:
        scope = {"__name__": "__candidate__"}
        exec(compile(code, "<candidate>", "exec"), scope)
        function = scope.get(task.entrypoint)
        if not callable(function):
            raise ValueError(f"The entry point {task.entrypoint!r} is not callable")
    result = task.evaluate_program(code, function)
    if isinstance(result, bool) or not isinstance(result, numbers.Real):
        raise ValueError(
            f"Task returned {result!r}; expected a finite real score after all solution checks"
        )
    fitness = float(result)
    if not math.isfinite(fitness):
        raise ValueError(f"Task returned non-finite fitness: {fitness!r}")
    return fitness


class Evaluator:
    def __init__(self, task: Task):
        self.task = task

    def evaluate(self, code: str) -> Evaluation:
        try:
            fitness, _ = isolated_call(_execute, (self.task, code), self.task.time_limit)
            return Evaluation("success", fitness=fitness)
        except ProcessTimeout:
            return Evaluation(
                "timeout",
                failure=f"{TIMEOUT_PREFIX} the total time limit of {self.task.time_limit:g} seconds.",
            )
        except (ProcessFailure, OSError) as exc:
            return Evaluation("error", failure=str(exc))
