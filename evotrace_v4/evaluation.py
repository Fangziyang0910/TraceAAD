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


def _parse_failure(failure: str, code: str | None = None) -> tuple[str, list[list[str]]]:
    """异常信息，以及候选自身代码中的报错帧（帧行、源码行与标记行）。

    以 exec 运行的源码只给出 "<string>" 帧的行号，这一行从执行的原文中补出。
    """
    lines = failure.strip().splitlines()
    start = max((i for i, line in enumerate(lines)
                 if line and not line[0].isspace() and not line.startswith("Traceback")), default=len(lines) - 1)
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
    source = (code or "").splitlines()
    for block in blocks:
        number = re.search(r", line (\d+)", block[0])
        if len(block) == 1 and number and 0 < int(number.group(1)) <= len(source):
            block.append("    " + source[int(number.group(1)) - 1].strip())
    # 去掉异常类的模块前缀（如 core.evaluate.），只留类名与信息。
    return re.sub(r"^(?:\w+\.)+(?=\w+(?::|$))", "", "\n".join(lines[start:])), blocks


def failure_feedback(failure: str, code: str | None = None) -> str:
    """修复用的报错：候选代码中的报错帧与异常信息。"""
    message, blocks = _parse_failure(failure, code)
    return "\n".join([*(line for block in blocks for line in block), message])


def compact_failure(failure: str | None, code: str | None = None) -> str:
    """一行失败原因，附出错的源码行。"""
    text = (failure or "").strip()
    if not text:
        return "infeasible (unknown error)"
    if text.startswith(TIMEOUT_PREFIX):
        return "timeout (exceeded evaluation time limit)"
    message, blocks = _parse_failure(text, code)
    if blocks and len(blocks[-1]) > 1:
        return f"{message} at `{blocks[-1][1].strip()[:80]}`"
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
            return Evaluation("success", fitness=isolated_call(_execute, (self.task, code), self.task.time_limit))
        except ProcessTimeout:
            return Evaluation(
                "timeout",
                failure=f"{TIMEOUT_PREFIX} the total time limit of {self.task.time_limit:g} seconds.",
            )
        except (ProcessFailure, OSError) as exc:
            return Evaluation("error", failure=str(exc))
