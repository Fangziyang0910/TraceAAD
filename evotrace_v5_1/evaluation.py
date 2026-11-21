"""整任务黑盒评价：隔离执行、目标值校验、超时处理与保留原文行号的报错摘要。"""

import importlib.util
import linecache
import math
import numbers
import random
import re
import sys
from pathlib import Path

from .models import Evaluation
from .runtime import ProcessFailure, ProcessTimeout, isolated_call
from .task import Task

TIMEOUT_PREFIX = "Evaluation exceeded"


def _parse_failure(failure: str, code: str | None = None) -> tuple[str, list[list[str]]]:
    """提取异常信息与候选源码中的报错帧；exec 仅给行号时从原文补出对应源码行。"""
    lines = failure.strip().splitlines()
    start = max(
        (
            i
            for i, line in enumerate(lines)
            if line and not line[0].isspace() and not line.startswith("Traceback")
        ),
        default=len(lines) - 1,
    )
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
    return re.sub(r"^(?:\w+\.)+(?=\w+(?::|$))", "", "\n".join(lines[start:])), blocks


def failure_feedback(failure: str | None, code: str | None = None) -> str:
    """修复提示使用的报错信息：候选源码报错帧 + 异常说明。"""
    if not failure or not failure.strip():
        return "Execution failed with an unknown error."
    message, blocks = _parse_failure(failure, code)
    return "\n".join([*(line for block in blocks for line in block), message])


def compact_failure(
    failure: str | None, code: str | None = None, *, status: str | None = None
) -> str:
    """一行失败原因，附出错源码行。"""
    text = (failure or "").strip()
    if status == "timeout" or text.startswith(TIMEOUT_PREFIX):
        return "timeout (exceeded evaluation time limit)"
    if not text:
        return "infeasible (unknown error)"
    message, blocks = _parse_failure(text, code)
    if blocks and len(blocks[-1]) > 1:
        return f"{message} at `{blocks[-1][1].strip()[:80]}`"
    return message[:120]


def _execute(task: Task, code: str) -> float:
    """在子进程中以黑盒方式执行候选程序，返回有限实数目标值。"""
    if task.memory_limit_bytes is not None and sys.platform.startswith("linux"):
        import resource

        data = next(
            line
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmData:")
        )
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
    linecache.cache["<candidate>"] = (
        len(code),
        None,
        code.splitlines(keepends=True),
        "<candidate>",
    )
    task.validate_source(code)
    function = None
    if not task.executes_source:
        scope = {"__name__": "__candidate__"}
        exec(compile(code, "<candidate>", "exec"), scope)  # noqa: S102 - runs in an isolated worker
        function = scope.get(task.entrypoint)
        if not callable(function):
            raise TypeError(f"The entry point {task.entrypoint!r} is not callable")
    result = task.evaluate_program(code, function)
    if isinstance(result, bool) or not isinstance(result, numbers.Real):
        raise TypeError(
            f"Task returned {result!r}; expected a finite real score after all solution checks"
        )
    fitness = float(result)
    if not math.isfinite(fitness):
        raise ValueError(f"Task returned non-finite fitness: {fitness!r}")
    return fitness


class Evaluator:
    """标准整任务黑盒评价器：单次隔离执行，只返回状态、目标值或报错信息。"""

    def __init__(self, task: Task):
        self.task = task

    def evaluate(self, code: str) -> Evaluation:
        try:
            fitness = isolated_call(_execute, (self.task, code), self.task.time_limit)
            return Evaluation("success", fitness=fitness)
        except ProcessTimeout:
            return Evaluation(
                "timeout",
                failure=f"{TIMEOUT_PREFIX} the total time limit of {self.task.time_limit:g} seconds.",
            )
        except (ProcessFailure, OSError) as exc:
            return Evaluation("error", failure=str(exc))
