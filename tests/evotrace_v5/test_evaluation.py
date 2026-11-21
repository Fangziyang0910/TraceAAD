"""纯黑盒评价器测试：标量任务、基准任务、并行实例池对齐、报错行号与超时处理。"""

import math
import time

import pytest

from benchmarks.tasks import training_task
from evotrace_v5.evaluation import Evaluator, compact_failure, failure_feedback
from evotrace_v5.task import Task
from experiments.traceaad.parallel_evaluation import PooledEvaluator


def _scalar_task(fn=None, *, time_limit=2.0):
    return Task(
        "Return a small score.",
        "def solve(x):\n    return x\n",
        fn or (lambda _source, solve: float(solve(3))),
        time_limit=time_limit,
    )


def test_scalar_task_returns_clean_fitness():
    ev = Evaluator(_scalar_task()).evaluate("def solve(x):\n    return x * 2.5\n")
    assert ev.status == "success"
    assert ev.fitness == pytest.approx(7.5)
    assert ev.failure is None


def test_invalid_result_and_runtime_error_include_candidate_line():
    bad_code = "def solve(x):\n    y = x + 1\n    return y / 0\n"
    ev = Evaluator(_scalar_task()).evaluate(bad_code)
    assert ev.status == "error"
    assert ev.fitness is None
    fb = failure_feedback(ev.failure, bad_code)
    assert "ZeroDivisionError" in fb
    assert "return y / 0" in fb
    assert "ZeroDivisionError" in compact_failure(ev.failure, bad_code, status=ev.status)

    nan_ev = Evaluator(_scalar_task()).evaluate("def solve(x):\n    return float('nan')\n")
    assert nan_ev.status == "error" and "non-finite" in (nan_ev.failure or "")


def test_timeout_is_reported_cleanly():
    def slow(_source, solve):
        time.sleep(0.5)
        return float(solve(1))

    ev = Evaluator(_scalar_task(slow, time_limit=0.1)).evaluate("def solve(x):\n    return x\n")
    assert ev.status == "timeout"
    assert ev.fitness is None
    assert "timeout" in compact_failure(ev.failure, status=ev.status)


def test_tsp_sequential_and_pooled_evaluators_match():
    benchmark, settings = training_task("tsp_construct", condition="traceaad")
    task = Task.from_evaluator(benchmark, metadata={"name": "tsp_construct", "settings": settings})
    code = task.template_program
    seq = Evaluator(task).evaluate(code)
    pooled = PooledEvaluator(Evaluator(task)).evaluate(code, [0, 1])
    assert seq.status == "success" and pooled.status == "success"
    assert math.isclose(seq.fitness, pooled.fitness, rel_tol=1e-12, abs_tol=1e-12)
