import time

from evotrace_v3 import Task
from evotrace_v3.evaluation import Evaluator
from experiments.traceaad.parallel_evaluation import PooledEvaluator

TEMPLATE = "def score(value):\n    return value\n"


class Instances:
    def __init__(self, values):
        self._datasets = list(values)
        self.n_instance = len(values)

    def evaluate_program(self, code, function):
        results = [function(value) for value in self._datasets[:self.n_instance]]
        return sum(results) / len(results)


def make(values, limit):
    evaluation = Instances(values)
    task = Task("Return the value.", TEMPLATE, evaluation.evaluate_program, limit)
    task.metadata["n_instance"] = len(values)
    return Evaluator(task)


def test_parallel_score_matches_sequential_mean():
    evaluator = make([1, 2, 3, 4, 5, 6], 30)
    code = "def score(value):\n    return value * 2\n"
    assert evaluator.evaluate(code).fitness == 7
    result = PooledEvaluator(evaluator).evaluate(code, [0, 1, 2])
    assert (result.status, result.fitness) == ("success", 7)


def test_instance_error_ends_the_evaluation_with_its_traceback():
    evaluator = make([1, 2, 3, 4], 30)
    code = "def score(value):\n    if value == 3:\n        raise KeyError('three')\n    return value\n"
    result = PooledEvaluator(evaluator).evaluate(code, [0, 1])
    assert result.status == "error" and "KeyError" in result.failure


def test_pool_counts_instance_seconds_across_workers():
    evaluator = make([0, 1, 2, 3], 4)
    code = "import time\n\ndef score(value):\n    time.sleep(10)\n    return value\n"
    started = time.monotonic()
    result = PooledEvaluator(evaluator).evaluate(code, [0, 1, 2, 3])
    assert result.status == "timeout"
    assert time.monotonic() - started < 3


def test_pool_allows_one_slow_instance_when_the_others_are_quick():
    evaluator = make([0, 1, 2, 3], 6)
    code = "import time\n\ndef score(value):\n    time.sleep(3 if value == 0 else 0)\n    return value\n"
    result = PooledEvaluator(evaluator).evaluate(code, [0, 1])
    assert (result.status, result.fitness) == ("success", 1.5)
