import multiprocessing
import random

import pytest

from benchmarks.tasks import CLASSES, CO_TASKS, FIXED_TASKS
from experiments.infra.whole_task_evaluation import WholeTaskEvaluator
from tests.support import TinyEvaluation, small_task
from traceaad.instance_evaluation import InstanceProgramEvaluator


class IndexedEvaluation(TinyEvaluation):
    def __init__(self, count=4):
        super().__init__()
        self.n_instance = count
        self.problem_size = 1
        self._datasets = list(range(count))

    def evaluate_program(self, source, function):
        return sum((function(x) for x in self._datasets)) / len(self._datasets)


def test_per_call_workers_and_deadlines(tmp_path):
    code = f"import time\ndef score(x):\n    start = time.monotonic()\n    time.sleep(0.3)\n    end = time.monotonic()\n    with open({str(tmp_path)!r} + '/' + str(x), 'w') as f:\n        f.write(str(start) + ',' + str(end))\n    return x\n"
    evaluator = InstanceProgramEvaluator(IndexedEvaluation(), timeout_seconds=0.1)
    assert evaluator.evaluate(code, "short")["failure"]["kind"] == "timeout"
    serial = evaluator.evaluate(code, "serial", timeout_seconds=1, n_workers=1)
    assert serial["fitness"] == 1.5 and serial["seconds"] > 1
    parallel = evaluator.evaluate(code, "parallel", timeout_seconds=1, n_workers=2)
    assert parallel["fitness"] == serial["fitness"]
    assert parallel["calls"] == 4
    assert parallel["instance_seconds"] > parallel["seconds"]
    intervals = [tuple(map(float, (tmp_path / str(i)).read_text().split(","))) for i in range(4)]
    concurrency = [sum((start <= t < end for start, end in intervals)) for t, _ in intervals]
    assert max(concurrency) == 2
    assert evaluator.timeout_seconds == 0.1 and evaluator.n_workers == 1
    assert evaluator.evaluate(code, "default")["failure"]["kind"] == "timeout"


def test_seed_and_globals_do_not_depend_on_workers():
    code = "import random\ncount = 0\ndef score(x):\n    global count\n    count += 1\n    assert count == 1\n    return random.random()\n"
    evaluator = InstanceProgramEvaluator(IndexedEvaluation(), seeds=(30, 40))
    serial = evaluator.evaluate(code, "x", n_workers=1)
    parallel = evaluator.evaluate(code, "x", n_workers=4)
    assert serial["fitness"] == parallel["fitness"]
    for record in parallel["evaluations"]:
        assert [r["score"] for r in record["instances"]] == [
            random.Random(record["seed"] + i).random() for i in range(4)
        ]


@pytest.mark.parametrize(
    "body,kind",
    [
        ("while True: pass", "timeout"),
        ('raise ValueError("bad")', "runtime_error"),
        ('return float("nan")', "invalid_output"),
        ("os._exit(7)", "runtime_error"),
    ],
)
def test_failure_cancels_other_instances_and_releases_workers(body, kind):
    before = {p.pid for p in multiprocessing.active_children()}
    code = f"import os, time\ndef score(x):\n    if x == 0:\n        {body}\n    time.sleep(10)\n    return 1"
    result = InstanceProgramEvaluator(IndexedEvaluation()).evaluate(
        code, "bad", timeout_seconds=0.5, n_workers=2
    )
    assert result["fitness"] is None and result["failure"]["kind"] == kind
    assert result["seconds"] < 3
    assert {p.pid for p in multiprocessing.active_children()} == before


class SlowSolver(IndexedEvaluation):
    def evaluate_program(self, source, function):
        import time

        time.sleep(0.4)
        return super().evaluate_program(source, function)


BUSY = "import time\ndef score(x):\n    end = time.process_time() + {}\n    while time.process_time() < end:\n        pass\n    return x"

SLEEP = "import time\ndef score(x):\n    time.sleep({})\n    return x"


def test_the_budget_counts_cpu_inside_the_function_only():
    evaluator = InstanceProgramEvaluator(SlowSolver(2), timeout_seconds=5, function_seconds=0.2)
    fast = evaluator.evaluate("def score(x): return x", "fast", n_workers=2)
    assert (
        fast["failure"] is None
        and min((r["seconds"] for r in fast["evaluations"][0]["instances"])) > 0.4
    )
    waiting = evaluator.evaluate(SLEEP.format(0.5), "waiting", n_workers=2)
    assert waiting["failure"] is None
    assert waiting["function_seconds"] > 0.9 and waiting["function_cpu_seconds"] < 0.2
    within = evaluator.evaluate(BUSY.format(0.1), "within", n_workers=2)
    assert within["failure"] is None and 0.15 < within["function_cpu_seconds"] < 0.4
    over = evaluator.evaluate(BUSY.format(0.6), "over", n_workers=2)
    assert (
        over["failure"]["kind"] == "timeout"
        and over["failure"]["error_type"] == "FunctionBudgetExceeded"
    )
    assert len(over["evaluations"][0]["instances"]) == 2
    assert all((r["valid"] for r in over["evaluations"][0]["instances"]))
    assert over["failure"]["instance_index"] is None
    assert over["mean_function_cpu_seconds"] > 0.2
    hang = InstanceProgramEvaluator(
        SlowSolver(2), timeout_seconds=0.3, function_seconds=1
    ).evaluate("def score(x): return x", "hang", n_workers=2)
    assert hang["failure"]["kind"] == "timeout" and hang["failure"]["error_type"] == "TimeoutError"


@pytest.mark.parametrize("workers", [1, 2])
def test_mean_budget_allows_a_slow_instance(workers):
    code = BUSY.format("0.3 if x == 0 else 0.02")
    task = IndexedEvaluation(2)
    evaluator = InstanceProgramEvaluator(task, timeout_seconds=5, function_seconds=0.2)
    result = evaluator.evaluate(code, "heterogeneous", n_workers=workers)
    assert result["fitness"] == 0.5
    record = result["evaluations"][0]
    assert len(record["instances"]) == 2 and all((r["valid"] for r in record["instances"]))
    assert record["instances"][0]["function_cpu_seconds"] > 0.2
    assert result["mean_function_cpu_seconds"] < 0.2


@pytest.mark.parametrize("task", CO_TASKS)
def test_six_task_scores_match_serial_fixed_solvers(task):
    if task in FIXED_TASKS:
        evaluation = CLASSES[task](limit=2)
        evaluation.outer_settings = {name: 1 for name in evaluation.outer_settings}
    else:
        evaluation = small_task(task)
    code = str(evaluation.template_program)
    old = WholeTaskEvaluator(evaluation).evaluate(code, "template")
    evaluator = InstanceProgramEvaluator(evaluation)
    serial = evaluator.evaluate(code, "template", n_workers=1)
    parallel = evaluator.evaluate(code, "template", n_workers=2)
    assert old["failure"] is None and serial["failure"] is None and (parallel["failure"] is None)
    assert serial["fitness"] == pytest.approx(old["fitness"])
    assert parallel["fitness"] == serial["fitness"]
    assert parallel["calls"] == serial["calls"] == old["calls"]


def test_process_cpu_reader_keeps_finished_thread_time():
    import os
    import threading
    import time

    from traceaad.evaluation import process_cpu_seconds

    before = process_cpu_seconds(os.getpid())
    clock_before = time.process_time()

    def work():
        end = time.thread_time() + 0.12
        while time.thread_time() < end:
            pass

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    expected = time.process_time() - clock_before
    measured = process_cpu_seconds(os.getpid()) - before
    assert measured == pytest.approx(expected, abs=0.04)
    assert measured >= 0.09
    assert process_cpu_seconds(999999999) is None
