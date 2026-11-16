"""Whole-task evaluation with the instances spread over several cores.

The total time limit is a pool of instance-seconds, as when the instances ran one
after another: the time every instance has used, finished or still running, must stay
within the limit. Eight cores finish a program that needs 160 instance-seconds in
about 20 seconds, and a program that hangs is stopped after 20 seconds, not 160.
"""

import copy
from multiprocessing.connection import wait
import multiprocessing
import os
import statistics
import sys
import time

from experiments.infra.instance_evaluation import _instance_view


def independent_instances(task):
    """The benchmark evaluator whose instances can run apart, or None."""
    evaluation = getattr(task.evaluate_program, "__self__", None)
    if evaluation is None or not hasattr(evaluation, "_datasets"):
        return None
    return evaluation if int(getattr(evaluation, "n_instance", 1)) > 1 else None


def _pinned(cpu, execute, task, code):
    os.sched_setaffinity(0, {cpu})
    return execute(task, code)


class PooledEvaluator:
    """Run one program on every instance in its own process, with one shared time pool."""

    def __init__(self, evaluator):
        self.task = evaluator.task
        evaluation = independent_instances(self.task)
        if evaluation is None:
            raise ValueError("The task has no independent instances")
        self.evaluation = evaluation
        self.count = int(self.evaluation.n_instance)
        module = sys.modules[type(evaluator).__module__]
        self.runtime = sys.modules[module.isolated_call.__module__]
        self.execute, self.result_type = module._execute, module.Evaluation

    def _instance_task(self, index):
        view = _instance_view(self.evaluation, index, self.task.time_limit)
        task = copy.copy(self.task)
        task.evaluate_program = view.evaluate_program
        task.evaluation_seed = self.task.evaluation_seed + index
        return task

    def evaluate(self, code, cpus):
        context = multiprocessing.get_context("fork")
        limit = self.task.time_limit
        pending, free = list(range(self.count)), list(cpus)
        running, scores, spent = {}, {}, 0.0
        try:
            while pending or running:
                while pending and free:
                    index, cpu = pending.pop(0), free.pop()
                    reader, writer = context.Pipe(duplex=False)
                    process = context.Process(
                        target=self.runtime._worker,
                        args=(writer, _pinned, (cpu, self.execute, self._instance_task(index), code)))
                    started = time.monotonic()
                    process.start()
                    writer.close()
                    running[index] = (process, reader, cpu, started)
                now = time.monotonic()
                remaining = limit - spent - sum(now - r[3] for r in running.values())
                if remaining <= 0:
                    return self.result_type(
                        "timeout", failure=f"Evaluation exceeded the total time limit of {limit:g} seconds.")
                ready = wait([r[1] for r in running.values()],
                             timeout=min(1.0, remaining / len(running)))
                for index, (process, reader, cpu, started) in list(running.items()):
                    if reader not in ready:
                        continue
                    try:
                        message = reader.recv()
                    except EOFError:
                        return self.result_type(
                            "error", failure="Execution process exited without returning a result")
                    if message[0] == "started":
                        continue
                    spent += time.monotonic() - started
                    del running[index]
                    free.append(cpu)
                    reader.close()
                    self.runtime._stop(process)
                    if message[0] == "error":
                        return self.result_type("error", failure=message[1])
                    scores[index] = message[1]
            return self.result_type("success", fitness=statistics.fmean(scores[i] for i in range(self.count)))
        finally:
            for process, reader, _, _ in running.values():
                reader.close()
                self.runtime._stop(process)
