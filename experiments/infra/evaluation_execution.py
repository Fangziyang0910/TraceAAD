"""Evaluate frozen programs with the current task execution defaults."""

from benchmarks.tasks import CO_TASKS, INSTANCE_SECONDS, heldout_task
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.instance_evaluation import InstanceProgramEvaluator


def heldout_evaluator(config, split, seeds, *, workers, timeout_seconds, measure_calls,
                      task_factory=heldout_task, scheduler_socket=None):
    params = config.get("method_params", {})
    count = params.get("eval_workers", 1) if workers is None else workers
    if config["task"] in CO_TASKS:
        timeout = params.get("eval_timeout_seconds", INSTANCE_SECONDS) if timeout_seconds is None else timeout_seconds
        task = task_factory(config["task"], split, 1, None)
        evaluator = InstanceProgramEvaluator(task, seeds, "heldout", measure_calls=measure_calls,
                                             timeout_seconds=timeout, n_workers=count,
                                             scheduler_socket=(params.get('scheduler_socket') if scheduler_socket is None
                                                               else scheduler_socket))
        metadata = {"timeout_seconds": timeout, "function_seconds": evaluator.function_seconds,
                    "timeout_scope": "instance",
                    "workers": min(count, getattr(task, "n_instance", 1))}
        if evaluator.scheduler_socket:
            metadata['scheduler_socket'] = evaluator.scheduler_socket
    else:
        task = task_factory(config["task"], split, count, timeout_seconds)
        evaluator = ProgramEvaluator(task, seeds, "heldout", measure_calls=measure_calls)
        metadata = {"timeout_seconds": task.timeout_seconds, "workers": getattr(task, "n_workers", 1)}
    return evaluator, metadata
