"""Evaluate frozen programs under the current rules; old rules use saved snapshots."""

from benchmarks.tasks import (
    CO_TASKS,
    FUNCTION_LIMITS,
    INSTANCE_LIMITS,
    heldout_task,
    validate_task_interface,
)
from traceaad.instance_evaluation import InstanceProgramEvaluator
from .whole_task_evaluation import WholeTaskEvaluator


def heldout_evaluator(
    config,
    split,
    seeds,
    *,
    workers,
    timeout_seconds,
    task_factory=heldout_task,
    scheduler_socket=None,
):
    validate_task_interface(config["task"], config)
    params = config.get("method_params", {})
    count = params.get("eval_workers", 1) if workers is None else workers
    if config["task"] not in CO_TASKS:
        task = task_factory(config["task"], split, count, timeout_seconds)
        evaluator = WholeTaskEvaluator(task, seeds, "heldout")
        return evaluator, {
            "timeout_seconds": task.timeout_seconds,
            "workers": getattr(task, "n_workers", 1),
        }
    timeout = (
        params.get("eval_timeout_seconds", INSTANCE_LIMITS[config["task"]])
        if timeout_seconds is None
        else timeout_seconds
    )
    task = task_factory(config["task"], split, 1, None)
    evaluator = InstanceProgramEvaluator(
        task,
        seeds,
        "heldout",
        function_seconds=params.get("function_seconds", FUNCTION_LIMITS[config["task"]]),
        timeout_seconds=timeout,
        n_workers=count,
        scheduler_socket=params.get("scheduler_socket")
        if scheduler_socket is None
        else scheduler_socket,
    )
    metadata = {
        "score_metric": getattr(task, "score_metric", "raw_objective"),
        "timeout_seconds": timeout,
        "function_seconds": evaluator.function_seconds,
        "function_budget_scope": evaluator.function_budget_scope,
        "timeout_scope": "instance",
        "workers": min(count, evaluator.n_instance),
    }
    if evaluator.scheduler_socket:
        metadata["scheduler_socket"] = evaluator.scheduler_socket
    return evaluator, metadata
