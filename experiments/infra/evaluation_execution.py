"""Evaluate frozen programs with the current task execution defaults."""

from benchmarks.tasks import CO_TASKS, FUNCTION_LIMITS, FULL_SOLVER_TASKS, INSTANCE_LIMITS, PRIMARY_SPLITS, TEST_INSTANCES, heldout_task, validate_task_interface
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from traceaad.common.config import REVISION


def heldout_evaluator(config, split, seeds, *, workers, timeout_seconds, measure_calls,
                      task_factory=heldout_task, scheduler_socket=None):
    validate_task_interface(config['task'], config)
    params = config.get("method_params", {})
    count = params.get("eval_workers", 1) if workers is None else workers
    if config["task"] in CO_TASKS:
        scope = params.get('function_budget_scope', config.get('evaluation_execution', {}).get('function_budget_scope',
            'mean_instance' if config.get('revision') == REVISION else 'instance'))
        default_timeout = INSTANCE_LIMITS[config['task']] if scope == 'mean_instance' else (30 if config['task'] in FULL_SOLVER_TASKS else 20)
        timeout = params.get("eval_timeout_seconds", default_timeout) if timeout_seconds is None else timeout_seconds
        primary = split == 'eval' or split in PRIMARY_SPLITS[config['task']]
        task = task_factory(config["task"], split, 1,
                            TEST_INSTANCES * default_timeout if scope == 'instance' and primary else None)
        evaluator = InstanceProgramEvaluator(task, seeds, "heldout", measure_calls=measure_calls,
                                             function_seconds=params.get('function_seconds', FUNCTION_LIMITS[config['task']]),
                                             function_budget_scope=scope,
                                             timeout_seconds=timeout, n_workers=count,
                                             scheduler_socket=(params.get('scheduler_socket') if scheduler_socket is None
                                                               else scheduler_socket))
        metadata = {"timeout_seconds": timeout, "function_seconds": evaluator.function_seconds,
                    "function_budget_scope": evaluator.function_budget_scope,
                    "timeout_scope": "instance",
                    "workers": min(count, getattr(task, "n_instance", 1))}
        if evaluator.scheduler_socket:
            metadata['scheduler_socket'] = evaluator.scheduler_socket
    else:
        task = task_factory(config["task"], split, count, timeout_seconds)
        evaluator = ProgramEvaluator(task, seeds, "heldout", measure_calls=measure_calls)
        metadata = {"timeout_seconds": task.timeout_seconds, "workers": getattr(task, "n_workers", 1)}
    return evaluator, metadata
