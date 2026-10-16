"""Offline model replies and small tasks used by the research checks."""

import json

from core import Evaluation


class TokenLLM:
    """Scripted replies with approximate counts; never connects to a model service."""

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 16

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {
            "content": next(self.responses),
            "finish_reason": "stop",
            "usage": {},
            "model": "test",
        }

    def close(self):
        pass


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program="def score(x):\n    pass",
            task_description="Return a numeric score.",
            safe_evaluate=False,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return -callable_func(1)


def response(value, *, analysis="", design=None):
    return json.dumps(
        {
            "analysis": analysis,
            "design": design or f"return {value}",
            "code": f"def score(x):\n    return {value}\n",
        }
    )


def small_task(task, seed=10):
    from benchmarks.cvrp_aco import CVRPACOEvaluation
    from benchmarks.online_bin_packing import OBPEvaluation
    from benchmarks.op_aco import OPACOEvaluation
    from benchmarks.tsp_construct import TSPEvaluation
    from benchmarks.vrptw_construct import VRPTWEvaluation

    if task == "tsp_construct":
        return TSPEvaluation(n_instance=2, problem_size=10, seed=seed)
    if task == "vrptw_construct":
        return VRPTWEvaluation(n_instance=2, problem_size=10, seed=seed)
    if task == "online_bin_packing":
        return OBPEvaluation(
            dataset_specs=[{"n_instances": 1, "n_items": 64, "capacities": [100, 500]}], seed=seed
        )
    cls = CVRPACOEvaluation if task == "cvrp_aco" else OPACOEvaluation
    evaluation = cls(
        split="train" if seed == 10 else "val_50", n_ants=3, n_iterations=2, n_workers=1
    )
    evaluation._datasets = evaluation._datasets[:2]
    evaluation.n_instance = 2
    return evaluation
