import json
import tempfile
import unittest
from pathlib import Path

from llm4ad.base import Evaluation, LLM
from baselines.pathwise import PathWise
from baselines.pathwise.profiler import PathWiseProfiler


def make_init_code(label: int) -> str:
    return f"""Description: init-{label}
Derivation Rationale: initialize {label}
```python
def heuristic_v2(x):
    return {label}
```"""


def make_world_code(label: int) -> str:
    return f"""Description: rollout-{label}
```python
def heuristic_v2(x):
    return {label}
```"""


class ImmediateResult:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        return ImmediateResult(fn(*args, **kwargs))

    def shutdown(self, *args, **kwargs):
        pass


class ScriptedLLM(LLM):
    def __init__(
            self,
            *,
            init_responses=None,
            policy_responses=None,
            world_responses=None,
            policy_critic_responses=None,
            world_critic_responses=None,
            temperature=None,
    ):
        super().__init__()
        self.init_responses = list(init_responses or [])
        self.policy_responses = list(policy_responses or [])
        self.world_responses = list(world_responses or [])
        self.policy_critic_responses = list(policy_critic_responses or [])
        self.world_critic_responses = list(world_critic_responses or [])
        self.prompts = []
        self.closed = False
        if temperature is not None:
            self.temperature = temperature

    def draw_sample(self, prompt, *args, **kwargs):
        self.prompts.append((prompt, kwargs))
        prompt = str(prompt)
        if "PathWise initialization agent" in prompt:
            return self.init_responses.pop(0) if self.init_responses else "invalid"
        if "PathWise policy critic" in prompt:
            return self.policy_critic_responses.pop(0) if self.policy_critic_responses else "policy reflection"
        if "PathWise world model critic" in prompt:
            return self.world_critic_responses.pop(0) if self.world_critic_responses else "world reflection"
        if "PathWise policy agent" in prompt:
            return self.policy_responses.pop(0) if self.policy_responses else "PARENTS: [init_0]\nDIRECTIVE: refine"
        if "PathWise world model agent" in prompt:
            return self.world_responses.pop(0) if self.world_responses else "invalid"
        return "invalid"

    def close(self):
        self.closed = True


class FakeEvaluation(Evaluation):
    def __init__(self, scores=None):
        super().__init__(
            template_program=(
                "def heuristic(x):\n"
                "    \"\"\"Return a priority for x.\"\"\"\n"
                "    return x\n"
            ),
            task_description="Design a heuristic.",
            safe_evaluate=False,
        )
        self.scores = list(scores or [])
        self.programs = []

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.programs.append(program_str)
        if not self.scores:
            return None
        return self.scores.pop(0)


def make_method(
        *,
        llm=None,
        scores=None,
        max_sample_nums=20,
        pop_size=2,
        init_pop_size=2,
        num_actions=1,
        num_rollouts=1,
        max_inner_steps=1,
) -> tuple[PathWise, ScriptedLLM, FakeEvaluation]:
    llm = llm or ScriptedLLM()
    evaluation = FakeEvaluation(scores=scores)
    method = PathWise(
        llm=llm,
        evaluation=evaluation,
        profiler=None,
        max_sample_nums=max_sample_nums,
        pop_size=pop_size,
        init_pop_size=init_pop_size,
        num_actions=num_actions,
        num_rollouts=num_rollouts,
        max_inner_steps=max_inner_steps,
        num_evaluators=1,
        use_prompt_perturbation=False,
    )
    method._evaluation_executor.shutdown(cancel_futures=True)
    method._evaluation_executor = ImmediateExecutor()
    return method, llm, evaluation


class PathWiseMechanicsTest(unittest.TestCase):
    def test_initialization_uses_llm4ad_evaluator_and_keeps_top_distinct_scores(self):
        llm = ScriptedLLM(
            init_responses=[make_init_code(1), make_init_code(2), make_init_code(3)],
            temperature=1.0,
        )
        method, llm, evaluation = make_method(
            llm=llm,
            scores=[1.0, 3.0, 2.0],
            init_pop_size=3,
            max_sample_nums=3,
        )

        method._initialize_population()

        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual([node.score for node in method._population.nodes], [3.0, 2.0])
        self.assertEqual(method.best_node.score, 3.0)
        self.assertEqual(len(evaluation.programs), 3)
        self.assertTrue(all("import random" in program for program in evaluation.programs))
        self.assertTrue(all(kwargs == {"temperature": 1.3} for _, kwargs in llm.prompts))

    def test_initialization_duplicates_best_when_valid_population_is_short(self):
        llm = ScriptedLLM(init_responses=[make_init_code(7)])
        method, _, _ = make_method(
            llm=llm,
            scores=[7.0],
            pop_size=3,
            init_pop_size=1,
            max_sample_nums=1,
        )

        method._initialize_population()

        self.assertEqual(len(method._population.nodes), 3)
        self.assertEqual([node.score for node in method._population.nodes], [7.0, 7.0, 7.0])
        self.assertEqual(len({node.node_id for node in method._population.nodes}), 3)
        self.assertIn("duplicated", method._population.nodes[1].description)

    def test_original_pathwise_parameter_aliases_are_consumed(self):
        llm = ScriptedLLM()
        evaluation = FakeEvaluation(scores=[])
        method = PathWise(
            llm=llm,
            evaluation=evaluation,
            profiler=None,
            max_fe=9,
            N_p=3,
            N_a=4,
            N_w=5,
            use_prompt_perturbation=False,
        )
        method._evaluation_executor.shutdown(cancel_futures=True)

        self.assertEqual(method._max_sample_nums, 9)
        self.assertEqual(method._pop_size, 3)
        self.assertEqual(method._num_actions, 4)
        self.assertEqual(method._num_rollouts, 5)

    def test_run_builds_entailment_graph_and_updates_best_function(self):
        llm = ScriptedLLM(
            init_responses=[make_init_code(1), make_init_code(2)],
            policy_responses=["PARENTS: [init_0]\nDIRECTIVE: make the return larger"],
            world_responses=[make_world_code(3)],
        )
        method, llm, _ = make_method(
            llm=llm,
            scores=[1.0, 2.0, 3.0],
            max_sample_nums=3,
        )

        method.run()

        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual(method.best_node.score, 3.0)
        self.assertIn("return 3", str(method.best_function))
        self.assertEqual([node.score for node in method._population.nodes], [3.0, 2.0])
        self.assertTrue(llm.closed)

    def test_population_update_keeps_leaf_before_discarded_and_roots(self):
        llm = ScriptedLLM(
            init_responses=[make_init_code(1), make_init_code(2)],
            policy_responses=[
                "PARENTS: [init_0]\nDIRECTIVE: improve first",
                "PARENTS: [init_1]\nDIRECTIVE: improve second",
            ],
            world_responses=[make_world_code(3), make_world_code(4)],
            policy_critic_responses=["prefer second parent"],
            world_critic_responses=["simplify the low score variant"],
        )
        method, _, _ = make_method(
            llm=llm,
            scores=[1.0, 2.0, 3.0, 4.0],
            max_sample_nums=4,
            num_actions=2,
            num_rollouts=1,
        )

        method.run()

        self.assertEqual([node.score for node in method._population.nodes], [4.0, 3.0])
        self.assertEqual(method._policy_reflection_history, ["prefer second parent"])
        self.assertEqual(method._world_model_reflection_history, ["simplify the low score variant"])

    def test_invalid_world_model_output_retries_then_falls_back_to_parent_code(self):
        llm = ScriptedLLM(
            init_responses=[make_init_code(1), make_init_code(2)],
            policy_responses=["PARENTS: [init_0]\nDIRECTIVE: improve first"],
            world_responses=[
                "Description: invalid\nnot code",
                "Description: invalid\nnot code",
                "Description: invalid\nnot code",
                "Description: invalid\nnot code",
            ],
        )
        method, _, _ = make_method(
            llm=llm,
            scores=[1.0, 2.0, 1.0],
            max_sample_nums=3,
        )

        method.run()

        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual(method.best_node.score, 2.0)
        self.assertEqual([node.score for node in method._population.nodes], [1.0, 2.0])

    def test_invalid_policy_parent_retries_before_accepting_action(self):
        llm = ScriptedLLM(
            init_responses=[make_init_code(1), make_init_code(2)],
            policy_responses=[
                "PARENTS: [missing]\nDIRECTIVE: invalid parent",
                "PARENTS: [init_0]\nDIRECTIVE: retry succeeds",
            ],
            world_responses=[make_world_code(3)],
        )
        method, _, _ = make_method(
            llm=llm,
            scores=[1.0, 2.0, 3.0],
            max_sample_nums=3,
        )

        method.run()

        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual(method.best_node.score, 3.0)
        policy_prompts = [prompt for prompt, _ in llm.prompts if "PathWise policy agent" in prompt]
        self.assertEqual(len(policy_prompts), 2)

    def test_no_valid_action_falls_back_to_best_population_node(self):
        llm = ScriptedLLM(
            init_responses=[make_init_code(1), make_init_code(2)],
            policy_responses=[
                "PARENTS: [missing]\nDIRECTIVE: invalid parent",
                "PARENTS: [still_missing]\nDIRECTIVE: invalid parent",
            ],
        )
        method, _, _ = make_method(
            llm=llm,
            scores=[1.0, 2.0, 2.0],
            max_sample_nums=3,
        )

        method.run()

        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual(method._outer_iteration, 1)
        self.assertEqual(method.best_node.score, 2.0)
        self.assertEqual([node.score for node in method._population.nodes], [2.0, 1.0])

    def test_run_summary_marks_unhandled_exception_as_error(self):
        llm = ScriptedLLM(init_responses=[make_init_code(1), make_init_code(2)])
        method, _, _ = make_method(
            llm=llm,
            scores=[1.0, 2.0],
            max_sample_nums=5,
        )
        method._initialize_population()
        method._resume_mode = True

        def fail_construct():
            raise RuntimeError("pathwise failure")

        method._construct_entailment_graph = fail_construct

        with tempfile.TemporaryDirectory() as tmpdir:
            method._profiler = PathWiseProfiler(log_dir=tmpdir, create_random_path=False, log_style="simple")
            method.run()
            summary = json.loads((Path(tmpdir) / "run_summary.json").read_text())

        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["error_type"], "RuntimeError")
        self.assertIn("pathwise failure", summary["error"])


if __name__ == "__main__":
    unittest.main()
