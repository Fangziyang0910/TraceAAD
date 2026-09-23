import math
import unittest
from unittest.mock import patch

import numpy as np

from llm4ad.base import Evaluation, Function, LLM, TextFunctionProgramConverter
from baselines.eoh.eoh import EoH
from baselines.eoh.population import Population


def make_function(label: int, score=None) -> Function:
    func = Function(name="heuristic", args="x", body=f"    return {label}")
    func.algorithm = f"algorithm-{label}"
    func.score = score
    return func


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


class FakeSampler:
    def __init__(self, funcs):
        self.funcs = list(funcs)
        self.prompts = []

    def get_thought_and_function(self, prompt):
        self.prompts.append(prompt)
        if not self.funcs:
            return None, None
        return "generated description", self.funcs.pop(0)


class FakeEvaluator:
    def __init__(self, scores):
        self.scores = list(scores)

    def evaluate_program_record_time(self, program):
        if not self.scores:
            return None, 0.01
        return self.scores.pop(0), 0.01


class FakeLLM(LLM):
    def __init__(self):
        super().__init__()
        self.closed = False

    def draw_sample(self, prompt, *args, **kwargs):
        return "{generated}\n    return 1"

    def close(self):
        self.closed = True


class FakeEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program="def heuristic(x):\n    return x\n",
            task_description="Design a heuristic.",
            safe_evaluate=False,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return 1.0


def make_method(pop_size=2, scores=None, funcs=None) -> EoH:
    method = object.__new__(EoH)
    method._task_description_str = "Design a heuristic."
    method._function_to_evolve = make_function(0)
    method._template_program = TextFunctionProgramConverter.text_to_program(str(method._function_to_evolve))
    method._pop_size = pop_size
    method._selection_num = 2
    method._operators = ["e1", "e2", "m1", "m2"]
    method._operator_weights = [1.0, 1.0, 1.0, 1.0]
    method._population = Population(pop_size=pop_size)
    method._tot_sample_nums = 0
    method._max_sample_nums = 100
    method._max_generations = None
    method._initial_sample_nums_max = 2 * pop_size
    method._profiler = None
    method._resume_mode = False
    method._debug_mode = False
    method._evaluation_executor = ImmediateExecutor()
    method._sampler = FakeSampler(funcs or [])
    method._evaluator = FakeEvaluator(scores or [])
    method._sample_lock = None
    return method


class EoHMechanicsTest(unittest.TestCase):
    def test_initialization_runs_two_pop_size_i1_attempts_and_keeps_top_valid(self):
        method = make_method(
            pop_size=2,
            funcs=[make_function(i) for i in range(4)],
            scores=[1.0, None, 3.0, 2.0],
        )

        method._initialize_population()

        self.assertEqual(len(method._sampler.prompts), 4)
        self.assertTrue(all("describe your new algorithm" in p for p in method._sampler.prompts))
        self.assertEqual([func.score for func in method._population.population], [3.0, 2.0])
        self.assertEqual(method._tot_sample_nums, 0)
        self.assertEqual(method._population.generation, 0)

    def test_valid_evolution_sample_counts_without_profiler_and_immediately_survives(self):
        method = make_method(pop_size=1, funcs=[make_function(2)], scores=[2.0])
        method._population.register_function(make_function(1, 1.0), increment_generation=False)
        method._operators = ["m1"]
        method._operator_weights = [1.0]

        accepted = method._run_one_evolution_sample()

        self.assertTrue(accepted)
        self.assertEqual(method._tot_sample_nums, 1)
        self.assertEqual([func.score for func in method._population.population], [2.0])

    def test_invalid_generation_or_score_does_not_count_or_enter_population(self):
        method = make_method(pop_size=2, funcs=[make_function(2)], scores=[None])
        method._population.register_function(make_function(1, 1.0), increment_generation=False)
        method._operators = ["m1"]
        method._operator_weights = [1.0]

        accepted = method._run_one_evolution_sample()

        self.assertFalse(accepted)
        self.assertEqual(method._tot_sample_nums, 0)
        self.assertEqual([func.score for func in method._population.population], [1.0])
        self.assertEqual(method._population.generation, 0)

    def test_operator_selection_uses_weighted_random_choice(self):
        method = make_method()
        method._operators = ["e1", "m3"]
        method._operator_weights = [0.25, 0.75]

        with patch("baselines.eoh.eoh.random.choices", return_value=["m3"]) as choices:
            operator = method._select_operator()

        self.assertEqual(operator, "m3")
        choices.assert_called_once_with(["e1", "m3"], weights=[0.25, 0.75], k=1)

    def test_population_selection_uses_original_rank_priority_formula(self):
        pop = Population(pop_size=3)
        for score in [3.0, 2.0, 1.0]:
            pop.register_function(make_function(int(score), score), increment_generation=False)

        with patch("baselines.eoh.population.np.random.choice") as choice:
            choice.return_value = pop.population[0]
            selected = pop.selection()

        self.assertIs(selected, pop.population[0])
        probs = choice.call_args.kwargs["p"]
        expected = np.array([1 / 4, 1 / 5, 1 / 6], dtype=float)
        expected = expected / expected.sum()
        np.testing.assert_allclose(probs, expected)

    def test_m3_prompt_and_dispatch_are_available_but_not_default(self):
        llm = FakeLLM()
        method = EoH(
            llm=llm,
            evaluation=FakeEvaluation(),
            profiler=None,
            max_generations=None,
            max_sample_nums=1,
            pop_size=2,
        )
        try:
            self.assertEqual(method._operators, ["e1", "e2", "m1", "m2"])
        finally:
            method._evaluation_executor.shutdown(cancel_futures=True)

        method = make_method(pop_size=1, funcs=[make_function(2)], scores=[2.0])
        method._population.register_function(make_function(1, 1.0), increment_generation=False)
        method._operators = ["m3"]
        method._operator_weights = [1.0]

        accepted = method._run_one_evolution_sample()

        self.assertTrue(accepted)
        self.assertIn("simplify", method._sampler.prompts[-1].lower())
        self.assertIn("return 1", method._sampler.prompts[-1])

    def test_failed_generation_does_not_count(self):
        method = make_method(pop_size=1, funcs=[], scores=[])
        method._population.register_function(make_function(1, 1.0), increment_generation=False)
        method._operators = ["m1"]
        method._operator_weights = [1.0]

        accepted = method._run_one_evolution_sample()

        self.assertFalse(accepted)
        self.assertEqual(method._tot_sample_nums, 0)
        self.assertTrue(math.isfinite(method._population.population[0].score))

    def test_max_sample_nums_takes_precedence_over_generation_budget(self):
        method = make_method(pop_size=2)
        method._max_sample_nums = 20
        method._max_generations = 5

        self.assertEqual(method._sample_budget(), 20)

        method._max_sample_nums = None
        self.assertEqual(method._sample_budget(), 10)

    def test_run_enters_evolution_when_initialization_gets_one_valid_individual(self):
        method = make_method(
            pop_size=2,
            funcs=[make_function(i) for i in range(4)],
            scores=[1.0, None, None, None],
        )
        method._sampler.llm = FakeLLM()
        calls = []

        def fake_sampling(fn, *args, **kwargs):
            calls.append(fn)

        method._multi_threaded_sampling = fake_sampling

        with patch("builtins.print"):
            method.run()

        self.assertEqual(len(method._population), 1)
        self.assertEqual(calls, [method._iteratively_use_eoh_operator])
        self.assertTrue(method._sampler.llm.closed)


if __name__ == "__main__":
    unittest.main()
