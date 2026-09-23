import unittest
from pathlib import Path
from unittest.mock import patch

from core import Evaluation, Function, LLM
from baselines.reevo.prompt import ReEvoPrompt
from baselines.reevo.reevo import ReEvo


def make_function(label: int, score=None) -> Function:
    func = Function(name="heuristic", args="x", body=f"    return {label}")
    func.score = score
    func.algorithm = f"algorithm-{label}"
    return func


def make_code(label: int) -> str:
    return f"def heuristic(x):\n    return {label}\n"


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
    def __init__(self, code_samples=None, reflections=None, temperature=None):
        super().__init__()
        self.code_samples = list(code_samples or [])
        self.reflections = list(reflections or [])
        self.prompts = []
        self.closed = False
        if temperature is not None:
            self.temperature = temperature

    def draw_sample(self, prompt, *args, **kwargs):
        self.prompts.append((prompt, kwargs))
        prompt_text = str(prompt)
        if "Your task is to give hints" in prompt_text:
            if self.reflections:
                return self.reflections.pop(0)
            return "reflection"
        if not self.code_samples:
            return "invalid generation"
        return self.code_samples.pop(0)

    def close(self):
        self.closed = True


class FakeEvaluation(Evaluation):
    def __init__(self, scores=None):
        super().__init__(
            template_program="def heuristic(x):\n    return x\n",
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
        scores=None,
        code_samples=None,
        reflections=None,
        pop_size=2,
        init_pop_size=2,
        mutation_rate=0.5,
        max_sample_nums=100,
        llm_temperature=None,
) -> tuple[ReEvo, ScriptedLLM, FakeEvaluation]:
    llm = ScriptedLLM(code_samples=code_samples, reflections=reflections, temperature=llm_temperature)
    evaluation = FakeEvaluation(scores=scores)
    method = ReEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=None,
        max_sample_nums=max_sample_nums,
        pop_size=pop_size,
        init_pop_size=init_pop_size,
        mutation_rate=mutation_rate,
        num_samplers=1,
        num_evaluators=1,
    )
    method._evaluation_executor.shutdown(cancel_futures=True)
    method._evaluation_executor = ImmediateExecutor()
    return method, llm, evaluation


class ReEvoMechanicsTest(unittest.TestCase):
    def test_initialization_evaluates_template_seed_and_samples_init_pop_size(self):
        method, llm, _ = make_method(
            scores=[100.0, 1.0, 2.0, 3.0],
            code_samples=[make_code(1), make_code(2), make_code(3)],
            pop_size=2,
            init_pop_size=3,
            llm_temperature=1.0,
        )

        method._evaluate_seed()
        method._initialize_population()

        self.assertEqual(method._tot_sample_nums, 4)
        self.assertEqual(method._elite_function.score, 100.0)
        self.assertEqual([func.score for func in method._population.population], [1.0, 2.0, 3.0])
        self.assertNotIn(100.0, [func.score for func in method._population.population])
        self.assertEqual(len(llm.prompts), 3)
        self.assertTrue(all("trivial design above" in prompt for prompt, _ in llm.prompts))
        self.assertTrue(all(kwargs == {"temperature": 1.3} for _, kwargs in llm.prompts))

    def test_parent_selection_matches_original_random_pairing(self):
        method, _, _ = make_method()
        parent_a = make_function(1, score=1.0)
        parent_b = make_function(2, score=1.0)
        parent_c = make_function(3, score=2.0)
        method._population.set_population([parent_a, parent_b, parent_c], increment_generation=False)
        method._elite_function = None

        with patch("baselines.reevo.reevo.np.random.choice") as choice:
            choice.side_effect = [
                [parent_a, parent_b],
                [parent_a, parent_c],
                [parent_c, parent_a],
            ]
            pairs = method._select_parent_pairs()

        self.assertEqual(pairs, [[parent_a, parent_c], [parent_c, parent_a]])
        self.assertEqual(choice.call_count, 3)
        self.assertEqual(choice.call_args_list[0].kwargs["size"], 2)
        self.assertFalse(choice.call_args_list[0].kwargs["replace"])

    def test_parent_selection_falls_back_when_all_scores_tied(self):
        method, _, _ = make_method(pop_size=2)
        parent_a = make_function(1, score=1.0)
        parent_b = make_function(2, score=1.0)
        parent_c = make_function(3, score=1.0)
        method._population.set_population([parent_a, parent_b, parent_c], increment_generation=False)
        method._elite_function = None

        with patch("baselines.reevo.reevo.np.random.choice") as choice:
            choice.side_effect = [
                [parent_a, parent_b],
                [parent_b, parent_c],
            ]
            pairs = method._select_parent_pairs()

        self.assertEqual(pairs, [[parent_a, parent_b], [parent_b, parent_c]])
        self.assertEqual(choice.call_count, 2)

    def test_short_reflection_orders_worse_then_better_by_high_score(self):
        worse = make_function(1, score=1.0)
        better = make_function(2, score=2.0)

        prompt = ReEvoPrompt.get_short_term_reflection_prompt("Design a heuristic.", [better, worse])

        self.assertIn("second version performs better", prompt)
        self.assertLess(prompt.index("[Worse code]"), prompt.index("return 1"))
        self.assertLess(prompt.index("return 1"), prompt.index("[Better code]"))
        self.assertLess(prompt.index("[Better code]"), prompt.index("return 2"))

    def test_generation_replaces_with_crossover_then_appends_mutation(self):
        method, _, _ = make_method(
            scores=[10.0, 11.0, 12.0],
            code_samples=[make_code(10), make_code(11), make_code(12)],
            reflections=["short-1", "short-2", "long"],
            pop_size=2,
            init_pop_size=0,
            mutation_rate=0.5,
        )
        parent_a = make_function(1, score=1.0)
        parent_b = make_function(2, score=2.0)
        method._population.set_population([parent_a, parent_b], increment_generation=False)
        method._elite_function = parent_b

        with patch("baselines.reevo.reevo.np.random.choice") as choice:
            choice.side_effect = [[parent_a, parent_b], [parent_b, parent_a]]
            method._run_evolution_generation()

        self.assertEqual([func.score for func in method._population.population], [10.0, 11.0, 12.0])
        self.assertEqual(method._elite_function.score, 12.0)
        self.assertEqual(method._tot_sample_nums, 3)

    def test_long_reflection_receives_all_current_short_reflections(self):
        reflections = [f"short-{i}" for i in range(10)]
        method, llm, _ = make_method(reflections=["long-reflection"], pop_size=10)

        method._long_term_reflection(reflections)

        prompt = llm.prompts[-1][0]
        for reflection in reflections:
            self.assertIn(reflection, prompt)
        self.assertEqual(method._long_term_reflection_str, "long-reflection")

    def test_invalid_candidates_count_budget_but_do_not_enter_population(self):
        method, _, _ = make_method(
            scores=[None, 3.0],
            code_samples=["not valid python", make_code(2), make_code(3)],
            pop_size=2,
            init_pop_size=3,
        )

        method._initialize_population()

        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual([func.score for func in method._population.population], [3.0])
        with self.assertRaisesRegex(RuntimeError, "fewer than two valid functions"):
            method._select_parent_pairs()

    def test_defaults_match_original_reevo(self):
        method, _, _ = make_method(init_pop_size=30, pop_size=10)
        default_method, _, _ = make_method()
        default_method._evaluation_executor.shutdown(cancel_futures=True)

        llm = ScriptedLLM()
        evaluation = FakeEvaluation()
        constructed = ReEvo(llm=llm, evaluation=evaluation, profiler=None)
        constructed._evaluation_executor.shutdown(cancel_futures=True)

        self.assertEqual(constructed._pop_size, 10)
        self.assertEqual(constructed._init_pop_size, 30)
        self.assertEqual(constructed._max_sample_nums, 100)
        self.assertEqual(constructed._mutation_rate, 0.5)

        paras = Path("baselines/reevo/paras.yaml").read_text()
        self.assertIn("max_sample_nums: 100", paras)
        self.assertIn("pop_size: 10", paras)
        self.assertIn("init_pop_size: 30", paras)
        self.assertIn("mutation_rate: 0.5", paras)

        method._evaluation_executor.shutdown(cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
