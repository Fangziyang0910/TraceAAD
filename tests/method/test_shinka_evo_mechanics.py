import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import Evaluation, Function, LLM
from baselines.shinka_evo import ShinkaArchive, ShinkaEvo, ShinkaEvoProfiler, ShinkaLLMBandit, ShinkaProgram


def code_block(value: int, *, name="heuristic_v2", args="y") -> str:
    return f"""```python
def {name}({args}):
    \"\"\"changed by model\"\"\"
    return {value}
```"""


def diff_block(search: str, replace: str) -> str:
    return "\n".join([
        "<" * 7 + " SEARCH",
        search,
        "=" * 7,
        replace,
        ">" * 7 + " REPLACE",
    ])


class ScriptedLLM(LLM):
    def __init__(self, responses=None):
        super().__init__()
        self.responses = list(responses or [])
        self.prompts = []
        self.closed = False

    def draw_sample(self, prompt, *args, **kwargs):
        self.prompts.append(str(prompt))
        if not self.responses:
            return "invalid generation"
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class FakeEvaluation(Evaluation):
    def __init__(self, results=None):
        super().__init__(
            template_program=(
                "import numpy as np\n"
                "\n"
                "def heuristic(x: np.ndarray) -> int:\n"
                "    \"\"\"Pick the next item from x.\"\"\"\n"
                "    return int(x[0])\n"
            ),
            task_description="Design a heuristic.",
            safe_evaluate=False,
        )
        self.results = list(results or [])
        self.programs = []

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.programs.append(program_str)
        if not self.results:
            return None
        return self.results.pop(0)


def make_method(
        *,
        llm=None,
        novelty_llm=None,
        meta_llm=None,
        results=None,
        embedding_fn=None,
        max_sample_nums=20,
        num_generations=5,
        **kwargs,
):
    llm = llm or ScriptedLLM()
    evaluation = FakeEvaluation(results=results)
    method = ShinkaEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=kwargs.pop("profiler", None),
        max_sample_nums=max_sample_nums,
        num_generations=num_generations,
        novelty_llm=novelty_llm,
        meta_llm=meta_llm,
        embedding_fn=embedding_fn,
        random_seed=0,
        **kwargs,
    )
    return method, llm, evaluation


def make_program(label: int, *, score=1.0, island=0, correct=True, embedding=None) -> ShinkaProgram:
    func = Function(name="heuristic", args="x", body=f"    return {label}")
    program = f"def heuristic(x):\n    return {label}\n"
    return ShinkaProgram.create(
        func,
        program,
        island_idx=island,
        generation=label,
        combined_score=score,
        correct=correct,
        embedding=embedding,
        program_id=f"p{label}",
    )


class DeterministicRng:
    def choice(self, seq):
        return seq[-1]

    def choices(self, population, weights=None, k=1):
        idx = max(range(len(population)), key=lambda i: weights[i] if weights else i)
        return [population[idx]]

    def shuffle(self, seq):
        return None

    def random(self):
        return 1.0

    def randrange(self, n):
        return n - 1


class ShinkaEvoMechanicsTest(unittest.TestCase):
    def test_seed_evaluated_once_and_dict_result_mapping(self):
        result = {
            "combined_score": 3.5,
            "correct": True,
            "public_metrics": {"gap": 1.0},
            "private_metrics": {"hidden": 2.0},
            "text_feedback": "stable",
        }
        method, _, evaluation = make_method(results=[result], num_islands=3)

        seed = method._evaluate_seed()

        self.assertEqual(method._tot_sample_nums, 1)
        self.assertEqual(len(evaluation.programs), 1)
        self.assertEqual(len(method._archive.all_programs()), 3)
        self.assertEqual(seed.combined_score, 3.5)
        self.assertEqual(seed.public_metrics, {"gap": 1.0})
        self.assertEqual(seed.private_metrics, {"hidden": 2.0})
        self.assertEqual(seed.text_feedback, "stable")
        self.assertEqual(seed.function.operator, "seed")

    def test_scalar_result_mapping_marks_invalid_scores_incorrect(self):
        self.assertEqual(ShinkaEvo._coerce_evaluation_result(2.0)["combined_score"], 2.0)
        invalid = ShinkaEvo._coerce_evaluation_result(None)
        self.assertFalse(invalid["correct"])
        self.assertEqual(invalid["combined_score"], 0.0)

    def test_weighted_parent_selection_uses_median_mad_sigmoid_and_children(self):
        archive = ShinkaArchive(parent_selection_lambda=1.0, rng=DeterministicRng())
        low = make_program(1, score=1.0)
        mid = make_program(2, score=2.0)
        high = make_program(3, score=3.0)
        outside_archive = make_program(4, score=100.0)
        high.children_count = 3
        for program in [low, mid, high]:
            archive.add_program(program)
        archive.add_program(outside_archive, update_archive=False)

        probabilities = dict((program.id, prob) for program, prob in archive.weighted_parent_probabilities(0))

        # median=2, MAD=1. high gets sigmoid(1)/(1+3), mid gets sigmoid(0).
        expected_high_weight = 0.7310585786300049 / 4
        expected_mid_weight = 0.5
        expected_low_weight = 0.2689414213699951
        total = expected_high_weight + expected_mid_weight + expected_low_weight
        self.assertAlmostEqual(probabilities["p3"], expected_high_weight / total)
        self.assertAlmostEqual(probabilities["p2"], expected_mid_weight / total)
        self.assertAlmostEqual(probabilities["p1"], expected_low_weight / total)
        self.assertNotIn("p4", probabilities)

    def test_weighted_parent_selection_falls_back_when_archive_is_empty(self):
        archive = ShinkaArchive(archive_size=0, rng=DeterministicRng())
        best_other_island = make_program(1, score=10.0, island=1)
        local_first = make_program(2, score=2.0, island=0)
        local_second = make_program(3, score=3.0, island=0)
        for program in [best_other_island, local_first, local_second]:
            archive.add_program(program)

        self.assertEqual(archive.weighted_parent_probabilities(0), [])
        self.assertIs(archive.sample_parent(0), local_second)

    def test_unknown_parent_strategy_raises(self):
        archive = ShinkaArchive(parent_selection_strategy="unknown")
        archive.add_program(make_program(1, score=1.0))

        with self.assertRaisesRegex(ValueError, "Unknown parent selection strategy"):
            archive.sample_parent(0)

    def test_fix_mode_randomly_samples_incorrect_program(self):
        archive = ShinkaArchive(rng=DeterministicRng())
        first = make_program(1, correct=False, island=0)
        second = make_program(2, correct=False, island=0)
        archive.add_program(first)
        archive.add_program(second)

        parent, fix_mode = archive.sample_parent_with_fix_mode(0)

        self.assertTrue(fix_mode)
        self.assertIs(parent, second)

    def test_power_law_and_beam_search_parent_selection(self):
        archive = ShinkaArchive(
            parent_selection_strategy="power_law",
            exploitation_alpha=0.0,
            rng=DeterministicRng(),
        )
        first = make_program(1, score=1.0)
        second = make_program(2, score=2.0)
        archive.add_program(first)
        archive.add_program(second)

        self.assertIs(archive.sample_parent(0), first)

        beam = ShinkaArchive(parent_selection_strategy="beam_search", num_beams=1, rng=DeterministicRng())
        weak = make_program(3, score=3.0)
        strong = make_program(4, score=4.0)
        beam.add_program(weak)
        beam.add_program(strong)
        self.assertIs(beam.sample_parent(0), strong)
        strong.children_count = 0
        self.assertIs(beam.sample_parent(0), strong)

    def test_inspiration_sampling_obeys_island_separation_and_top_k(self):
        archive = ShinkaArchive(num_archive_inspirations=1, num_top_k_inspirations=1, rng=DeterministicRng())
        parent = make_program(1, score=1.0, island=0)
        best_same = make_program(2, score=5.0, island=0)
        other_island = make_program(3, score=9.0, island=1)
        random_same = make_program(4, score=3.0, island=0)
        for program in [parent, best_same, other_island, random_same]:
            archive.add_program(program)

        archive_inspirations, top_k = archive.sample_inspirations(parent)

        self.assertIn(best_same, archive_inspirations)
        self.assertNotIn(other_island, archive_inspirations)
        self.assertEqual(top_k, [random_same])

    def test_cross_patch_is_filtered_without_inspirations(self):
        method, _, _ = make_method(patch_types=("cross", "full"), patch_type_probs=(0.9, 0.1))
        with patch("numpy.random.choice") as choice:
            choice.return_value = "full"
            selected = method._sampler.sample_patch_type([], [])

        self.assertEqual(selected, "full")
        self.assertNotIn("cross", choice.call_args.args[0])

    def test_diff_patch_rejects_immutable_edits_and_accepts_mutable_body_edits(self):
        method, _, _ = make_method()
        seed = method._evaluate_seed()
        immutable = diff_block("def heuristic(x: np.ndarray) -> int:", "def heuristic(y) -> int:")
        rejected = method._sampler.apply_response(immutable, seed, "diff")
        self.assertFalse(rejected.success)
        self.assertIn("immutable", rejected.error)

        valid = diff_block("    return int(x[0])", "    return 7")
        accepted = method._sampler.apply_response(valid, seed, "diff")
        self.assertTrue(accepted.success)
        self.assertEqual(accepted.function.name, "heuristic")
        self.assertEqual(accepted.function.args, "x: np.ndarray")
        self.assertIn("return 7", accepted.function.body)

    def test_full_cross_and_fix_preserve_template_signature_docstring_and_imports(self):
        method, _, _ = make_method()
        seed = method._evaluate_seed()
        for patch_type, fix_mode in [("full", False), ("cross", False), ("fix", True)]:
            result = method._sampler.apply_response(code_block(9), seed, patch_type, fix_mode=fix_mode)
            self.assertTrue(result.success)
            self.assertEqual(result.function.name, "heuristic")
            self.assertEqual(result.function.args, "x: np.ndarray")
            self.assertIn("Pick the next item", result.function.docstring)
            self.assertIn("import numpy as np", str(result.program))
            self.assertIn("return 9", result.function.body)

    def test_novelty_rejection_happens_before_evaluation_and_does_not_spend_budget(self):
        llm = ScriptedLLM([code_block(2)])
        novelty_llm = ScriptedLLM(["NOT NOVEL: identical structure"])
        method, _, evaluation = make_method(
            llm=llm,
            novelty_llm=novelty_llm,
            results=[1.0],
            embedding_fn=lambda code: [1.0, 0.0],
            max_sample_nums=2,
            num_generations=1,
            max_novelty_attempts=1,
            patch_types=("full",),
            patch_type_probs=(1.0,),
        )
        method.run()

        self.assertEqual(method._tot_sample_nums, 1)
        self.assertEqual(len(evaluation.programs), 1)
        self.assertIn("meaningfully novel", novelty_llm.prompts[0])

    def test_archive_fitness_replacement_and_crowding_fallback(self):
        archive = ShinkaArchive(archive_size=1)
        low = make_program(1, score=1.0)
        high = make_program(2, score=2.0)
        archive.add_program(low)
        archive.add_program(high)
        self.assertEqual(archive.archive_ids, ["p2"])

        crowding = ShinkaArchive(archive_size=1, archive_selection_strategy="crowding")
        crowding.add_program(make_program(3, score=3.0))
        better_without_embedding = make_program(4, score=4.0)
        crowding.add_program(better_without_embedding)
        self.assertEqual(crowding.archive_ids, ["p4"])

    def test_bandit_relative_improvement_and_single_llm_fixed_selection(self):
        llm_a = ScriptedLLM()
        llm_b = ScriptedLLM()
        bandit = ShinkaLLMBandit([llm_a, llm_b], seed=0, auto_decay=None)
        update = bandit.update(arm=1, reward=5.0, baseline=3.0)
        self.assertEqual(update.shifted_reward, 2.0)
        self.assertEqual(bandit.total_reward[1], 2.0)

        fixed = ShinkaLLMBandit([llm_a])
        arm, selected, meta = fixed.select()
        self.assertEqual(arm, 0)
        self.assertIs(selected, llm_a)
        self.assertEqual(meta["probabilities"], [1.0])

    def test_bandit_uniformly_explores_unseen_llm_arms(self):
        llm_a = ScriptedLLM()
        llm_b = ScriptedLLM()
        bandit = ShinkaLLMBandit([llm_a, llm_b], seed=0, epsilon=0.0, auto_decay=None)

        first_arm, _, first_meta = bandit.select()
        second_arm, _, second_meta = bandit.select()

        self.assertEqual(first_meta["probabilities"], [0.5, 0.5])
        self.assertNotEqual(first_arm, second_arm)
        self.assertEqual(second_meta["probabilities"][second_arm], 1.0)

    def test_bandit_parameters_are_forwarded_from_method(self):
        llm_a = ScriptedLLM()
        llm_b = ScriptedLLM()
        method, _, _ = make_method(
            llm=llm_a,
            llms=[llm_a, llm_b],
            llm_ucb_exploration_coef=2.5,
            llm_ucb_epsilon=0.0,
            llm_ucb_auto_decay=None,
        )

        self.assertEqual(method._bandit.exploration_coef, 2.5)
        self.assertEqual(method._bandit.epsilon, 0.0)
        self.assertIsNone(method._bandit.auto_decay)

    def test_resample_reselects_parent_context_and_patch_type(self):
        llm = ScriptedLLM(["not a function", code_block(2)])
        method, _, _ = make_method(
            llm=llm,
            results=[1.0, 2.0],
            max_patch_resamples=2,
            max_novelty_attempts=1,
            patch_types=("full",),
            patch_type_probs=(1.0,),
        )
        method._evaluate_seed()
        calls = []
        original = method._select_parent_context

        def counted_parent_context():
            calls.append(True)
            return original()

        method._select_parent_context = counted_parent_context
        method._run_generation(1)

        self.assertEqual(len(calls), 2)
        self.assertEqual(method._tot_sample_nums, 2)

    def test_meta_scratchpad_updates_and_recommendation_enters_prompt(self):
        llm = ScriptedLLM([code_block(2), code_block(3)])
        meta_llm = ScriptedLLM(["summary", "insights", "1. prefer sparse logic\n2. avoid noise"])
        method, _, _ = make_method(
            llm=llm,
            meta_llm=meta_llm,
            results=[1.0, 2.0],
            max_sample_nums=2,
            num_generations=1,
            meta_rec_interval=1,
            patch_types=("full",),
            patch_type_probs=(1.0,),
        )
        method.run()

        self.assertEqual(method._meta_summary, "summary")
        self.assertEqual(method._meta_scratch_pad, "insights")
        self.assertTrue(method._current_meta_recommendation())
        parent = method.best_program
        prompt = method._sampler.build_prompt(parent, [], [], "full", meta_recommendations=method._current_meta_recommendation())
        self.assertIn("Potential Recommendations", prompt)

    def test_profiler_records_functions_and_shinka_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = ShinkaEvoProfiler(log_dir=tmpdir, create_random_path=False)
            llm = ScriptedLLM([code_block(2)])
            method, _, _ = make_method(
                llm=llm,
                profiler=profiler,
                results=[1.0, 2.0],
                max_sample_nums=2,
                num_generations=1,
                patch_types=("full",),
                patch_type_probs=(1.0,),
            )
            method.run()

            self.assertTrue((Path(tmpdir) / "samples" / "samples_1~200.json").exists())
            self.assertTrue((Path(tmpdir) / "shinka_evo" / "patch_attempt.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "shinka_evo" / "bandit_update.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
