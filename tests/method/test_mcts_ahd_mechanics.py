import math
import json
import random
import tempfile
import unittest
from pathlib import Path

from core import Function, TextFunctionProgramConverter
from baselines.mcts_ahd.mcts import MCTS, MCTSNode
from baselines.mcts_ahd.mcts_ahd import MCTS_AHD
from baselines.mcts_ahd.population import Population
from baselines.mcts_ahd.prompt import MAPrompt
from baselines.mcts_ahd.profiler import MAProfiler


def make_function(label: int, score=None) -> Function:
    func = Function(name="heuristic", args="x", body=f"    return {label}")
    func.algorithm = f"algorithm-{label}"
    func.score = score
    return func


def make_method() -> MCTS_AHD:
    method = object.__new__(MCTS_AHD)
    method._task_description_str = "Design a heuristic."
    method._function_to_evolve = make_function(0)
    method._template_program = TextFunctionProgramConverter.text_to_program(str(method._function_to_evolve))
    method._population = Population(init_pop_size=4, pop_size=10)
    method._pop_size = 10
    method._e1_min_refs = MCTS_AHD.E1_MIN_REFS
    method._e1_max_refs = MCTS_AHD.E1_MAX_REFS
    method._max_sample_nums = 1000
    method._tot_sample_nums = 0
    method._consecutive_sample_failures = 0
    method._max_consecutive_sample_failures = 20
    method._search_aborted = False
    method._debug_mode = False
    method._profiler = None
    return method


def attach_node(parent: MCTSNode, func: Function, depth: int) -> MCTSNode:
    node = MCTSNode(
        func.algorithm,
        str(func),
        -1 * func.score,
        individual=func,
        parent=parent,
        depth=depth,
        visit=1,
        Q=func.score,
        raw_info=func,
    )
    parent.add_child(node)
    return node


class ImmediateResult:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        return ImmediateResult(fn(*args, **kwargs))


class FakeSampler:
    def __init__(self, func: Function):
        self.func = func

    def get_thought_and_function(self, task_description, prompt, **kwargs):
        return "aligned description", self.func


class FailingSampler:
    def __init__(self, exc: Exception):
        self.exc = exc

    def get_thought_and_function(self, task_description, prompt, **kwargs):
        raise self.exc


class FakeEvaluator:
    def __init__(self, score):
        self.score = score

    def evaluate_program_record_time(self, program):
        return self.score, 0.01


class MCTSAHDMechanicsTest(unittest.TestCase):
    def test_child_depth_increments_from_parent(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, make_function(1, 1.0), depth=3)
        parent.subtree.append(parent)
        method._sample_evaluate_register = lambda prompt, func_only=False, **kwargs: make_function(2, 2.0)

        method.expand(mcts, [], parent, "m1")

        self.assertEqual(parent.children[-1].depth, 4)
        self.assertEqual(mcts.max_depth, 10)

    def test_e1_samples_two_to_five_root_subtree_refs(self):
        random.seed(7)
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        for i in range(6):
            node = attach_node(mcts.root, make_function(i, float(i)), depth=1)
            node.subtree.append(node)

        refs = method._sample_e1_references_from_root(mcts)

        self.assertGreaterEqual(len(refs), 2)
        self.assertLessEqual(len(refs), 5)

    def test_e1_uses_exactly_two_refs_when_two_root_subtrees_exist(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        for i in range(2):
            node = attach_node(mcts.root, make_function(i, float(i)), depth=1)
            node.subtree.append(node)

        refs = method._sample_e1_references_from_root(mcts)

        self.assertEqual(len(refs), 2)

    def test_progressive_root_e1_allows_single_root_subtree(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        node = attach_node(mcts.root, make_function(1, 1.0), depth=1)
        node.subtree.append(node)
        before = len(mcts.root.children)
        captured = {}

        def sample(prompt, func_only=False, **kwargs):
            captured["prompt"] = prompt
            captured["operator"] = kwargs["operator"]
            return make_function(2, 2.0)

        method._sample_evaluate_register = sample

        method.expand(mcts, mcts.root.children, mcts.root, "e1")

        self.assertEqual(captured["operator"], "e1")
        self.assertIn("algorithm-1", captured["prompt"])
        self.assertEqual(len(mcts.root.children), before + 1)

    def test_s1_expand_checks_duplicates_against_path_set(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, make_function(1, 1.0), depth=1)
        parent.subtree.append(parent)
        leaf = attach_node(parent, make_function(2, 2.0), depth=2)
        method._sample_evaluate_register = lambda prompt, func_only=False, **kwargs: make_function(3, 3.0)

        method.expand(mcts, [], leaf, "s1")

        self.assertEqual(len(leaf.children), 1)
        self.assertEqual(leaf.children[0].individual.score, 3.0)

    def test_eval_counter_increments_without_profiler(self):
        method = make_method()
        method._sampler = FakeSampler(make_function(3))
        method._evaluator = FakeEvaluator(score=3.5)
        method._evaluation_executor = ImmediateExecutor()

        func = method._sample_evaluate_register("prompt", func_only=True)

        self.assertEqual(method._tot_sample_nums, 1)
        self.assertEqual(func.score, 3.5)

    def test_sample_register_sets_operator(self):
        method = make_method()
        method._sampler = FakeSampler(make_function(3))
        method._evaluator = FakeEvaluator(score=3.5)
        method._evaluation_executor = ImmediateExecutor()

        func = method._sample_evaluate_register("prompt", func_only=True, operator="m1")

        self.assertEqual(func.operator, "m1")

    def test_prompt_contract_uses_reference_style_names(self):
        template = Function(
            name="select_next_node",
            args="current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray",
            body="    return current_node",
        )

        prompt = MAPrompt.get_prompt_i1("Task.", template)

        self.assertIn("function named 'select_next_node'", prompt)
        self.assertIn("'current_node'", prompt)
        self.assertIn("'destination_node'", prompt)
        self.assertIn("'unvisited_nodes'", prompt)
        self.assertIn("'distance_matrix'", prompt)
        self.assertNotIn("'current_node: int'", prompt)
        self.assertIn("return 1 output(s): 'next_node'", prompt)

    def test_sample_exception_is_invalid_sample_without_incrementing_count(self):
        method = make_method()
        method._sampler = FailingSampler(TimeoutError("request timed out"))
        method._evaluator = FakeEvaluator(score=3.5)
        method._evaluation_executor = ImmediateExecutor()
        method._max_consecutive_sample_failures = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            method._profiler = MAProfiler(log_dir=tmpdir, create_random_path=False, log_style="simple")
            result = method._sample_evaluate_register("prompt", func_only=True, operator="s1")
            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "mcts_events.jsonl").read_text().splitlines()
            ]
            llm_calls = [
                json.loads(line)
                for line in (Path(tmpdir) / "llm_calls.jsonl").read_text().splitlines()
            ]

        self.assertFalse(result)
        self.assertEqual(method._tot_sample_nums, 0)
        self.assertEqual(method._consecutive_sample_failures, 1)
        self.assertFalse(method._search_aborted)
        self.assertEqual(events[-1]["event"], "sample_error")
        self.assertEqual(events[-1]["error_type"], "TimeoutError")
        self.assertEqual(llm_calls[-1]["stage"], "sample_error")

    def test_consecutive_sample_failures_abort_search_loop(self):
        method = make_method()
        method._sampler = FailingSampler(TimeoutError("request timed out"))
        method._evaluator = FakeEvaluator(score=3.5)
        method._evaluation_executor = ImmediateExecutor()
        method._max_consecutive_sample_failures = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            method._profiler = MAProfiler(log_dir=tmpdir, create_random_path=False, log_style="simple")
            result = method._sample_evaluate_register("prompt", func_only=True, operator="s1")

        self.assertFalse(result)
        self.assertTrue(method._search_aborted)
        self.assertFalse(method._continue_loop())

    def test_duplicate_e1_score_skips_child_and_backprop(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        for i, score in enumerate([1.0, 2.0], start=1):
            node = attach_node(mcts.root, make_function(i, score), depth=1)
            node.subtree.append(node)
        method._sample_evaluate_register = lambda prompt, func_only=False, **kwargs: make_function(9, 1.0)
        before_children = len(mcts.root.children)
        before_visits = mcts.root.visits

        method.expand(mcts, mcts.root.children, mcts.root, "e1")

        self.assertEqual(len(mcts.root.children), before_children)
        self.assertEqual(mcts.root.visits, before_visits)

    def test_progressive_widening_requires_threshold_greater_than_children(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        node = MCTSNode("parent", "code", 0, depth=1, visit=4, Q=0)
        node.children = [
            MCTSNode("child-1", "code-1", 0, parent=node, visit=1, Q=0),
            MCTSNode("child-2", "code-2", 0, parent=node, visit=1, Q=0),
        ]

        self.assertFalse(method._should_progressively_widen(mcts, node))
        node.children.pop()
        self.assertTrue(method._should_progressively_widen(mcts, node))

    def test_population_management_s1_uses_reference_order_under_negative_scores(self):
        method = make_method()
        best = make_function(1, -1.0)
        middle = make_function(2, -5.0)
        worst = make_function(3, -10.0)

        managed = method.population_management_s1([best, middle, worst], 3)

        self.assertEqual([func.score for func in managed], [-10.0, -5.0, -1.0])

    def test_e2_selects_reference_from_nodes_set_before_population(self):
        method = make_method()
        parent_func = make_function(1, 1.0)
        node_set_reference = make_function(42, 42.0)
        population_reference = make_function(99, 99.0)
        method._population._population = [parent_func, population_reference]
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, parent_func, depth=1)
        captured = {}

        def sample(prompt, func_only=False, **kwargs):
            captured["prompt"] = prompt
            return make_function(2, 2.0)

        method._sample_evaluate_register = sample

        method.expand(mcts, [parent_func, node_set_reference], parent, "e2")

        self.assertIn("algorithm-42", captured["prompt"])
        self.assertNotIn("algorithm-99", captured["prompt"])

    def test_initialization_adds_i1_then_e1_brothers_to_root_before_survival(self):
        method = make_method()
        method._init_pop_size = 4
        method._selection_num = 2
        method._initial_sample_nums_max = 40
        method._population = Population(init_pop_size=4, pop_size=1)
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        generated = [
            make_function(1, 1.0),
            make_function(2, 2.0),
            make_function(3, 3.0),
            make_function(4, 4.0),
        ]
        calls = []

        def sample(prompt, func_only=False, **kwargs):
            calls.append(kwargs["operator"])
            return generated[len(calls) - 1]

        method._sample_evaluate_register = sample

        brothers = method._initialize_mcts_root(mcts)

        self.assertEqual(calls, ["i1", "e1", "e1", "e1"])
        self.assertEqual([func.score for func in brothers], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual([child.individual.score for child in mcts.root.children], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual([func.score for func in method._population.population], [4.0])

    def test_uct_with_equal_q_bounds_does_not_crash(self):
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        child = MCTSNode("child", "code", -1, parent=mcts.root, depth=1, visit=1, Q=1.0)
        mcts.root.visits = 1
        mcts.q_min = 1.0
        mcts.q_max = 1.0

        value = mcts.uct(child, eval_remain=1.0)

        self.assertTrue(math.isfinite(value))

    def test_expansion_schedule_and_tree_preserve_low_score_nodes(self):
        expanded_ops = []
        for op, weight in zip(MCTS_AHD.DEFAULT_OPERATORS, MCTS_AHD.DEFAULT_OPERATOR_WEIGHTS):
            expanded_ops.extend([op] * weight)
        self.assertEqual(expanded_ops, ["e2", "m1", "m1", "m2", "m2", "s1"])

        method = make_method()
        method._population = Population(init_pop_size=4, pop_size=1)
        high = make_function(10, 10.0)
        method._population._population = [high]
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, make_function(1, 1.0), depth=1)
        parent.subtree.append(parent)
        low = make_function(0, 0.0)
        method._sample_evaluate_register = lambda prompt, func_only=False, **kwargs: low

        method.expand(mcts, [], parent, "m1")

        self.assertIs(parent.children[-1].individual, low)
        self.assertEqual(parent.children[-1].individual.score, 0.0)
        self.assertEqual([func.score for func in method._population.population], [10.0])

    def test_e2_samples_pending_elite_before_survival(self):
        method = make_method()
        parent_func = make_function(1, 1.0)
        pending_elite = make_function(99, 99.0)
        method._population._population = [parent_func]
        method._population._next_gen_pop = [pending_elite]
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, parent_func, depth=1)
        captured = {}

        def sample(prompt, func_only=False, **kwargs):
            captured["prompt"] = prompt
            return make_function(2, 2.0)

        method._sample_evaluate_register = sample

        method.expand(mcts, [], parent, "e2")

        self.assertIn("algorithm-99", captured["prompt"])

    def test_profiler_writes_mcts_state_and_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = MAProfiler(log_dir=tmpdir, create_random_path=False, log_style="simple")
            mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
            child = attach_node(mcts.root, make_function(1, 1.5), depth=1)
            child.subtree.append(child)
            mcts.backpropagate(child)

            profiler.log_mcts_state(
                phase="iteration_start",
                sample_order=3,
                max_sample_nums=10,
                mcts=mcts,
            )
            profiler.log_mcts_event(
                event="operator_start",
                status="scheduled",
                operator="m1",
                sample_order=3,
                parent_score=1.5,
            )
            profiler.log_llm_call(
                stage="generate",
                operator="m1",
                sample_order=3,
                prompt="prompt",
                response="response",
            )

            state = json.loads((Path(tmpdir) / "mcts_state.jsonl").read_text().splitlines()[0])
            event = json.loads((Path(tmpdir) / "mcts_events.jsonl").read_text().splitlines()[0])
            llm_call = json.loads((Path(tmpdir) / "llm_calls.jsonl").read_text().splitlines()[0])

        self.assertEqual(state["phase"], "iteration_start")
        self.assertEqual(state["sample_order"], 3)
        self.assertEqual(state["root_children"][0]["score"], 1.5)
        self.assertEqual(state["root_children"][0]["subtree_size"], 1)
        self.assertEqual(event["operator"], "m1")
        self.assertEqual(event["status"], "scheduled")
        self.assertEqual(llm_call["prompt"], "prompt")
        self.assertEqual(llm_call["response"], "response")

    def test_expand_records_mcts_event(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent_func = make_function(1, 1.0)
        parent = attach_node(mcts.root, parent_func, depth=1)
        parent.subtree.append(parent)
        method._tot_sample_nums = 7
        method._sample_evaluate_register = lambda prompt, func_only=False, **kwargs: make_function(2, 2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            method._profiler = MAProfiler(log_dir=tmpdir, create_random_path=False, log_style="simple")
            method.expand(mcts, [], parent, "m1")
            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "mcts_events.jsonl").read_text().splitlines()
            ]

        expanded = [event for event in events if event["event"] == "expand"]
        self.assertEqual(expanded[-1]["status"], "expanded")
        self.assertEqual(expanded[-1]["operator"], "m1")
        self.assertEqual(expanded[-1]["parent_score"], 1.0)
        self.assertEqual(expanded[-1]["child_score"], 2.0)


if __name__ == "__main__":
    unittest.main()
