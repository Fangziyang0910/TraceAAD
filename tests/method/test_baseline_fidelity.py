import copy
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.calm.calm import CALM
from baselines.calm.parse import extract_function_from_string
from baselines.eoh.sampler import EoHSampler
from baselines.funsearch.funsearch import FunSearch
from baselines.funsearch.programs_database import Island
from baselines.mcts_ahd.sampler import MASampler
from baselines.pathwise.graph import PathWiseAction, PathWiseNode
from baselines.pathwise.pathwise import PathWise, _Rollout
from baselines.pathwise.prompt import PathWisePrompt
from baselines.pathwise.sampler import PathWiseSampler
from baselines.reevo.sampler import response_to_function as reevo_function
from baselines.pathwise.graph import PathWiseGraph
from core.code import function_from_source
from baselines.shinka_evo.population import ShinkaProgram
from baselines.shinka_evo.sampler import ShinkaSampler
from core.code import parse_function, parse_program, replace_function
from core.evaluate import EvaluationOutcome

TEMPLATE = "def solve(x):\n    return x\n"

SOURCE = """import math

def helper(x):
    return math.floor(x) + OFFSET

def solve_v2(x):
    return helper(x)

OFFSET = 3
"""

METHODS = ["eoh", "reevo", "mcts_ahd", "pathwise", "calm", "funsearch", "shinka_evo"]


class Replies:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def draw_sample(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return next(self.responses)


def run_function(function, x=2):
    namespace = {}
    exec(str(function), namespace)
    return namespace['solve'](x)


def test_method_parsers_keep_their_own_acceptance_rules():
    template = parse_program(TEMPLATE)
    source = SOURCE.replace('solve_v2', 'solve')
    response = "{A multiline\nidea}\n```python\n" + source + "\n```"
    thought, function = EoHSampler(Replies(response), template).get_thought_and_function('task')
    assert thought == 'A multiline\nidea'
    assert run_function(function, 2.9) == 5
    assert run_function(reevo_function(response.replace('def solve(', 'def solve_v2('), template), 2.9) == 5
    assert run_function(PathWiseSampler.response_to_function(response, template), 2.9) == 5

    calm = CALM.__new__(CALM)
    calm._template_program_str = TEMPLATE
    assert calm._code_to_program(response) == (None, None)  # upstream requires one function
    function, _ = calm._code_to_program('```python\ndef solve(x):\n    return x + 1\n```')
    assert run_function(function) == 3

    funsearch = FunSearch.__new__(FunSearch)
    funsearch._function_name, funsearch._template_program = 'solve', template
    program = funsearch.sample_to_program('    return x + 1\n', 2)
    assert run_function(program.functions[0]) == 3
    program = funsearch.sample_to_program(response, 2)
    assert 'def helper' not in str(program) and 'OFFSET = 3' not in str(program)


def test_response_is_not_repaired_by_rewriting_its_arguments():
    template = parse_program(TEMPLATE)
    response = '{A complete idea}\n```python\ndef solve(y):\n    return y + 1\n```'
    _, function = EoHSampler(Replies(response), template).get_thought_and_function('task')
    assert function.args == 'y'
    assert run_function(replace_function(function, template).functions[0]) == 3


def test_eoh_retries_missing_description_without_treating_a_code_dict_as_thought():
    bad = '```python\ndef solve(x):\n    d = {"value": x}\n    return d["value"]\n```'
    good = '{Return the input value}\n```python\ndef solve(x):\n    return x\n```'
    llm = Replies(bad, good)
    thought, function = EoHSampler(llm, TEMPLATE).get_thought_and_function('task')
    assert len(llm.calls) == 2 and thought == 'Return the input value'
    assert run_function(function) == 2


def test_mcts_keeps_official_return_extraction_and_shared_sampling_controls():
    llm = Replies('{Use one increment}\n```python\ndef solve(x):\n    result = x + 1\n    return result\n```', 'aligned')
    thought, function = MASampler(llm, TEMPLATE).get_thought_and_function('task', 'generate')
    assert thought == 'aligned' and run_function(function) == 3
    assert llm.calls[1][1] == {}


def test_pathwise_requires_metadata_and_rejects_placeholder_rollouts():
    code = '```python\ndef solve_v2(x):\n    return x + 1\n```'
    assert PathWiseSampler.parse_initialization_response(code, TEMPLATE) is None
    assert PathWiseSampler.parse_world_model_response(code, TEMPLATE) is None
    initialized = PathWiseSampler.parse_initialization_response(
        code + '\nDescription: Increment\nDerivation Rationale: Improve', TEMPLATE)
    assert run_function(initialized[0]) == 3
    assert PathWiseSampler.parse_world_model_response(code.replace('return x + 1', 'return x + 1  # TODO') + '\nDescription: Increment', TEMPLATE) is None


def test_shinka_full_patch_preserves_immutable_imports():
    source = 'import math\ndef score(x):\n    return math.floor(x)\n'
    template = parse_program(source)
    parent_function = function_from_source(source, 'score')
    parent = ShinkaProgram.create(parent_function, source, correct=True)
    sampler = ShinkaSampler(None, template, task_description='task')
    response = '```python\nimport os\ndef score(x):\n    # EVOLVE-BLOCK-START\n    return x + 1\n    # EVOLVE-BLOCK-END\n```'
    patch = sampler.apply_response(response, parent, patch_type='full')
    assert patch.success, patch.error
    assert 'import math' in str(patch.program) and 'import os' not in str(patch.program)
    namespace = {}
    exec(str(patch.program), namespace)
    assert namespace['score'](2) == 3


def test_complete_solver_diff_can_edit_helpers_and_preserves_module_state():
    template = parse_program(TEMPLATE)
    function = function_from_source(SOURCE.replace("solve_v2", "solve"), "solve")
    parent = ShinkaProgram.create(function, str(function), correct=True)
    sampler = ShinkaSampler(None, template, task_description="task")
    response = "<<<<<<< SEARCH\nOFFSET = 3\n=======\nOFFSET = 7\n>>>>>>> REPLACE"
    result = sampler.apply_response(response, parent, patch_type="diff")
    assert result.success, result.error
    namespace = {}
    exec(str(result.program), namespace)
    assert namespace["solve"](2.9) == 9


def test_calm_retains_instance_profile():
    calm = CALM.__new__(CALM)
    calm._template_program_str = TEMPLATE
    outcome = EvaluationOutcome(2.0, instance_scores=(1.0, 3.0))
    calm._evaluator = SimpleNamespace(
        evaluate_program_record_time_with_details=lambda p: (outcome, 0.1)
    )
    calm._evaluation_executor = SimpleNamespace(
        submit=lambda fn, *a: SimpleNamespace(result=lambda: fn(*a))
    )
    score, perfs, status = calm._evaluate_code(TEMPLATE)
    assert status == "ok" and score == 2.0
    np.testing.assert_array_equal(perfs, [1.0, 3.0])


def test_pathwise_critic_sees_failed_rollouts_and_official_messages():
    function = parse_function(TEMPLATE)
    node = PathWiseNode(function, "rationale", "description", 1, "parent")
    action = PathWiseAction(["parent"], "change logic")
    method = PathWise.__new__(PathWise)
    method._use_policy_critic = True
    method._policy_reflection = ""
    method._policy_reflection_history = []
    method._tot_sample_nums = 0
    method._prompt = PathWisePrompt("task", function)
    method._debug_print = lambda *a: None
    method._policy_critic_llm = Replies("reflection")
    method._policy_critic_reflection(
        [node],
        [action, action],
        [
            [
                _Rollout(node, 2, "valid", action, 0),
                _Rollout(None, float("inf"), "FAILED_ROLLOUT", action, 0),
            ],
            [_Rollout(node, 3, "valid", action, 0), _Rollout(node, 4, "valid", action, 0)],
        ],
    )
    messages = method._policy_critic_llm.calls[0][0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "FAILED_ROLLOUT" in messages[-1]["content"]
    assert "rollout_1 (objective value: inf): FAILED_ROLLOUT" in messages[-1]["content"]


def test_funsearch_clusters_by_profile_not_only_mean():
    template = parse_program(TEMPLATE)
    island = Island(template, "solve", 2, 0.1, 30000)
    for profile in [(1.0, 3.0), (2.0, 2.0)]:
        function = parse_function(TEMPLATE)
        function.instance_scores = profile
        island.register_program(function, 2.0)
    assert len(island.clusters) == 2


def test_shinka_official_full_and_meta_prompts_keep_roles_and_objective_direction():
    from baselines.shinka_evo.prompts.prompts_base import BASE_SYSTEM_MSG
    from baselines.shinka_evo.prompts.prompts_meta import META_STEP3_SYSTEM_MSG
    from baselines.shinka_evo.shinka_evo import ShinkaEvo

    template = parse_program(TEMPLATE)
    parent = ShinkaProgram.create(template.functions[0], TEMPLATE, combined_score=7, correct=True)
    sampler = ShinkaSampler(None, template, task_description="task")
    messages = sampler.build_prompt(parent, [], [], "full")
    assert messages[0]["content"].startswith(sampler.BASE_SYSTEM_MSG)
    assert "<NAME>" in messages[0]["content"] and "<CODE>" in messages[0]["content"]
    assert "Combined score to maximize: -7.0" in messages[1]["content"]
    method = ShinkaEvo.__new__(ShinkaEvo)
    method._archive = SimpleNamespace(best_program=parent)
    method._meta_recommendations = "previous recommendation"
    messages = method._meta_prompt_recommendations("new insights")
    assert messages[0]["content"] == META_STEP3_SYSTEM_MSG
    assert "previous recommendation" in messages[1]["content"]


def test_inline_entrypoint_keeps_its_decorator_and_module_constant():
    source = "from functools import lru_cache\n@lru_cache(None)\ndef solve(\n    x,\n): return x + OFFSET\nOFFSET = 3\n"
    function = function_from_source(source, "solve")
    program = replace_function(function, TEMPLATE)
    namespace = {}
    exec(str(program), namespace)
    assert namespace["solve"](2) == 5
    assert namespace["solve"].cache_info().misses == 1


def test_pathwise_protects_the_unique_best_in_the_current_graph():
    method = PathWise.__new__(PathWise)
    method._all_nodes_archive = {}
    method._outer_iteration = method._inner_step = 0
    method._preserve_best_nodes_innersteps = True
    method._root_node_ids = {'a', 'b'}
    method._best_node = None
    base = parse_function(TEMPLATE)
    graph = PathWiseGraph()
    for identity, score in [('a', 1.), ('b', 2.)]:
        graph.add_node(PathWiseNode(base, 'why', 'desc', score, identity))
    first = PathWiseNode(base, 'why', 'improved', .5, 'rollout_0')
    method._add_selected_node_to_graph(graph, _Rollout(first, .5, 'improved', PathWiseAction(['b'], 'improve'), 0))
    protected = next(n.node_id for n in graph.get_state() if n.score == .5)
    method._inner_step = 1
    worse = PathWiseNode(base, 'why', 'worse', .7, 'rollout_1')
    method._add_selected_node_to_graph(graph, _Rollout(worse, .7, 'worse', PathWiseAction([protected], 'try'), 0))
    assert protected in graph.nodes
    assert protected in {n.node_id for n in method._policy_state(graph)}


def test_shinka_archive_keeps_the_random_tail_after_island_elites():
    from baselines.shinka_evo.population import ShinkaArchive

    import random
    archive = ShinkaArchive(rng=random.Random(0), num_archive_inspirations=4, elite_selection_ratio=.3)
    programs = []
    for i in range(8):
        function = parse_function(TEMPLATE)
        program = ShinkaProgram.create(function, str(function), correct=True,
            combined_score=float(i), island_idx=0, program_id=str(i), generation=1)
        archive.add_program(program)
        programs.append(program)
    samples = [tuple(p.id for p in archive._sample_archive_inspirations(programs[-1], 4)) for _ in range(10)]
    assert all(sample[:3] == ('0', '1', '2') for sample in samples)
    assert len(set(samples)) > 1


def test_mcts_retries_a_duplicate_before_evaluation_and_rounds_its_score():
    from baselines.mcts_ahd import MCTS_AHD
    from tests.support import TinyEvaluation

    first = 'def score(x):\n    result = 1.000004\n    return result\n'
    second = first.replace('1.000004', '2.000004')
    duplicate = function_from_source(first.rstrip(), 'score')
    duplicate.score = -1.
    llm = Replies('{Return one constant}\n```python\n' + first + '```', 'idea',
                  '{Return a new constant}\n```python\n' + second + '```', 'idea')
    method = MCTS_AHD(llm, TinyEvaluation(), max_sample_nums=1, num_samplers=1)
    try:
        function = method._sample_evaluate_register('task', func_only=True, operator='i1', population=[duplicate])
        assert function.score == -2.0
        assert len(llm.calls) == 4 and method._tot_sample_nums == 1
    finally:
        method._evaluation_executor.shutdown()


def test_shinka_embedding_failure_and_empty_novelty_follow_upstream():
    from baselines.shinka_evo.shinka_evo import ShinkaEvo

    method = ShinkaEvo.__new__(ShinkaEvo)
    lengths = []
    method._embedding_fn = lambda code: lengths.append(len(code)) or [1.0]
    method._register_event = lambda *args: None
    assert method._compute_embedding('x' * 12000) == [1.0] and lengths == [10000]
    def fail(code):
        raise ConnectionError('offline')
    method._embedding_fn = fail
    assert method._compute_embedding('code') == []
    assert method._is_novelty_response_accept('')


def test_pathwise_samples_the_batch_before_evaluating_rollouts():
    from tests.support import TinyEvaluation

    class BatchReplies:
        def __init__(self):
            self.observed = []
        def draw_sample(self, prompt):
            self.observed.append(method._tot_sample_nums)
            return '```python\ndef score_v2(x):\n    return x + 1\n```\nDescription: Increment'
    llm = BatchReplies()
    method = PathWise(llm, TinyEvaluation(), num_rollouts=2, max_sample_nums=4)
    function = parse_function('def score(x):\n    return x\n')
    state = [PathWiseNode(function, 'why', 'baseline', 0., 'a')]
    graph = PathWiseGraph()
    graph.add_node(state[0])
    try:
        rollouts = method._run_world_model_rollouts([PathWiseAction(['a'], 'change')] * 2, state, graph)
        assert llm.observed == [0, 0, 0, 0]
        assert all(r.node is not None for group in rollouts for r in group)
        assert method._tot_sample_nums == 4
    finally:
        method._evaluation_executor.shutdown()


def test_accepted_source_is_unchanged_until_a_method_edits_the_function():
    source = '# Keep layout\ndef solve( x ):\n    """ original description """\n    return x+1\nTAIL=3\n'
    function = function_from_source(source, 'solve')
    function.score = 2.0
    assert str(replace_function(function, TEMPLATE)) == source
    function.name = 'solve_v1'
    assert 'def solve_v1(' in str(function) and 'TAIL=3' in str(function)
    function.body = '    return x + 2'
    namespace = {}
    exec(str(function), namespace)
    assert namespace['solve_v1'](2) == 4
