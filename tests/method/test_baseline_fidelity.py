"""Regression checks for official prompts, complete programs and feedback routing."""
import ast
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import TextFunctionProgramConverter as C
from core.evaluate import EvaluationOutcome
from core.llm import OpenAIAPI
from baselines.shared.sampling import SampleTrimmer
from baselines.calm.calm import CALM
from baselines.calm.parse import extract_function_from_string
from baselines.calm.sampler import CALMSampler
from baselines.calm.seeds import load_seed
from baselines.calm.task_config import resolve_task_key
from baselines.eoh.sampler import EoHSampler
from baselines.mcts_ahd.sampler import MASampler
from baselines.pathwise.graph import PathWiseAction, PathWiseNode
from baselines.pathwise.pathwise import PathWise, _Rollout
from baselines.pathwise.prompt import PathWisePrompt
from baselines.pathwise.sampler import PathWiseSampler
from baselines.reevo.prompt import ReEvoPrompt
from baselines.funsearch.funsearch import FunSearch
from baselines.funsearch.programs_database import Island
from baselines.shinka_evo.sampler import ShinkaSampler
from baselines.shinka_evo.population import ShinkaProgram

TEMPLATE = 'def solve(x):\n    return x\n'
SOURCE = '''import math

def helper(x):
    return math.floor(x) + OFFSET

def solve_v2(x):
    return helper(x)

OFFSET = 3
'''
METHODS = ['eoh', 'reevo', 'mcts_ahd', 'pathwise', 'calm', 'funsearch', 'shinka_evo']


class Replies:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def draw_sample(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return next(self.responses)


@pytest.mark.parametrize('method', METHODS)
def test_complete_program_survives_sampling_reconstruction_and_parent_context(method):
    template = C.text_to_program(TEMPLATE)
    response = '{A multiline\nidea}\n```python\n' + SOURCE + '\n```'
    llm = Replies(response, 'Aligned idea')
    if method == 'eoh':
        _, function = EoHSampler(llm, template).get_thought_and_function('task')
    elif method == 'mcts_ahd':
        _, function = MASampler(llm, template).get_thought_and_function('task', 'generate')
        assert llm.calls[1][1] == {'temperature': 0}
        assert 'def helper' in llm.calls[1][0]
    elif method == 'reevo':
        function = SampleTrimmer.sample_to_function(SampleTrimmer(llm).draw_sample('task'), template)
    elif method == 'pathwise':
        function = PathWiseSampler.response_to_function(response, template)
    elif method == 'calm':
        calm = CALM.__new__(CALM)
        calm._template_program_str = TEMPLATE
        function, _ = calm._code_to_program(response)
    elif method == 'funsearch':
        funsearch = FunSearch.__new__(FunSearch)
        funsearch._function_name = 'solve'
        funsearch._template_program = template
        function = funsearch.sample_to_program(response, 2).functions[0]
    else:
        sampler = ShinkaSampler(llm, template, task_description='task')
        patch = sampler._apply_full_response(response, SimpleNamespace(program=TEMPLATE), patch_type='full')
        assert patch.success, patch.error
        function = patch.function
    assert function is not None
    program = C.function_to_program(copy.deepcopy(function), template)
    namespace = {}
    exec(str(program), namespace)
    assert namespace['solve'](2.9) == 5
    assert 'def helper' in str(function) and 'OFFSET = 3' in str(function)
    assert extract_function_from_string(str(function), 'solve')(2.9) == 5
    # Prompt formatting may rename an entrypoint, but must keep its dependencies.
    function.name = 'solve_v1'
    assert 'def helper' in str(function) and 'def solve_v1' in str(function)


def test_complete_solver_diff_can_edit_helpers_and_preserves_module_state():
    template = C.text_to_program(TEMPLATE)
    function = SampleTrimmer.sample_to_function(SOURCE, template)
    parent = ShinkaProgram.create(function, str(function), correct=True)
    sampler = ShinkaSampler(None, template, task_description='task')
    response = '<<<<<<< SEARCH\nOFFSET = 3\n=======\nOFFSET = 7\n>>>>>>> REPLACE'
    result = sampler.apply_response(response, parent, patch_type='diff')
    assert result.success, result.error
    namespace = {}
    exec(str(result.program), namespace)
    assert namespace['solve'](2.9) == 9


def test_module_decorator_and_future_import_are_retained():
    code = '''from __future__ import annotations
from functools import lru_cache

@lru_cache(None)
def solve(x):
    return x + 1
'''
    function = SampleTrimmer.sample_to_function(code, 'import math\n' + TEMPLATE)
    namespace = {}
    exec(str(function), namespace)
    assert namespace['solve'](2) == 3
    assert namespace['solve'].cache_info().misses == 1


def test_calm_transmits_roles_without_flattening():
    llm = Replies('response')
    messages = [{'role': 'system', 'content': 'instructions'}, {'role': 'user', 'content': 'task'}]
    CALMSampler(llm).sample(messages)
    assert OpenAIAPI._build_messages(llm.calls[0][0], None) == messages


def test_reevo_refuses_fictitious_better_worse_context():
    a, b = C.text_to_function(TEMPLATE), C.text_to_function(TEMPLATE)
    a.score = b.score = 1
    with pytest.raises(ValueError, match='different scores'):
        ReEvoPrompt.get_short_term_reflection_prompt('task', [a, b])


def test_calm_retains_instance_profile():
    calm = CALM.__new__(CALM)
    calm._template_program_str = TEMPLATE
    outcome = EvaluationOutcome(2.0, instance_scores=(1.0, 3.0))
    calm._evaluator = SimpleNamespace(evaluate_program_record_time_with_details=lambda p: (outcome, 0.1))
    calm._evaluation_executor = SimpleNamespace(submit=lambda fn, *a: SimpleNamespace(result=lambda: fn(*a)))
    score, perfs, status = calm._evaluate_code(TEMPLATE)
    assert status == 'ok' and score == 2.0
    np.testing.assert_array_equal(perfs, [1.0, 3.0])


@pytest.mark.parametrize('task', ['fssp_gls', 'graph_colouring', 'jssp_construct'])
def test_calm_new_tasks_resolve_and_load_executable_seed(task):
    from benchmarks.tasks import CLASSES
    evaluation = object.__new__(CLASSES[task])
    assert resolve_task_key(evaluation) == task
    code, idea = load_seed(task)
    assert idea.startswith('The idea of the algorithm is to')
    ast.parse(code)
    assert extract_function_from_string(code) is not None


def test_pathwise_critic_sees_failed_rollouts_and_official_messages():
    function = C.text_to_function(TEMPLATE)
    node = PathWiseNode(function, 'rationale', 'description', 1, 'parent')
    action = PathWiseAction(['parent'], 'change logic')
    method = PathWise.__new__(PathWise)
    method._use_policy_critic = True
    method._policy_reflection = ''
    method._policy_reflection_history = []
    method._tot_sample_nums = 0
    method._prompt = PathWisePrompt('task', function)
    method._debug_print = lambda *a: None
    method._policy_critic_llm = Replies('reflection')
    method._policy_critic_reflection([node], [action, action], [
        [_Rollout(node, 2, 'valid', action, 0), _Rollout(None, float('inf'), 'FAILED_ROLLOUT', action, 0)],
        [_Rollout(node, 3, 'valid', action, 0), _Rollout(node, 4, 'valid', action, 0)],
    ])
    messages = method._policy_critic_llm.calls[0][0]
    assert [m['role'] for m in messages] == ['system', 'user']
    assert 'FAILED_ROLLOUT' in messages[-1]['content']
    assert 'rollout_1 (objective value: inf): FAILED_ROLLOUT' in messages[-1]['content']
    assert 'less than 60 words' in messages[-1]['content']
    assert len(method._prompt.POLICY_PHRASES) == 16
    assert len(method._prompt.WORLD_MODEL_PHRASES) == 10


def test_funsearch_clusters_by_profile_not_only_mean():
    template = C.text_to_program(TEMPLATE)
    island = Island(template, 'solve', 2, .1, 30000)
    for profile in [(1., 3.), (2., 2.)]:
        function = C.text_to_function(TEMPLATE)
        function.instance_scores = profile
        island.register_program(function, 2.)
    assert len(island.clusters) == 2


def test_shinka_official_full_and_meta_prompts_keep_roles_and_objective_direction():
    from baselines.shinka_evo.shinka_evo import ShinkaEvo
    from baselines.shinka_evo.prompts.prompts_base import BASE_SYSTEM_MSG
    from baselines.shinka_evo.prompts.prompts_meta import META_STEP3_SYSTEM_MSG
    template = C.text_to_program(TEMPLATE)
    parent = ShinkaProgram.create(template.functions[0], TEMPLATE, combined_score=7, correct=True)
    sampler = ShinkaSampler(None, template, task_description='task')
    messages = sampler.build_prompt(parent, [], [], 'full')
    assert messages[0]['content'].startswith(BASE_SYSTEM_MSG)
    assert '<NAME>' in messages[0]['content'] and '<CODE>' in messages[0]['content']
    assert 'Combined score to maximize: -7.0' in messages[1]['content']
    method = ShinkaEvo.__new__(ShinkaEvo)
    method._archive = SimpleNamespace(best_program=parent)
    method._meta_recommendations = 'previous recommendation'
    messages = method._meta_prompt_recommendations('new insights')
    assert messages[0]['content'] == META_STEP3_SYSTEM_MSG
    assert 'previous recommendation' in messages[1]['content']


def test_embedding_adapter_validates_vectors_without_exposing_key(monkeypatch):
    from experiments.shinka_evo.embedding import CodeEmbedding
    service = CodeEmbedding()
    service._client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        data=[SimpleNamespace(embedding=[1., 0.])])) )
    assert service('code') == [1., 0.]
    service._client.embeddings.create = lambda **kw: SimpleNamespace(data=[SimpleNamespace(embedding=[float('nan')])])
    with pytest.raises(ValueError, match='invalid code vector'):
        service('code')
    assert 'api_key' not in service.settings


def test_unfenced_module_keeps_leading_constants():
    program = SampleTrimmer.sample_to_program('OFFSET = 3\ndef solve(x):\n    return x + OFFSET\n', TEMPLATE)
    namespace = {}
    exec(str(program), namespace)
    assert namespace['solve'](2) == 5


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('task', ['tsp_construct', 'cvrp_aco', 'op_aco', 'fssp_gls', 'graph_colouring', 'jssp_construct'])
def test_all_six_task_run_configs_are_constructible(method, task, tmp_path):
    run = importlib.import_module(f'experiments.{method}.run')
    spec = run.make_run_spec(task=task, experiments_root=tmp_path)
    run.write_run_config(spec, tmp_path, 'fidelity')
    assert (tmp_path / 'run_config.json').is_file()


def test_deepseek_system_user_count_uses_published_role_tokens():
    client = OpenAIAPI.__new__(OpenAIAPI)
    client.chars_per_token = None
    client._tokenizer = object()
    client.enable_thinking = False
    client.prompt_format = '<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜></think>'
    captured = []
    client.count_tokens = lambda text: captured.append(text) or 12
    messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'task'}]
    assert client.count_prompt_tokens(messages) == 12
    assert captured == ['<｜begin▁of▁sentence｜><｜System｜>rules<｜User｜>task<｜Assistant｜></think>']
    assert client._build_messages(messages, None) == messages


@pytest.mark.parametrize('task', ['fssp_gls', 'graph_colouring', 'jssp_construct'])
def test_calm_new_task_seed_spends_exactly_one_candidate(task, tmp_path):
    from experiments.calm import run
    method = run.build_method(run.make_run_spec(task=task, experiments_root=tmp_path), tmp_path)
    method._max_sample_nums = 1
    method._evaluate_code = lambda code: (2., np.asarray([1., 3.]), 'ok')
    try:
        method._initialize_seeds()
        assert method._tot_sample_nums == 1
        assert method._seed_algos[0].perfs.tolist() == [1., 3.]
        assert method._seed_algos[0].function is not None
    finally:
        method._evaluation_executor.shutdown()
        method._llm.close()
