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
from baselines.shared.sampling import SampleTrimmer
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


@pytest.mark.parametrize("method", METHODS)
def test_complete_program_survives_sampling_reconstruction_and_parent_context(method):
    template = parse_program(TEMPLATE)
    response = "{A multiline\nidea}\n```python\n" + SOURCE + "\n```"
    llm = Replies(response, "Aligned idea")
    if method == "eoh":
        _, function = EoHSampler(llm, template).get_thought_and_function("task")
    elif method == "mcts_ahd":
        _, function = MASampler(llm, template).get_thought_and_function("task", "generate")
        assert llm.calls[1][1] == {"temperature": 0}
        assert "def helper" in llm.calls[1][0]
    elif method == "reevo":
        function = SampleTrimmer.sample_to_function(
            SampleTrimmer(llm).draw_sample("task"), template
        )
    elif method == "pathwise":
        function = PathWiseSampler.response_to_function(response, template)
    elif method == "calm":
        calm = CALM.__new__(CALM)
        calm._template_program_str = TEMPLATE
        function, _ = calm._code_to_program(response)
    elif method == "funsearch":
        funsearch = FunSearch.__new__(FunSearch)
        funsearch._function_name = "solve"
        funsearch._template_program = template
        function = funsearch.sample_to_program(response, 2).functions[0]
    else:
        sampler = ShinkaSampler(llm, template, task_description="task")
        patch = sampler._apply_full_response(
            response, SimpleNamespace(program=TEMPLATE), patch_type="full"
        )
        assert patch.success, patch.error
        function = patch.function
    assert function is not None
    program = replace_function(copy.deepcopy(function), template)
    namespace = {}
    exec(str(program), namespace)
    assert namespace["solve"](2.9) == 5
    assert "def helper" in str(function) and "OFFSET = 3" in str(function)
    assert extract_function_from_string(str(function), "solve")(2.9) == 5
    # Prompt formatting may rename an entrypoint, but must keep its dependencies.
    function.name = "solve_v1"
    assert "def helper" in str(function) and "def solve_v1" in str(function)


def test_complete_solver_diff_can_edit_helpers_and_preserves_module_state():
    template = parse_program(TEMPLATE)
    function = SampleTrimmer.sample_to_function(SOURCE, template)
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
    assert messages[0]["content"].startswith(BASE_SYSTEM_MSG)
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
    function = SampleTrimmer.sample_to_function(source, TEMPLATE)
    program = replace_function(function, TEMPLATE)
    namespace = {}
    exec(str(program), namespace)
    assert namespace["solve"](2) == 5
    assert namespace["solve"].cache_info().misses == 1
