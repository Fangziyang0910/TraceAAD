"""Current offline fidelity probes. evidence.json preserves the pre-repair snapshot."""

import argparse
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from core import TextFunctionProgramConverter as Converter
from core.llm import OpenAIAPI
from baselines.calm.calm import CALM
from baselines.calm.parse import dedent
from baselines.calm.prompt import PromptBuilder
from baselines.calm.sampler import CALMSampler
from baselines.calm.seeds import load_seed
from baselines.calm.task_config import resolve_task_key
from baselines.eoh.eoh import EoH
from baselines.funsearch.funsearch import FunSearch
from baselines.mcts_ahd.sampler import MASampler
from baselines.pathwise.graph import PathWiseAction, PathWiseNode
from baselines.pathwise.pathwise import PathWise, _Rollout
from baselines.pathwise.prompt import PathWisePrompt
from baselines.pathwise.sampler import PathWiseSampler
from baselines.reevo.prompt import ReEvoPrompt
from baselines.shared.sampling import SampleTrimmer
from baselines.shinka_evo.sampler import ShinkaSampler
from benchmarks.tasks import CLASSES, CO_TASKS


def local_probes():
    template = 'def solve(x):\n    return x\n'
    program = Converter.text_to_program(template)
    source = 'def solve(x):\n    return helper(x)\n\ndef helper(x):\n    return x + 1\n'
    namespace = {}
    exec(source, namespace)
    assert namespace['solve'](2) == 3

    calm = CALM.__new__(CALM)
    calm._template_program_str = template
    funsearch = FunSearch.__new__(FunSearch)
    funsearch._function_name = 'solve'
    funsearch._template_program = program
    shinka = ShinkaSampler(None, program, task_description='test')
    parent = SimpleNamespace(program=template)
    parsed = {}
    for method in ('eoh', 'reevo', 'mcts_ahd'):
        parsed[method] = SampleTrimmer.sample_to_program(
            source, program)
    parsed['calm'] = calm._code_to_program(source)[1]
    function = PathWiseSampler.response_to_function(source, program)
    parsed['pathwise'] = Converter.function_to_program(function, program)
    parsed['funsearch'] = funsearch.sample_to_program(source, 2)
    parsed['shinka_evo'] = shinka._apply_full_response(
        f'```python\n{source}```', parent, patch_type='full').program
    helper_results = {}
    for method, candidate in parsed.items():
        namespace = {}
        try:
            exec(str(candidate), namespace)
            outcome = {'value': namespace['solve'](2)}
        except Exception as error:
            outcome = {'error': f'{type(error).__name__}: {error}'}
        helper_results[method] = {
            'functions': [f.name for f in candidate.functions] if candidate else None,
            **outcome,
        }

    task_results = {}
    for task in CO_TASKS:
        # Resolution only inspects the class. Do not generate task datasets.
        evaluation = object.__new__(CLASSES[task])
        row = {}
        for label, operation in (
            ('task_resolution', lambda: resolve_task_key(evaluation)),
            ('seed', lambda: bool(load_seed(task)[0])),
        ):
            try:
                row[label] = operation()
            except Exception as error:
                row[label] = f'{type(error).__name__}: {error}'
        task_results[task] = row

    first = Converter.text_to_function(template)
    second = copy.deepcopy(first)
    first.score = second.score = 1
    second.body = '    return x + 1\n'
    try:
        ReEvoPrompt.get_short_term_reflection_prompt('test', [first, second])
        tie_rejected = False
    except ValueError:
        tie_rejected = True
    second.score = 2
    reflection = ReEvoPrompt.get_short_term_reflection_prompt('test', [first, second])
    node = PathWiseNode(first, 'rationale', 'description', 1, 'node')
    role_prompts = {
        'reevo': reflection,
        'pathwise': PathWisePrompt('test', first).policy_prompt([node], ''),
        'shinka_evo': shinka.build_prompt(SimpleNamespace(
            function=first, combined_score=1, public_metrics={}, text_feedback='', correct=True), [], [], 'full'),
        'calm': CALMSampler.messages_to_prompt([
            {'role': 'system', 'content': 'system instructions'},
            {'role': 'user', 'content': 'task instructions'},
        ]),
    }

    eoh = EoH.__new__(EoH)
    eoh._pop_size = 2
    eoh._continue_loop = lambda: True
    eoh._task_description_str = 'test'
    eoh._function_to_evolve = first
    eoh._profiler = None
    eoh._debug_mode = False
    calls = []

    def sample(*args, **kwargs):
        calls.append(kwargs)
        function = copy.deepcopy(first)
        function.score = len(calls)
        function.body = f'    return {len(calls)}\n'
        return function

    eoh._sample_evaluate_register = sample
    with contextlib.redirect_stdout(io.StringIO()):
        eoh._initialize_population()
    critic = PathWise.__new__(PathWise)
    critic._use_policy_critic = True
    critic._policy_reflection = ''
    critic._policy_reflection_history = []
    critic._tot_sample_nums = 0
    critic._prompt = PathWisePrompt('test', first)
    critic._debug_print = lambda *args: None
    captured = []
    critic._policy_critic_llm = SimpleNamespace(draw_sample=lambda prompt: captured.append(prompt) or 'reflection')
    action = PathWiseAction(['node'], 'test action')
    critic._policy_critic_reflection([node], [action, action], [
        [_Rollout(node, 2, 'valid A', action, 0), _Rollout(None, float('inf'), 'FAILED_A', action, 0)],
        [_Rollout(node, 3, 'valid B1', action, 0), _Rollout(node, 4, 'valid B2', action, 0)],
    ])
    assert len(captured) == 1
    return {
        'complete_program_control': 3,
        'helper_parsing': helper_results,
        'calm_task_support': task_results,
        'reevo_tie': {'scores': [1, 1], 'tie_rejected': tie_rejected},
        'transport_roles': {method: [m['role'] for m in OpenAIAPI._build_messages(prompt, None)]
                            for method, prompt in role_prompts.items()},
        'eoh_initialization': {'population_size': 2, 'generated': len(calls),
                               'counted': sum(c['count_sample'] for c in calls)},
        'mcts_multiline_thought': MASampler.trim_thought_from_response('{First line\nSecond line}\n' + template),
        'pathwise_failed_rollout_context': {
            'inputs': [[2, 'inf (failed)'], [3, 4]],
            'failure_description_present': 'FAILED_A' in str(captured[0]),
            'action_A_displayed_average_is_2': 'Average score: 2.000000' in str(captured[0]),
        },
    }


def upstream_probes(root):
    """Execute only extracted prompt methods, without importing upstream trainers."""
    path = root / 'calm/calm_trainer.py'
    tree = ast.parse(path.read_text())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    names = ['system_prompt', 'prompt_creation', 'prompt_design_strategy',
             'prompt_local_refinement_guidance', 'prompt_simplification', 'prompt_injection',
             'prompt_replacement', 'prompt_algo_details', 'prompt_score_profile',
             'prompt_algo_requirements', 'prompt_crossover']
    reference_class = ast.ClassDef(name='ReferencePrompt', bases=[], keywords=[], decorator_list=[],
                                  body=[n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in names])
    module = ast.fix_missing_locations(ast.Module(body=[reference_class], type_ignores=[]))
    namespace = {'np': np, 'dedent': dedent}
    exec(compile(module, str(path), 'exec'), namespace)
    reference = namespace['ReferencePrompt']()
    reference.problem = SimpleNamespace(name='test', description='task', unit='units')
    reference.algorithm_template = 'def heuristic(x):\n    pass'
    reference.injected_components = ['component1']
    reference.train_epoch = 1
    ours = PromptBuilder(problem_name='test', problem_description='task', problem_unit='units',
                         algorithm_template=reference.algorithm_template, rs=np.random.RandomState(7),
                         injected_components=reference.injected_components, train_epoch_getter=lambda: 1)
    matches = {}
    for name in names:
        if name in ('prompt_algo_details', 'prompt_score_profile'):
            continue  # These legitimately reverse objective direction; audit their data separately.
        reference.rs = np.random.RandomState(7)
        ours.rs = np.random.RandomState(7)
        left, right = getattr(reference, name), getattr(ours, name)
        if callable(left):
            args = ([],) if name in ('prompt_simplification', 'prompt_injection', 'prompt_replacement', 'prompt_crossover') else ()
            left, right = left(*args), right(*args)
        matches[name] = left == right
    revisions = {}
    for method in ('eoh', 'reevo', 'mcts_ahd', 'funsearch', 'pathwise', 'calm', 'shinka_evo'):
        directory = root / method
        revisions[method] = {
            'repository': subprocess.check_output(['git', '-C', str(directory), 'remote', 'get-url', 'origin'], text=True).strip(),
            'commit': subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip(),
        }
    return {'official_sources': revisions, 'calm_prompt_equal_empty_parents': matches}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path)
    args = parser.parse_args()
    result = {'audited_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              **local_probes()}
    if args.upstream_root:
        result.update(upstream_probes(args.upstream_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
