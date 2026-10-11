"""New baselines restore search state without evaluating or sampling old programs."""
import gzip
import json
import random

import numpy as np
import pytest

from core import Evaluation, LLM
from baselines.calm import CALM, CALMProfiler
from baselines.calm.prompt import Prompt
from baselines.pathwise import PathWise, PathWiseProfiler
from baselines.shinka_evo import ShinkaEvo, ShinkaEvoProfiler
from baselines.shared.checkpoint import NAME, encode, decode, read_checkpoint, save_checkpoint, restore_checkpoint
from experiments.infra.resume_baseline import history, recover
from traceaad.common.storage import append_jsonl, read_json, write_json


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program='def solve(x):\n    return 1\n',
                         task_description='Minimize the returned value.', safe_evaluate=False)
        self.evaluated = []

    def evaluate_program(self, code, function, **kwargs):
        self.evaluated.append(code)
        return float(function(1))


class ScriptedLLM(LLM):
    def __init__(self, values, *, stop=True):
        super().__init__()
        self.values = list(values)
        self.calls = []
        self.stop = stop

    def draw_sample(self, prompt, **kwargs):
        self.calls.append(prompt)
        text = str(prompt)
        if 'You are controlling the evolutionary search' in text:
            return 'PARENTS: [init_0]\nDIRECTIVE: reduce the returned value'
        if 'Analyze heuristic evolution for' in text:
            return 'Prefer smaller values.'
        if 'Phrase your hints as design principles' in text:
            return 'Keep the return deterministic.'
        if not self.values:
            if self.stop:
                raise KeyboardInterrupt
            pytest.fail('Search exceeded its remaining budget')
        value = self.values.pop(0)
        return ('{{The idea of the algorithm is to return a smaller deterministic value.}}\n'
                'Description: return a smaller value\nRationale: minimize the objective\n'
                f'```python\ndef solve(x):\n    return {value}\n```')


def make_method(name, directory, llm, *, budget=6):
    evaluation = TinyEvaluation()
    if name == 'calm':
        method = CALM(llm, evaluation, CALMProfiler(directory), max_sample_nums=budget,
                      task_key='tsp_construct', numeric_refine_variants=0,
                      numeric_refine_initial_variants=0, population_size=2,
                      n_prompts=1, n_generations=1, ub_crossover=0)
    elif name == 'pathwise':
        method = PathWise(llm, evaluation, PathWiseProfiler(directory), max_sample_nums=budget,
                          pop_size=2, init_pop_size=2, num_actions=1, num_rollouts=1,
                          max_inner_steps=1, use_prompt_perturbation=False)
    else:
        method = ShinkaEvo(llm, evaluation, ShinkaEvoProfiler(directory), max_sample_nums=budget,
                           num_generations=10, num_islands=2, patch_types=('full',), patch_type_probs=(1.,),
                           random_seed=0, embedding_fn=lambda code: [float(f'return {-i}\n' in code) for i in range(1, 10)]
                           + [float('return 1\n' in code)])
    return method, evaluation


def setup_config(directory, name, budget=6):
    write_json(directory / 'run_config.json', {'method': name, 'task': 'toy', 'budget': budget,
                                              'seed': 0, 'backend': 'test', 'run_name': 'test'})


def run_until_interrupt(method):
    try:
        method.run()
    except KeyboardInterrupt:
        pass


@pytest.fixture(autouse=True)
def toy_calm_seed(monkeypatch):
    monkeypatch.setattr('baselines.calm.calm.load_seed', lambda task:
                        ('def solve(x):\n    return 1\n', 'The idea of the algorithm is to return one.'))


@pytest.mark.parametrize('name', ['calm', 'pathwise', 'shinka_evo'])
def test_new_run_can_resume_twice_without_reinitialization_or_budget_reset(tmp_path, name):
    setup_config(tmp_path, name)
    first, _ = make_method(name, tmp_path, ScriptedLLM([-1]))
    run_until_interrupt(first)
    assert (tmp_path / NAME).exists()
    spent = history(tmp_path)[-1]
    assert spent == (1 if name == 'pathwise' else 2)
    assert read_json(tmp_path / 'summary.json')['status'] != 'finished'

    second, evaluation = make_method(name, tmp_path, ScriptedLLM([-2]))
    report = recover(second, tmp_path)
    assert report['recovery'] == 'search_checkpoint'
    assert second._tot_sample_nums == spent
    assert evaluation.evaluated == []
    run_until_interrupt(second)
    spent_again = history(tmp_path)[-1]
    assert spent_again > spent

    final, evaluation = make_method(name, tmp_path, ScriptedLLM([-3, -4, -5, -6, -7], stop=False))
    recover(final, tmp_path)
    assert evaluation.evaluated == []
    final.run()
    assert final._tot_sample_nums == 6
    assert len(evaluation.evaluated) == 6 - spent_again
    assert all('return 1\n' not in code for code in evaluation.evaluated)
    assert read_json(tmp_path / 'summary.json')['status'] == 'finished'


def test_calm_snapshot_preserves_prompt_aliases_profiles_and_numeric_history(tmp_path):
    setup_config(tmp_path, 'calm')
    method, _ = make_method('calm', tmp_path, ScriptedLLM([]))
    method._initialize_seeds()
    prompt = Prompt('Improve this code')
    prompt.record_trial({'performance': 0.5, 'n_epoch': 3, 'is_revisit': True})
    method._used_prompts = [prompt]
    method._last_revisited_prompts = [prompt]
    method._numeric_refine_attempted.add(('hash', 2))
    method._injected_components.append('a component')
    method._age_stuck = 4
    save_checkpoint(method, 'calm', 'round_complete')
    expected_local = method._rs.random()
    expected_global = np.random.random()
    expected_python = random.random()
    restored, evaluation = make_method('calm', tmp_path, ScriptedLLM([]))
    recover(restored, tmp_path)
    assert evaluation.evaluated == []
    assert restored._seed_algos[0] is restored._algos[0]
    assert restored._used_prompts[0] is restored._last_revisited_prompts[0]
    assert restored._used_prompts[0].trials[3][0]['performance'] == .5
    assert restored._numeric_refine_attempted == {('hash', 2)}
    assert restored._prompts.injected_components is restored._injected_components
    assert restored._age_stuck == 4
    assert np.array_equal(restored._algos[0].perfs, method._algos[0].perfs)
    assert restored._rs.random() == expected_local
    assert np.random.random() == expected_global
    assert random.random() == expected_python
    method._evaluation_executor.shutdown()
    restored._evaluation_executor.shutdown()


def test_shinka_keeps_embeddings_island_copies_bandit_and_meta(tmp_path):
    setup_config(tmp_path, 'shinka_evo')
    method, _ = make_method('shinka_evo', tmp_path, ScriptedLLM([-1]))
    method._evaluate_seed()
    method._generation = 1
    program = method._run_generation(1)
    method._meta_scratch_pad = 'remember this'
    method._meta_recommendations = '1. Use a smaller value'
    save_checkpoint(method, 'shinka_evo', 'generation')
    expected_local = method._rng.random()
    expected_bandit = method._bandit.rng.random()
    restored, evaluation = make_method('shinka_evo', tmp_path, ScriptedLLM([]))
    recover(restored, tmp_path)
    assert evaluation.evaluated == []
    assert restored._generation == 1
    assert restored._archive.islands == method._archive.islands
    assert restored._archive.archive_ids == method._archive.archive_ids
    assert restored._archive.programs[program.id].embedding == program.embedding
    assert restored._evaluated_since_meta[0] is restored._archive.programs[program.id]
    assert np.array_equal(restored._bandit.n_completed, method._bandit.n_completed)
    assert restored._bandit.history == method._bandit.history
    assert restored._meta_scratch_pad == 'remember this'
    assert restored._meta_recommendations == '1. Use a smaller value'
    assert restored._rng.random() == expected_local
    assert restored._bandit.rng.random() == expected_bandit


def test_pending_patch_spends_budget_but_meta_and_calm_generations_do_not(tmp_path):
    setup_config(tmp_path, 'shinka_evo')
    method, _ = make_method('shinka_evo', tmp_path, ScriptedLLM([]))
    method._evaluate_seed()
    save_checkpoint(method, 'shinka_evo', 'generation')
    append_jsonl(tmp_path / 'calls.jsonl', {'stage': 'patch', 'sample_order': 2})
    append_jsonl(tmp_path / 'calls.jsonl', {'stage': 'meta_summary', 'sample_order': 3})
    restored, _ = make_method('shinka_evo', tmp_path, ScriptedLLM([]))
    report = recover(restored, tmp_path)
    assert report['pending_budget'] == 1
    assert restored._tot_sample_nums == 2
    setup_config(tmp_path, 'calm')
    assert history(tmp_path)[-1] == 1
    append_jsonl(tmp_path / 'calls.jsonl', {'stage': 'generate', 'sample_order': 2})
    assert history(tmp_path)[-1] == 1
    append_jsonl(tmp_path / 'events.jsonl', {'kind': 'method', 'data': {
        'event': 'evaluation_batch_reserved', 'sample_order': 4, 'counts_budget': True}})
    assert history(tmp_path)[-1] == 4


@pytest.mark.parametrize('change', ['conditions', 'missing_journal', 'type'])
def test_corrupt_or_mismatched_checkpoint_is_rejected(tmp_path, change):
    setup_config(tmp_path, 'shinka_evo')
    method, _ = make_method('shinka_evo', tmp_path, ScriptedLLM([]))
    method._evaluate_seed()
    save_checkpoint(method, 'shinka_evo', 'generation')
    if change == 'conditions':
        config = read_json(tmp_path / 'run_config.json')
        write_json(tmp_path / 'run_config.json', {**config, 'budget': 7})
    elif change == 'missing_journal':
        (tmp_path / 'events.jsonl').unlink()
    else:
        payload = read_checkpoint(tmp_path)
        payload['state'] = {'type': 'os.system', 'value': 'must not execute'}
        with gzip.open(tmp_path / NAME, 'wt') as handle:
            json.dump(payload, handle)
    with pytest.raises(ValueError):
        restore_checkpoint(method, tmp_path, 'shinka_evo')


def test_codec_rejects_clients_and_preserves_nonfinite_values_and_integer_keys():
    value = {2: (np.array([1., 2.]), {('hash', 3)}, float('inf'))}
    result = decode(encode(value))
    assert np.array_equal(result[2][0], value[2][0])
    assert result[2][1:] == value[2][1:]
    with pytest.raises(TypeError):
        encode(ScriptedLLM([]))


@pytest.mark.parametrize('name', ['calm', 'pathwise', 'shinka_evo'])
def test_checkpoint_writes_do_not_change_uninterrupted_search(tmp_path, monkeypatch, name):
    import importlib
    module = importlib.import_module('baselines.' + name + '.' + ('shinka_evo' if name == 'shinka_evo' else name))
    outputs = []
    for enabled in (True, False):
        directory = tmp_path / str(enabled)
        directory.mkdir()
        setup_config(directory, name)
        random.seed(42)
        np.random.seed(42)
        llm = ScriptedLLM([-1, -2, -3, -4, -5, -6], stop=False)
        method, evaluation = make_method(name, directory, llm)
        with monkeypatch.context() as patch:
            if not enabled:
                patch.setattr(module, 'save_checkpoint', lambda *args: None)
            method.run()
        outputs.append((llm.calls, evaluation.evaluated, method._tot_sample_nums,
                        read_json(directory / 'summary.json')['best']['fitness']))
    assert outputs[0] == outputs[1]


def test_snapshot_array_retains_empty_dimensions_and_nan():
    value = np.empty((0, 2), dtype=np.uint32)
    result = decode(encode(value))
    assert result.shape == (0, 2) and result.dtype == value.dtype
    assert np.isnan(decode(encode(np.array([float('nan')])))[0])


@pytest.mark.parametrize('name', ['calm', 'pathwise', 'shinka_evo'])
def test_new_methods_reject_legacy_logs_without_complete_search_state(tmp_path, name):
    setup_config(tmp_path, name)
    method, _ = make_method(name, tmp_path, ScriptedLLM([]))
    append_jsonl(tmp_path / 'events.jsonl', {'kind': 'candidate', 'candidate_id': 1,
                                          'budget_used': 1, 'valid': False})
    with pytest.raises(ValueError, match='needs a search checkpoint'):
        recover(method, tmp_path)
    if hasattr(method, '_evaluation_executor'):
        method._evaluation_executor.shutdown()


@pytest.mark.parametrize('name', ['calm', 'pathwise', 'shinka_evo'])
def test_recovery_builder_preserves_original_run_parameters(tmp_path, monkeypatch, name):
    import importlib
    from experiments.infra.resume_baseline import build
    module = importlib.import_module('experiments.' + name + '.run')
    options = {'calm': {'num_evaluators': 2}, 'pathwise': {'num_actions': 3},
               'shinka_evo': {'archive_size': 5, 'num_generations': 8}}[name]
    spec = module.make_run_spec(task='tsp_construct', backend='deepseek',
                                max_sample_nums=6, seed=2, repeat=1, **options)
    module.write_run_config(spec, tmp_path, 'original')
    observed = []
    monkeypatch.setattr(module, 'build_method', lambda spec, log_dir: observed.append(spec) or 'method')
    assert build(read_json(tmp_path / 'run_config.json'), tmp_path) == 'method'
    rebuilt = observed[0]
    assert rebuilt.max_sample_nums == 6 and rebuilt.seed == 2
    assert all(getattr(rebuilt, key) == value for key, value in options.items())


def test_generation_limit_remains_a_normal_shinka_completion(tmp_path):
    setup_config(tmp_path, 'shinka_evo')
    method, _ = make_method('shinka_evo', tmp_path, ScriptedLLM([-1], stop=False))
    method._num_generations = 1
    method.run()
    assert method._tot_sample_nums == 2
    assert read_json(tmp_path / 'summary.json')['status'] == 'finished'


def test_calm_numeric_reservations_survive_interrupted_evaluation(tmp_path):
    from baselines.calm.numeric_refine import run_numeric_refinement
    setup_config(tmp_path, 'calm')
    method, _ = make_method('calm', tmp_path, ScriptedLLM([]))
    method._initialize_seeds()
    def interrupt(code):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_numeric_refinement(
            candidate_parents=method._seed_algos, algos=method._algos, hp=method._hp,
            rs=method._rs, evaluations_used=1, evaluation_budget=6, log_step=0,
            seen_code_signatures=method._seen_code_signatures,
            numeric_refine_attempted=method._numeric_refine_attempted,
            matches_expected_signature=method.matches_expected_signature,
            evaluate_code=interrupt, is_seen_performance_profile=method.is_seen_performance_profile,
            record_performance_profile=method.record_performance_profile, best_perf=method.best_perf,
            log_info=method._log_info, variants_per_parent=2, top_k=0,
            function_name=method._function_name, on_budget_reserved=method._reserve_budget)
    assert method._tot_sample_nums == history(tmp_path)[-1] == 3
    restored, _ = make_method('calm', tmp_path, ScriptedLLM([]))
    assert recover(restored, tmp_path)['pending_budget'] == 2
    assert restored._tot_sample_nums == 3
    method._evaluation_executor.shutdown()
    restored._evaluation_executor.shutdown()


def test_rejected_shinka_tail_is_in_the_final_budget(tmp_path):
    setup_config(tmp_path, 'shinka_evo', budget=3)
    llm = ScriptedLLM([])
    llm.draw_sample = lambda *args, **kwargs: 'invalid patch'
    method, _ = make_method('shinka_evo', tmp_path, llm, budget=3)
    method.run()
    assert method._tot_sample_nums == 3
    assert read_json(tmp_path / 'summary.json')['budget_used'] == 3
    assert history(tmp_path)[-1] == 3


def test_calm_resumes_seed_refinement_stage_without_reevaluating_seed(tmp_path):
    setup_config(tmp_path, 'calm')
    method, _ = make_method('calm', tmp_path, ScriptedLLM([]))
    method._initialize_seeds()
    save_checkpoint(method, 'calm', 'seed_evaluated')
    restored, evaluation = make_method('calm', tmp_path, ScriptedLLM([-1, -2, -3, -4, -5], stop=False))
    recover(restored, tmp_path)
    refined = []
    refine = restored._refine_seeds
    def observe_refinement():
        refined.append(True)
        refine()
    restored._refine_seeds = observe_refinement
    restored.run()
    assert refined == [True]
    assert len(evaluation.evaluated) == 5
    assert all('return 1\n' not in code for code in evaluation.evaluated)
    method._evaluation_executor.shutdown()


def test_failed_checkpoint_write_preserves_previous_snapshot(tmp_path, monkeypatch):
    import baselines.shared.checkpoint as checkpoint
    setup_config(tmp_path, 'shinka_evo')
    method, _ = make_method('shinka_evo', tmp_path, ScriptedLLM([]))
    method._evaluate_seed()
    save_checkpoint(method, 'shinka_evo', 'generation')
    previous = (tmp_path / NAME).read_bytes()
    def interrupt(payload, handle, **kwargs):
        handle.write('{"partial":')
        raise RuntimeError('checkpoint interrupted')
    monkeypatch.setattr(checkpoint.json, 'dump', interrupt)
    with pytest.raises(RuntimeError, match='checkpoint interrupted'):
        save_checkpoint(method, 'shinka_evo', 'generation')
    assert (tmp_path / NAME).read_bytes() == previous
    assert read_checkpoint(tmp_path)['budget_used'] == 1


def test_recovery_cleanup_closes_meta_embedding_and_executor():
    from types import SimpleNamespace
    from experiments.infra.resume_baseline import close_recovery_method
    closed = []
    primary = SimpleNamespace(close=lambda: closed.append('primary'))
    meta = SimpleNamespace(close=lambda: closed.append('meta'))
    embedding = SimpleNamespace(close=lambda: closed.append('embedding'))
    executor = SimpleNamespace(shutdown=lambda **kwargs: closed.append('executor'))
    method = SimpleNamespace(_all_llms=lambda: [primary, meta, primary],
                             _embedding_fn=embedding, _evaluation_executor=executor)
    close_recovery_method(method)
    assert closed == ['executor', 'primary', 'meta', 'embedding']
