"""Recovery must preserve budget and reject invented search state."""
import json
import gzip
from types import SimpleNamespace

import pytest

from core import TextFunctionProgramConverter as C
from core import Evaluation, LLM
from baselines.eoh.eoh import EoH
from baselines.funsearch.funsearch import FunSearch
from baselines.funsearch.programs_database import ProgramsDatabase
from baselines.funsearch.config import ProgramsDatabaseConfig
from baselines.mcts_ahd.mcts_ahd import MCTS_AHD
from baselines.shared.profiler import ProfilerBase
from experiments.infra.resume_baseline import equivalent_source, history, recover_database, recover_tree, reprofile_saved
from traceaad.common.storage import append_jsonl, write_json


def test_quota_errors_and_unfinished_alignment_do_not_spend_budget(tmp_path):
    write_json(tmp_path / 'run_config.json', {'method': 'mcts_ahd', 'budget': 1000})
    write_json(tmp_path / 'resume.json', {'method_state': {'sample_order': 7}})
    append_jsonl(tmp_path / 'events.jsonl', {'kind': 'candidate', 'budget_used': 6})
    append_jsonl(tmp_path / 'calls.jsonl', {'stage': 'generate', 'sample_order': 8,
        'function_parse_success': True, 'parsed_thought': 'idea'})
    append_jsonl(tmp_path / 'calls.jsonl', {'stage': 'sample_error', 'sample_order': 8,
        'error': '429'})
    assert history(tmp_path)[-1] == 7
    append_jsonl(tmp_path / 'calls.jsonl', {'stage': 'thought_alignment', 'sample_order': 8})
    assert history(tmp_path)[-1] == 8


def test_recovery_reads_compressed_history_and_new_calls_without_duplicates(tmp_path):
    write_json(tmp_path / 'run_config.json', {'method': 'eoh', 'budget': 1000})
    append_jsonl(tmp_path / 'events.jsonl', {'kind': 'candidate', 'budget_used': 7})
    old = {'transport_id': 'old', 'stage': 'generate', 'sample_order': 7}
    with gzip.open(tmp_path / 'calls.jsonl.gz', 'wt') as f:
        f.write(json.dumps(old) + '\n')
    append_jsonl(tmp_path / 'calls.jsonl', old)
    append_jsonl(tmp_path / 'calls.jsonl', {'transport_id': 'new', 'stage': 'generate', 'sample_order': 8})
    _, _, calls, _, _, used = history(tmp_path)
    assert [r['transport_id'] for r in calls] == ['old', 'new']
    assert used == 8


def test_resume_profiler_uses_actual_budget_including_unrecorded_failures(tmp_path):
    profiler = ProfilerBase(tmp_path)
    profiler._method = SimpleNamespace(_tot_sample_nums=11)
    profiler._resume_budget_from_method = True
    profiler._num_samples = 6
    f = C.text_to_function('def solve(x):\n    return x\n')
    f.score, f.operator = 1.0, 'm1'
    profiler.register_function(f, program=str(f))
    row = json.loads((tmp_path / 'events.jsonl').read_text().splitlines()[-1])
    assert row['candidate_id'] == 7
    assert row['budget_used'] == 11


def test_completed_baseline_freezes_with_the_real_logging_configuration(tmp_path, monkeypatch):
    class FreezeEvaluator:
        def __init__(self, evaluation, *, seeds, n_workers, timeout_seconds):
            assert seeds == [730241]

        def evaluate(self, code, key):
            return {'fitness': 1.0, 'key': key, 'evaluations': []}

    monkeypatch.setattr('traceaad.common.instance_evaluation.InstanceProgramEvaluator', FreezeEvaluator)
    write_json(tmp_path / 'run_config.json', {'method': 'eoh', 'budget': 1,
        'method_params': {'evaluation_seeds': [730241]}})
    profiler = ProfilerBase(tmp_path)
    profiler._evaluation = SimpleNamespace(_instance_execution={
        'n_workers': 1, 'timeout_seconds': 20, 'record_path': '/unused/evaluations.jsonl'})
    f = function(1.0, 'x')
    profiler.register_function(f, program=str(f))
    other = function(2.0, 'x+1')
    profiler.register_function(other, program=str(other))
    from hashlib import sha256
    profiler._recovery_profiles = {
        sha256(str(f).encode()).hexdigest(): {'score': None},
        sha256(str(other).encode()).hexdigest(): {'score': 3.0}}
    profiler._recovery_candidate_cutoff = 2
    profiler.write_run_summary(status='finished')
    assert json.loads((tmp_path / 'summary.json').read_text())['status'] == 'finished'
    assert (tmp_path / 'selection.json').exists()
    assert json.loads((tmp_path / 'selection.json').read_text())['selected_node'] == 2


def function(score, body):
    f = C.text_to_function('def solve(x):\n    return ' + body + '\n')
    f.score, f.algorithm = score, body
    return f


def test_mcts_replays_edges_and_visits_without_flattening_tree():
    method = MCTS_AHD.__new__(MCTS_AHD)
    method.alpha, method.lambda_0, method._init_pop_size, method._selection_num = .5, .1, 1, 1
    funcs = {1: function(3.0, 'x'), 2: function(2.0, 'x+1')}
    events = [{'kind': 'candidate', 'candidate_id': 1},
        {'profiler_sample_order': 2, 'data': {'event': 'expand', 'status': 'expanded',
            'child_score': 2.0, 'parent_score': 3.0, 'parent_depth': 1,
            'parent_visits': 2, 'parent_children_count': 1, 'root_parent': False, 'operator': 'm1'}}]
    tree, nodes = recover_tree(method, events, [], funcs)
    assert nodes[2].parent is nodes[1]
    assert tree.root.Q == -2.0
    assert tree.root.visits == 2
    events[1]['data']['parent_score'] = 99
    with pytest.raises(ValueError, match='parent identity'):
        recover_tree(method, events, [], funcs)


def test_funsearch_reprofile_preserves_vector_clusters_and_does_not_recall_model(tmp_path):
    template = C.text_to_program('def solve(x):\n    return x\n')
    db = ProgramsDatabase(ProgramsDatabaseConfig(num_islands=2), template, 'solve')
    evaluator = SimpleNamespace(evaluate_program_record_time_with_details=lambda program:
        (SimpleNamespace(result=2.0, instance_scores=(1.0, 3.0)), 0.0))
    method = SimpleNamespace(_database=db, _template_program=template, _evaluator=evaluator,
                             _profiler=SimpleNamespace())
    candidates = [{'candidate_id': 1, 'operator': 'seed', 'program': {'key': 'a'},
                   'progress': {'elapsed': 3.0}}]
    state = {'island_resets': 0, 'island_program_counts': [1, 1],
             'island_cluster_counts': [1, 1], 'island_best_scores': [2.0, 2.0]}
    write_json(tmp_path / 'run_config.json', {'task': 'test', 'method_params': {'eval_workers': 1}})
    append_jsonl(tmp_path / 'recovery_profiles.jsonl', {'key': 'a', 'score': 3.0, 'instance_scores': [2.0, 4.0]})
    recover_database(method, tmp_path, [], state, candidates, {1: function(2.0, 'x')}, evaluate_profiles=True)
    assert list(db.islands[0].clusters) == [(-2.0, -4.0)]
    assert db.best_score_per_island == [-3.0, -3.0]
    assert method._profiler._canonical_best['fitness'] == 3.0
    assert (tmp_path / 'recovery_profiles.jsonl').exists()


def test_missing_island_reset_founders_are_not_guessed(tmp_path):
    with pytest.raises(ValueError, match='founders'):
        recover_database(SimpleNamespace(_database=None), tmp_path, [], {'island_resets': 1},
                         [], {}, evaluate_profiles=False)


def test_changed_reprofile_score_is_saved_separately_from_history(monkeypatch):
    evaluator = SimpleNamespace(evaluate_program_with_details=lambda code:
        SimpleNamespace(result=1.5, instance_scores=(1.0, 2.0), failure_kind=None))
    monkeypatch.setattr('experiments.infra.resume_baseline.profile_evaluator', lambda *args: evaluator)
    record = reprofile_saved(('key', 'source', 1.0, 'task', 8))
    assert record['score'] == 1.5 and record['previous_score'] == 1.0
    assert record['instance_scores'] == [1.0, 2.0]


def test_new_funsearch_candidates_save_the_instance_vector_needed_for_recovery():
    method = FunSearch.__new__(FunSearch)
    recorded = []
    method._profiler = SimpleNamespace(register_function=lambda *args, **kwargs: None,
        log_method_event=lambda **payload: recorded.append(payload))
    method._database = SimpleNamespace(register_program=lambda *args: None)
    method._tot_sample_nums = 12
    f = function(2.0, 'x')
    f.instance_scores = (-1.0, -3.0)
    method._register(f, C.text_to_program(str(f)), 2.0, 0.0, 0.0, 0, 'funsearch')
    assert recorded == [{'event': 'evaluation_profile', 'sample_order': 12,
                         'score': 2.0, 'instance_scores': [-1.0, -3.0]}]


def test_source_matching_tolerates_template_docstrings_but_not_computation_changes():
    assert equivalent_source('def f(x):\n    return x\n', 'def f(x):\n    """template"""\n    return x\n')
    assert not equivalent_source('def f(x):\n    return x\n', 'def f(x):\n    return x+1\n')


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program='def solve(x):\n    return x\n',
                         task_description='Minimize value.', safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return float(callable_func(1))


class OneCall(LLM):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def draw_sample(self, prompt, **kwargs):
        self.prompts.append(prompt)
        assert len(self.prompts) == 1, 'Recovery restarted initialization or exceeded remaining budget'
        return '{Change the return value.}\n```python\ndef solve(x):\n    return -2\n```'


@pytest.mark.parametrize('name', ['eoh', 'funsearch'])
def test_recovered_run_uses_only_remaining_budget_without_initialization(name):
    llm = OneCall()
    if name == 'eoh':
        method = EoH(llm, TinyEvaluation(), profiler=None, max_sample_nums=4,
                     pop_size=2, selection_num=2, num_samplers=1, operators=['m1'])
        method._population._population = [function(1.0, 'x'), function(2.0, 'x+1')]
    else:
        method = FunSearch(llm, TinyEvaluation(), profiler=None, max_sample_nums=4)
        method._database.register_program(function(1.0, 'x'), None, -1.0)
    method._resume_mode = True
    method._tot_sample_nums = 3
    method.run()
    assert method._tot_sample_nums == 4
    assert len(llm.prompts) == 1


def test_interrupted_search_is_not_finalized(tmp_path):
    from baselines.shared.observability import finish_profiler
    profiler = ProfilerBase(tmp_path)
    write_json(tmp_path / 'run_config.json', {'method': 'funsearch', 'budget': 10})
    method = SimpleNamespace(_profiler=profiler, _tot_sample_nums=3, _max_sample_nums=10)
    finish_profiler(method)
    assert json.loads((tmp_path / 'summary.json').read_text())['status'] == 'stopped'
    assert not (tmp_path / 'selection.json').exists()


def test_resume_after_hard_stop_uses_lock_instead_of_stale_running_status(tmp_path, monkeypatch):
    from experiments.infra import resume_baseline as resume
    import sys
    config = {'task': 'tsp_construct', 'run_name': 'test', 'method_params': {}}
    monkeypatch.setattr(resume, 'history', lambda directory: (config,))
    monkeypatch.setattr('benchmarks.tasks.validate_task_interface', lambda *args: None)
    executed = []
    method = SimpleNamespace(run=lambda: executed.append(True), _profiler=SimpleNamespace(
        _evaluation=SimpleNamespace(_instance_execution={})))
    monkeypatch.setattr(resume, 'build', lambda *args: method)
    monkeypatch.setattr(resume, 'recover', lambda *args, **kwargs: {'remaining': 1})
    monkeypatch.setattr(sys, 'argv', ['resume_baseline', '--run-dir', str(tmp_path), '--execute'])
    write_json(tmp_path / 'summary.json', {'status': 'running', 'writer_lock': True})
    resume.main()
    assert executed == [True]


def test_completed_funsearch_recovery_keeps_reprofiled_ranking(tmp_path):
    from experiments.infra.resume_baseline import recover
    profiler = ProfilerBase(tmp_path)
    write_json(tmp_path / 'run_config.json', {'method': 'funsearch', 'task': 'test', 'budget': 1})
    f = function(1.0, 'x')
    f.operator = 'seed'
    profiler.register_function(f, program=str(f))
    key = profiler._canonical_best['key']
    append_jsonl(tmp_path / 'recovery_profiles.jsonl', {'key': key, 'score': 3.0,
                                                       'instance_scores': [3.0]})
    method = SimpleNamespace(_profiler=profiler, _template_program=C.text_to_program(str(f)))
    report = recover(method, tmp_path)
    assert report['remaining'] == 0
    assert profiler._canonical_best['fitness'] == 3.0
    assert profiler._recovery_profiles[key]['score'] == 3.0


def test_run_lock_rejects_duplicate_writer_and_cleans_exception(tmp_path):
    from experiments.infra.runner import baseline_run_lock
    with pytest.raises(RuntimeError, match='interrupted'):
        with baseline_run_lock(tmp_path):
            write_json(tmp_path / 'summary.json', {'status': 'running', 'writer_lock': True})
            with pytest.raises(ValueError, match='active writer'):
                with baseline_run_lock(tmp_path):
                    pytest.fail('Two writers acquired the same run')
            raise RuntimeError('interrupted')
    assert json.loads((tmp_path / 'summary.json').read_text())['status'] == 'stopped'
    with baseline_run_lock(tmp_path):
        pass


def test_hard_kill_releases_writer_lock(tmp_path):
    import subprocess
    import sys
    from experiments.infra.runner import baseline_run_lock
    child = subprocess.Popen([sys.executable, '-u', '-c', '''
import sys, time
from pathlib import Path
from experiments.infra.runner import baseline_run_lock
from traceaad.common.storage import write_json
p = Path(sys.argv[1])
with baseline_run_lock(p):
    write_json(p / 'summary.json', {'status': 'running', 'writer_lock': True})
    print('ready', flush=True)
    time.sleep(60)
''', str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        with pytest.raises(ValueError, match='active writer'):
            with baseline_run_lock(tmp_path):
                pass
        child.kill()
        child.wait(timeout=5)
        assert json.loads((tmp_path / 'summary.json').read_text())['status'] == 'running'
        with baseline_run_lock(tmp_path):
            pass
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdout.close()


@pytest.mark.parametrize('execute', [False, True])
def test_batch_resume_uses_local_directories_and_skips_frozen_runs(tmp_path, monkeypatch, execute):
    from experiments.infra import resume_baseline as resume
    manifest = tmp_path / 'batch_test.json'
    plans = [{'task': 'test', 'run_name': name, 'session': name, 'run_dir': '/old/remote/' + name}
             for name in ('frozen', 'active', 'paused')]
    write_json(manifest, {'method': 'eoh', 'plan': plans})
    for plan in plans:
        directory = tmp_path / plan['task'] / plan['run_name']
        directory.mkdir(parents=True)
        write_json(directory / 'run_config.json', {'method': 'eoh', 'task': 'test',
                   'run_name': plan['run_name'], 'budget': 10})
        append_jsonl(directory / 'events.jsonl', {'kind': 'candidate', 'budget_used': 1})
    write_json(tmp_path / 'test/frozen/summary.json', {'status': 'finished'})
    write_json(tmp_path / 'test/frozen/selection.json', {})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0 if command[:3] == ['tmux', 'has-session', '-t']
                               and command[3] == 'active' else 1)
    monkeypatch.setattr(resume.subprocess, 'run', run)
    resume.resume_manifest(manifest, execute=execute)
    if execute:
        launched = [c for c in calls if c[:2] == ['tmux', 'new-session']]
        assert len(launched) == 1
        assert '--execute' in launched[0][-1]
        assert str(tmp_path / 'test/paused') in launched[0][-1]
        assert '/old/remote' not in launched[0][-1]
    else:
        assert len(calls) == 2
        assert all('--execute' not in c and 'tmux' not in c for c in calls)


def test_funsearch_saved_vectors_resume_without_cpu_reevaluation(tmp_path, monkeypatch):
    template = C.text_to_program('def solve(x):\n    return x\n')
    db = ProgramsDatabase(ProgramsDatabaseConfig(num_islands=2), template, 'solve')
    method = SimpleNamespace(_database=db, _profiler=SimpleNamespace())
    candidates = [{'candidate_id': 1, 'operator': 'seed', 'valid': True, 'fitness': 2.0,
                   'program': {'key': 'a'}, 'progress': {'elapsed': 0.0}}]
    events = [{'profiler_sample_order': 1, 'data': {'event': 'evaluation_profile',
               'instance_scores': [-1.0, -3.0]}}]
    write_json(tmp_path / 'run_config.json', {'task': 'test', 'method_params': {'eval_workers': 1}})
    def forbidden_reprofile(item):
        pytest.fail('Saved vectors must not be reevaluated during recovery')
    monkeypatch.setattr('experiments.infra.resume_baseline.reprofile_saved', forbidden_reprofile)
    recover_database(method, tmp_path, events, {'island_resets': 0}, candidates,
                     {1: function(2.0, 'x')}, evaluate_profiles=True)
    assert list(db.islands[0].clusters) == [(-1.0, -3.0)]
    assert db.best_score_per_island == [-2.0, -2.0]
    assert not (tmp_path / 'recovery_profiles.jsonl').exists()


def test_resumed_writer_preserves_and_removes_an_incomplete_jsonl_tail(tmp_path):
    import os
    from experiments.infra.runner import baseline_run_lock
    from traceaad.common.storage import rows
    path = tmp_path / 'events.jsonl'
    append_jsonl(path, {'complete': 1})
    with path.open('ab') as handle:
        handle.write(b'{"interrupted":')
    linked = tmp_path / 'old_copy'
    os.link(path, linked)
    with baseline_run_lock(tmp_path):
        append_jsonl(path, {'complete': 2})
    assert list(rows(path)) == [{'complete': 1}, {'complete': 2}]
    assert linked.read_bytes().endswith(b'{"interrupted":')
    assert next((tmp_path / 'logs/interrupted_tails').glob('*.partial')).read_bytes() == b'{"interrupted":'


def test_interrupted_call_compression_keeps_previous_archive_and_source(tmp_path, monkeypatch):
    from traceaad.common import storage
    append_jsonl(tmp_path / 'calls.jsonl', {'transport_id': 'old'})
    storage.seal_calls(tmp_path)
    previous = (tmp_path / 'calls.jsonl.gz').read_bytes()
    append_jsonl(tmp_path / 'calls.jsonl', {'transport_id': 'new'})
    copy = storage.shutil.copyfileobj
    def interrupt(source, destination, *args):
        copy(source, destination, *args)
        if source.name.endswith('calls.jsonl'):
            raise RuntimeError('compression interrupted')
    with monkeypatch.context() as patch:
        patch.setattr(storage.shutil, 'copyfileobj', interrupt)
        with pytest.raises(RuntimeError, match='compression interrupted'):
            storage.seal_calls(tmp_path)
    assert (tmp_path / 'calls.jsonl.gz').read_bytes() == previous
    assert list(storage.rows(tmp_path / 'calls.jsonl')) == [{'transport_id': 'new'}]
    storage.seal_calls(tmp_path)
    assert list(storage.rows(tmp_path / 'calls.jsonl.gz')) == [
        {'transport_id': 'old'}, {'transport_id': 'new'}]
