import json
from types import SimpleNamespace

import pytest

from traceaad_v19_2 import Config, Evaluation, TraceAAD
from traceaad_v19_2.cli import _config, _parser
from traceaad_v19_2.models import Attempt
from traceaad_v19_2.prompts import PromptBuilder
from .test_selection import program, attempt


def test_episode_advances_through_regression_then_finds_a_better_implementation(task, reply, scripted_model):
    config = Config(evaluation_budget=6, initial_size=1, development_calls=4,
                    refine_weight=1, explore_weight=0, crossover_weight=0)
    model = scripted_model([reply(v) for v in (10, 14, 12, 8, 9, 11)])
    with TraceAAD(task, model, config=config) as search:
        for _ in range(6):
            search.step()
        assert [a.parent_id for a in search.attempts] == [0, 1, 2, 3, 4, 4]
        assert search.best.evaluation.fitness == 8
        assert search.attempts[4].selection['episode_calls_before'] == 3
        assert search.attempts[5].selection['episode_calls_before'] == 0
        assert '[Development Anchor]' in model.prompts[2]
        assert 'return 10' in model.prompts[2] and 'return 14' in model.prompts[2]


@pytest.mark.parametrize('failure', ['format', 'runtime', 'api'])
def test_last_episode_slot_charges_failures_without_extra_repair_or_fake_reward(
    task, reply, scripted_model, failure
):
    bad = {'format': '{}', 'runtime': reply(body='return 1 / 0'), 'api': RuntimeError('offline')}[failure]
    config = Config(evaluation_budget=10, initial_size=1, development_calls=2)
    with TraceAAD(task, scripted_model([reply(1), reply(2), bad, reply(3)]), config=config) as search:
        search.step()
        first = search.step()
        failed = search.step()
        assert first.selection['episode_id'] == failed.selection['episode_id']
        assert failed.model_calls == 1 and not failed.repair_used
        assert failed.outcome_fitness is None
        next_attempt = search.step()
        assert next_attempt.selection['episode_id'] != first.selection['episode_id']
        assert search.model_calls == 4


@pytest.mark.parametrize('failure', ['format', 'runtime'])
def test_repair_consumes_the_same_episode_allowance(task, reply, scripted_model, failure):
    bad = '{}' if failure == 'format' else reply(body='return 1 / 0')
    config = Config(evaluation_budget=10, initial_size=1, development_calls=3)
    with TraceAAD(task, scripted_model([reply(1), bad, reply(2), reply(3), reply(4)]), config=config) as search:
        search.step()
        repaired = search.step()
        assert repaired.repair_used and repaired.model_calls == 2
        last = search.step()
        assert last.selection['episode_calls_before'] == 2
        assert last.selection['episode_id'] == repaired.selection['episode_id']
        assert search.step().selection['episode_calls_before'] == 0


def test_duplicates_are_archived_and_charged_but_do_not_create_fresh_competitors(task, reply, scripted_model):
    with TraceAAD(task, scripted_model([reply(2)] * 6),
                  config=Config(evaluation_budget=6, initial_size=3)) as search:
        result = search.run()
        assert result.evaluations == result.model_calls == result.valid_programs == 6
        assert search.attempts[0].duplicate_of is None
        assert [a.duplicate_of for a in search.attempts[1:]] == [1] * 5
        assert all(a.selection['segment_count'] == 1 for a in search.attempts[3:])


def test_repaired_result_context_names_final_version_and_records_real_source(task):
    archive = {1: program(1, 1), 2: program(2, 2, 1),
               3: program(3, 3, 1, operator='Explore'), 4: program(4, 4, 3)}
    old = attempt(1, 1, 2)
    relevant = attempt(2, 3, 4, calls=2)
    relevant.repair_used = True
    api_failure = attempt(3, 4)
    api_failure.calls = [{'status': 'error', 'error': 'service not reachable'}]
    format_failure = attempt(4, 4)
    format_failure.calls = [{'format_failure': 'bad JSON'}]
    builder = PromptBuilder(task, Config())
    prompt = builder.evolution(archive, archive[4], 'Refine', attempts=[old, relevant, api_failure, format_failure],
                               selection={'episode_anchor_id': 3})
    assert 'Attempt 1:' not in prompt
    assert 'Attempt 2:' in prompt and '-    return 3' in prompt and '+    return 4' in prompt
    assert 'repair used: True' in prompt and 'final evaluated version' in prompt
    assert 'service not reachable' not in prompt and 'Attempt 3' not in prompt
    assert 'Attempt 4' in prompt and 'no algorithm was evaluated' in prompt
    assert 'Return the complete implementation, not an edit patch.' in prompt
    assert '[Development Anchor]' in prompt


def test_crossover_context_keeps_both_actual_sources(task):
    archive = {1: program(1, 1), 2: program(2, 2)}
    text = PromptBuilder(task, Config()).evolution(archive, archive[1], 'Crossover', archive[2])
    assert '[Reference Algorithm]' in text and 'return 2' in text and 'return 1' in text


def test_source_edit_truncation_is_explicit(task):
    before, after = program(1, 1), program(2, 2)
    after.code += '\n'.join(f'v{i} = {i}' for i in range(200))
    edit = PromptBuilder(task, Config())._source_edit(before, after)
    assert 'Edit excerpt truncated' in edit
    assert len(edit) < 2500


def test_checkpoint_schema4_rejected_before_old_config_is_parsed(task, scripted_model, tmp_path):
    with TraceAAD(task, scripted_model([]), run_dir=tmp_path):
        pass
    path = tmp_path/'state.json'
    state = json.loads(path.read_text())
    state['schema_version'] = 4
    state['config']['trajectory_decay'] = .5
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='schema 5'):
        TraceAAD(task, scripted_model([]), run_dir=tmp_path, resume=True)
    args = _parser().parse_args(['run', '--task', 'tsp_construct', '--run-dir', str(tmp_path), '--resume'])
    with pytest.raises(ValueError, match='schema 5'):
        _config(args)


def test_interruption_spends_an_episode_slot_without_resending(task, reply, scripted_model, tmp_path):
    config = Config(evaluation_budget=4, initial_size=1, development_calls=2)
    with TraceAAD(task, scripted_model([reply(1), KeyboardInterrupt()]), config=config, run_dir=tmp_path) as search:
        search.step()
        with pytest.raises(KeyboardInterrupt):
            search.step()
        episode = search.attempts[-1].selection['episode_id']
    with TraceAAD(task, scripted_model([reply(2), reply(3)]), run_dir=tmp_path, resume=True) as resumed:
        trial = resumed.step()
        assert trial.selection['episode_id'] == episode
        assert trial.selection['episode_calls_before'] == 1
        assert resumed.model_calls == 3
        assert resumed.step().selection['episode_id'] != episode


@pytest.mark.parametrize('kwargs', [dict(development_calls=0), dict(development_calls=True),
                                  dict(parent_floor=1), dict(parent_floor=float('nan')),
                                  dict(local_history=1), dict(protect_new_segments=0)])
def test_invalid_policy_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_cli_and_batch_propagate_ablation_configuration(tmp_path):
    args = _parser().parse_args(['run', '--task', 'tsp_construct', '--run-dir', str(tmp_path),
                                '--development-calls', '1', '--parent-floor', '.2',
                                '--no-protect-new-segments', '--no-local-history', '--reselect-parent',
                                '--refine-weight', '.7', '--explore-weight', '.2', '--crossover-weight', '.1'])
    config = _config(args)
    assert config.development_calls == 1 and config.parent_floor == .2
    assert not config.protect_new_segments and not config.local_history
    assert config.reselect_parent and config.refine_weight == .7 and config.crossover_weight == .1
    from experiments.traceaad_v19_2.launch_host import build_plan
    args = SimpleNamespace(budget=20, initial_size=2, max_tokens=200, model_timeout=10,
                           development_calls=2, parent_floor=.1, no_protect_new_segments=True,
                           no_local_history=True, reselect_parent=True, refine_weight=.7, explore_weight=.2,
                           crossover_weight=.1, repeats=1, batch='offline', experiment='offline',
                           scheduler_socket='/tmp/unused', seconds_per_instance=10,
                           env_file=None, skip_heldout=True)
    rows = build_plan(args)
    assert len(rows) == 6
    for row in rows:
        cmd = row['command']
        assert cmd[cmd.index('--development-calls')+1] == '2'
        assert '--no-protect-new-segments' in cmd and '--no-local-history' in cmd
        assert '--reselect-parent' in cmd and cmd[cmd.index('--refine-weight')+1] == '0.7'


def test_reselection_changes_parent_continuation_without_changing_block_rules():
    from traceaad_v19_2.selection import joint_weights
    archive = {1: program(1, 1), 2: program(2, 2), 3: program(3, 4, 1)}
    trials = [attempt(1, 1, 3, calls=2)]
    probabilities, ordinary = joint_weights(archive, trials, Config(protect_new_segments=False))
    control, rows = joint_weights(archive, trials, Config(protect_new_segments=False, reselect_parent=True))
    assert probabilities == {(3, 'Refine'): 1}
    assert set(control) == {(1, 'Refine'), (2, 'Refine')}
    for (pid, op), row in rows.items():
        assert row['episode_id'] == ordinary[(3, 'Refine')]['episode_id']
        assert row['episode_calls_before'] == 2 and row['development_calls'] == 4
        assert row['episode_anchor_id'] == pid and row['phase'] == 'reselect'
    trials[0].selection['phase'] = 'initial_development'
    control, _ = joint_weights(archive, trials, Config(reselect_parent=True))
    assert control == probabilities  # Protected initial development stays fixed.


@pytest.mark.parametrize('reselect', [False, True])
def test_control_preserves_repair_and_request_cap(task, reply, scripted_model, reselect):
    config = Config(evaluation_budget=8, initial_size=1, development_calls=4,
                    protect_new_segments=False, reselect_parent=reselect,
                    refine_weight=1, explore_weight=0, crossover_weight=0)
    with TraceAAD(task, scripted_model([reply(1), reply(2), '{}', reply(3), '{}', reply(4)]), config=config) as search:
        search.step()
        first = search.step()
        repaired = search.step()
        last = search.step()
        following = search.step()
        assert [a.model_calls for a in (first, repaired, last)] == [1, 2, 1]
        assert repaired.repair_used and not last.repair_used
        assert [a.selection['episode_calls_before'] for a in (first,repaired,last,following)] == [0,1,3,0]
        assert len({a.selection['episode_id'] for a in (first,repaired,last)}) == 1
        assert following.selection['episode_id'] != first.selection['episode_id']


def test_control_resume_reproduces_prompts_and_parent_draws(task, reply, scripted_model, tmp_path):
    replies=[reply(v) for v in (10, 14, 12, 8, 9, 7, 6, 5)]
    config=Config(evaluation_budget=8, initial_size=1, protect_new_segments=False,
                  reselect_parent=True, refine_weight=1, explore_weight=0, crossover_weight=0)
    uninterrupted=TraceAAD(task, scripted_model(replies), config=config)
    expected=uninterrupted.run()
    with TraceAAD(task, scripted_model(replies[:4]), config=config, run_dir=tmp_path) as partial:
        for _ in range(4): partial.step()
    model=scripted_model(replies[4:])
    resumed=TraceAAD(task, model, run_dir=tmp_path, resume=True)
    assert resumed.run().to_dict() == expected.to_dict()
    assert model.prompts == uninterrupted.model.prompts[4:]
    assert [a.selection for a in resumed.attempts] == [a.selection for a in uninterrupted.attempts]


@pytest.mark.parametrize('operator', ['Explore', 'Crossover'])
def test_discovery_does_not_refer_to_an_absent_anchor(task, operator):
    archive={1:program(1,1),2:program(2,2)}
    prompt=PromptBuilder(task,Config()).evolution(archive,archive[1],operator,
                                               archive[2] if operator=='Crossover' else None)
    assert 'development anchor' not in prompt.lower()


def test_package_can_be_renamed_and_resumed_without_repository(task, tmp_path):
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    source=Path(__file__).resolve().parents[2]/'traceaad_v19_2'
    shutil.copytree(source, tmp_path/'independent_aad', ignore=shutil.ignore_patterns('__pycache__'))
    script='''
import sys, json, importlib.abc
sys.path.insert(0, sys.argv[1])
class BlockRepository(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'traceaad', 'traceaad_base', 'traceaad_v19_1', 'core', 'experiments', 'benchmarks', 'tests'}:
            raise AssertionError('Repository import: ' + fullname)
sys.meta_path.insert(0, BlockRepository())
from independent_aad import Config, Task, TraceAAD
class Model:
    def complete(self, prompt, **kwargs):
        return json.dumps({'analysis':'Use input.', 'design':'Return input.', 'code':'def solve(x): return x'})
task=Task('Minimize.', 'def solve(x): pass', lambda code, f:f(2), 2)
config=Config(evaluation_budget=4, initial_size=1, reselect_parent=True)
with TraceAAD(task,Model(),config=config,run_dir=sys.argv[2]) as search:
    search.step()
    search.step()
search=TraceAAD(task,Model(),run_dir=sys.argv[2],resume=True)
result=search.run()
assert result.evaluations == result.model_calls == 4
assert result.best.evaluation.fitness == 2
assert all(a.duplicate_of == 1 for a in search.attempts[1:])
print('isolated package resume passed')
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(tmp_path),str(tmp_path/'run')],
                          capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
