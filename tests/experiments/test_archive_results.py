import hashlib
import json
import os

import pytest

from experiments.infra.archive_results import archive_results
from experiments.infra import batch_status
from experiments.monitor import ResultsMonitor
from tests.experiments.test_training_monitor import make_run, write_json
from tests.experiments.test_batch_status import batch, freeze
from traceaad.v12_1.storage import selected_program


def test_archive_preserves_payload_and_original_hardlinked_metadata(tmp_path):
    source, target = tmp_path / 'results', tmp_path / 'archive'
    run = source / 'old' / 'task' / 'run'
    write_json(run / 'run_config.json', {'llm': {'model': 'Qwen3.8'}, 'task': 'task'})
    write_json(run / 'summary.json', {'status': 'running', 'budget_used': 7})
    raw = (run / 'summary.json').read_bytes()
    keep = source / 'keep.json'
    os.link(run / 'summary.json', keep)
    (run / 'events.jsonl').write_bytes(b'unchanged payload\n')
    result = archive_results(source, target, ['old'])
    assert result['status'] == 'archived' and result['verified']
    assert not (source / 'old').exists()
    assert keep.read_bytes() == raw
    assert (target / 'old/task/run/events.jsonl').read_bytes() == b'unchanged payload\n'
    assert (target / 'original_metadata/old/task/run/summary.json').read_bytes() == raw
    summary = json.loads((target / 'old/task/run/summary.json').read_text())
    assert summary['status'] == 'archived' and summary['archived_from_status'] == 'running'
    assert summary['budget_used'] == 7
    records = [json.loads(line) for line in (target / 'integrity.jsonl').read_text().splitlines()]
    assert next(r for r in records if r['path'].endswith('summary.json'))['sha256'] == hashlib.sha256(raw).hexdigest()


def test_archive_refuses_to_move_deepseek_results(tmp_path):
    source, target = tmp_path / 'results', tmp_path / 'archive'
    run = source / 'new/task/run'
    write_json(run / 'run_config.json', {'llm': {'model': 'deepseek-v4.1-flash'}})
    with pytest.raises(ValueError, match='DeepSeek'):
        archive_results(source, target, ['new'])
    assert run.exists() and not target.exists()


def test_archived_hot_journal_is_not_running(tmp_path):
    results, name = make_run(tmp_path)
    write_json(results / 'tsp_construct' / name / 'summary.json', {'status': 'archived'})
    state = ResultsMonitor(results.parent, results.name).overview('results')
    assert state['tasks'][0]['runs'][0]['status'] == 'archived'
    assert state['summary']['archived'] == 1 and state['summary']['running'] == 0
    assert state['summary']['timing']['active_runs'] == 0


def test_archive_view_does_not_rewrite_cached_files(tmp_path):
    results, name = make_run(tmp_path)
    run = results / 'tsp_construct' / name
    write_json(run / 'summary.json', {'status': 'archived'})
    cache = run / '.cache/history.json'
    write_json(cache, {'old_cache': 'preserve it'})
    before = cache.read_bytes()
    ResultsMonitor(results.parent, results.name).overview('results')
    assert cache.read_bytes() == before


def test_archive_protects_deepseek_logs_without_run_config(tmp_path):
    source, target = tmp_path / 'results', tmp_path / 'archive'
    logs = source / 'method/launch_logs'
    logs.mkdir(parents=True)
    (logs / 'deepseek_off_sync.log').write_text('live')
    with pytest.raises(ValueError, match='DeepSeek'):
        archive_results(source, target, ['method/launch_logs'])
    assert logs.exists() and not target.exists()


def test_archived_finished_program_remains_readable_and_batch_is_inactive(tmp_path):
    manifest, directory = batch(tmp_path)
    freeze(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    summary.update(status='archived', archived_from_status='finished')
    write_json(directory / 'summary.json', summary)
    assert selected_program(directory)['verified']
    result = batch_status.collect(manifest)
    assert result['runs'][0]['status'] == 'archived'
    assert result['search_timing']['active_runs'] == 0
    assert result['summary']['ready_for_heldout'] == 0
