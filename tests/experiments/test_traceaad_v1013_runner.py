import pytest


def test_runner_rejects_existing_results_without_journal(tmp_path, monkeypatch):
    from experiments.traceaad_v10_13 import run as runner

    monkeypatch.setattr(runner, 'RESULTS_ROOT', tmp_path)
    run_dir = tmp_path / 'traceaad_v10_13' / 'tsp_construct' / 'old_run'
    run_dir.mkdir(parents=True)
    (run_dir / 'run_config.json').write_text('{}')

    with pytest.raises(SystemExit, match='no V10.13 journal'):
        runner.main(['--task', 'tsp_construct', '--run-name', 'old_run'])

    assert (run_dir / 'run_config.json').read_text() == '{}'
