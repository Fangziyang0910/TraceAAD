from collections import Counter

import pytest

from experiments.traceaad_v10_13.launch import (
    BACKEND_POOL,
    TARGET_DISTRIBUTION,
    build_plan,
    validate_plan,
    run_dir,
)
from experiments.infra.base import RESULTS_ROOT, TASKS


def test_v1013_plan_has_four_repeats_and_confirmed_three_pool_distribution():
    plan = build_plan("test_v1013", "test_v1013")
    validate_plan(plan)
    assert len(plan) == 20
    assert Counter(row["backend"] for row in plan) == Counter(TARGET_DISTRIBUTION)
    assert set(row["backend"] for row in plan) == set(BACKEND_POOL)
    assert {(row["task"], row["repeat"]) for row in plan} == {
        (task, repeat) for task in TASKS for repeat in range(1, 5)
    }
    assert all(row["session"].startswith("test_v1013_") for row in plan)
    assert all(run_dir(row) == RESULTS_ROOT / "traceaad_v10_13" / row["task"] / row["run_name"]
               for row in plan)


def test_error_summary_blocks_relaunch_even_if_session_is_alive(monkeypatch):
    from experiments.traceaad_v10_13 import launch
    row = build_plan('test', 'test')[0]
    row['status'] = 'running'
    monkeypatch.setattr(launch, 'get_summary_status', lambda path: 'error')
    monkeypatch.setattr(launch, 'session_alive', lambda session: True)
    launch.refresh([row])
    assert row['status'] == 'blocked' and row['last_error'] == 'error'


def test_runner_rejects_existing_results_without_journal(tmp_path, monkeypatch):
    from experiments.traceaad_v10_13 import run as runner

    monkeypatch.setattr(runner, 'RESULTS_ROOT', tmp_path)
    run_dir = tmp_path / 'traceaad_v10_13' / 'tsp_construct' / 'old_run'
    run_dir.mkdir(parents=True)
    (run_dir / 'run_config.json').write_text('{}')

    with pytest.raises(SystemExit, match='no V10.13 journal'):
        runner.main(['--task', 'tsp_construct', '--run-name', 'old_run'])

    assert (run_dir / 'run_config.json').read_text() == '{}'
