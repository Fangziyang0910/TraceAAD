from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from experiments.infra import base as _common


def test_server1_capacity_matches_current_service_limit() -> None:
    # 2026-09-17 起 server3 恢复 gpu0/gpu1 各一单卡实例，各 8 路
    assert _common.BACKEND_CAPACITY["server1"] == 6
    assert _common.BACKEND_CAPACITY["server3"] == 8
    assert _common.BACKEND_CAPACITY["server3b"] == 8


def test_select_backend_balances_to_the_side_with_more_free_slots() -> None:
    remaining = {"server3": 1, "server3b": 3, "zhong": 9, "local": 2}

    assert _common.select_backend(remaining) == "server3b"
    remaining["server3b"] -= 1
    assert _common.select_backend(remaining) == "server3b"
    remaining["server3b"] -= 1
    assert _common.select_backend(remaining) == "server3"
    remaining["server3"] -= 1
    assert _common.select_backend(remaining) == "server3b"


def test_select_backend_breaks_ties_in_primary_order() -> None:
    assert _common.select_backend({"server3": 9, "server3b": 9}) == "server3"
    assert _common.select_backend({"server3": 0, "server3b": 0, "zhong": 9}) is None


def test_assign_backends_alternates_equal_primary_slots(monkeypatch) -> None:
    pending = [
        _common.LaunchItem(
            task="tsp_construct",
            repeat=index,
            backend=None,
            session=f"sess_{index}",
            run_name=f"run_{index}",
            run_dir=Path(f"/tmp/run_{index}"),
            seed=index,
            module="experiments.traceaad_v10_1.run",
        )
        for index in range(1, 5)
    ]
    monkeypatch.setattr(
        _common,
        "free_slots",
        lambda: {"server3": 2, "server3b": 2, "zhong": 9, "local": 0},
    )

    assigned = _common.assign_backends(pending)

    assert [item.backend for item in assigned] == [
        "server3",
        "server3b",
        "server3",
        "server3b",
    ]


def test_backend_usage_deduplicates_forked_evaluator_cmdlines(monkeypatch) -> None:
    client = (
        "/repo/.venv/bin/python3 -m experiments.traceaad_v10_1.run "
        "--task tsp_construct --backend server3 --run-name run_a"
    )
    other = client.replace("run_a", "run_b")
    launcher = (
        "--backend server3 --watch"
    )
    stdout = "\n".join((client, client, client, other, launcher)) + "\n"
    monkeypatch.setattr(
        _common.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=stdout),
    )

    assert _common.count_backend_usage()["server3"] == 2


def test_aco_tasks_default_to_four_local_eval_workers() -> None:
    from experiments.infra.base import DEFAULT_ACO_EVAL_WORKERS, build_task

    assert DEFAULT_ACO_EVAL_WORKERS == 4
    cvrp, cvrp_kwargs = build_task("cvrp_aco", None)
    op, op_kwargs = build_task("op_aco", None)
    assert cvrp.n_workers == 4
    assert op.n_workers == 4
    assert cvrp_kwargs["n_workers"] == 4
    assert op_kwargs["n_workers"] == 4

