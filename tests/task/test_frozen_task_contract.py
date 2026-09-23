"""冻结契约：五任务公共实验信息不得变动（2026-09-12 起）。

除非用户明确指示，以下内容必须与五个对比方法入表批次
（TSP/CVRP/OP/OBP 为 20260824_rerun 批，VRPTW 为 20260822_142500 批）
所用的信息逐字一致，任何后续版本/测试不得修改：

1. 五任务的 ``task_description`` 与 ``template_program``；
2. 训练/测试数据配置与种子（generated_data_config、CVRP/OP 数据划分、
   ACO 搜索参数 aco_seed=1234）；
3. 搜索种子约定：三重复 seed = repeat - 1 ∈ {0, 1, 2}。

契约说明见 docs/experiments/主实验/主实验配置说明.md 的「冻结契约」。
方法级上下文增强（如 v10.10 的 design_notes）不属于公共文本，
不在此冻结范围内，但不得改动上述公共字符串本身。
"""
from __future__ import annotations

import argparse
import hashlib

import pytest

from benchmarks.cvrp_aco import dataset as cvrp_dataset
from benchmarks.cvrp_aco import template as cvrp_template
from benchmarks.generated_data_config import (
    EVAL_SEED,
    TRAIN_SEED,
    get_generated_task_kwargs,
)
from benchmarks.online_bin_packing import template as obp_template
from benchmarks.op_aco import dataset as op_dataset
from benchmarks.op_aco import template as op_template
from benchmarks.tsp_construct import template as tsp_template
from benchmarks.vrptw_construct import template as vrptw_template

CONTRACT_DOC = "docs/02-实验结果/00-主实验配置与冻结契约.md"
CONTRACT_MSG = (
    "冻结契约被打破：任务公共文本或种子/数据配置发生变化。"
    "除非用户明确指示，不得修改这些内容；如确需变更，须先获得用户许可"
    "并同步更新本测试与契约文档（见 " + CONTRACT_DOC + "）。"
)

FROZEN_TEXT_HASHES = {
    # task: (sha256(task_description)[:16], sha256(template_program)[:16])
    "tsp_construct": ("effd18dcce5d4a2a", "898633cbd9feb3b1"),
    "cvrp_aco": ("4dacac6941fec2b8", "102544b67bde9efb"),
    "op_aco": ("8b77f14596517a6d", "dcd5060bfae5c03e"),
    "online_bin_packing": ("3bc488ec75bc3f92", "ea40237fd52e536c"),
    "vrptw_construct": ("21928d1b9acdfd9e", "39a8d504f31bd4af"),
}

TEMPLATES = {
    "tsp_construct": tsp_template,
    "cvrp_aco": cvrp_template,
    "op_aco": op_template,
    "online_bin_packing": obp_template,
    "vrptw_construct": vrptw_template,
}

FROZEN_GENERATED_KWARGS = {
    "tsp_construct": {
        "train": {"timeout_seconds": 20, "n_instance": 16, "problem_size": 50, "seed": 2024},
        "eval": {"timeout_seconds": 20, "n_instance": 16, "problem_size": 50, "seed": 2025},
    },
    "online_bin_packing": {
        "train": {
            "timeout_seconds": 30,
            "dataset_specs": [
                {"n_instances": 1, "n_items": 1000, "capacities": [100, 500]},
                {"n_instances": 1, "n_items": 5000, "capacities": [100, 500]},
            ],
            "seed": 2024,
        },
        "eval": {
            "timeout_seconds": 30,
            "dataset_specs": [
                {"n_instances": 5, "n_items": 1000, "capacities": [100, 500]},
                {"n_instances": 5, "n_items": 5000, "capacities": [100, 500]},
                {"n_instances": 5, "n_items": 10000, "capacities": [100, 500]},
            ],
            "seed": 2025,
        },
    },
    "vrptw_construct": {
        "train": {"timeout_seconds": 30, "problem_size": 50, "n_instance": 16, "seed": 2024},
        "eval": {"timeout_seconds": 30, "problem_size": 50, "n_instance": 16, "seed": 2025},
    },
}

# split: (role, problem_size, n_instances, seed)
FROZEN_ACO_SPLITS = {
    "cvrp_aco": {
        "train": ("train", 50, 10, 1234),
        "test_50": ("test", 50, 64, 1234),
        "test_100": ("test", 100, 64, 1234),
        "test_200": ("test", 200, 64, 1234),
    },
    "op_aco": {
        "train": ("train", 50, 5, 1234),
        "test_50": ("test", 50, 64, 4567),
        "test_100": ("test", 100, 64, 4567),
        "test_200": ("test", 200, 64, 4567),
    },
}

FROZEN_ACO_SEARCH_PARAMS = {
    "cvrp_aco": {
        "split": "train",
        "timeout_seconds": 120,
        "n_ants": 30,
        "n_iterations": 100,
        "aco_seed": 1234,
    },
    "op_aco": {
        "split": "train",
        "timeout_seconds": 60,
        "n_ants": 20,
        "n_iterations": 50,
        "aco_seed": 1234,
    },
}


def _short_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@pytest.mark.parametrize("task", sorted(FROZEN_TEXT_HASHES))
def test_frozen_task_description_and_template(task):
    template = TEMPLATES[task]
    frozen = FROZEN_TEXT_HASHES[task]
    actual = (_short_sha256(template.task_description), _short_sha256(template.template_program))
    assert actual == frozen, CONTRACT_MSG


def test_frozen_generated_data_seeds():
    assert (TRAIN_SEED, EVAL_SEED) == (2024, 2025), CONTRACT_MSG


@pytest.mark.parametrize("task", sorted(FROZEN_GENERATED_KWARGS))
@pytest.mark.parametrize("split", ["train", "eval"])
def test_frozen_generated_task_kwargs(task, split):
    assert get_generated_task_kwargs(task, split) == FROZEN_GENERATED_KWARGS[task][split], CONTRACT_MSG


@pytest.mark.parametrize("task", sorted(FROZEN_ACO_SPLITS))
def test_frozen_aco_split_specs(task):
    specs = cvrp_dataset.SPLIT_SPECS if task == "cvrp_aco" else op_dataset.SPLIT_SPECS
    for split, expected in FROZEN_ACO_SPLITS[task].items():
        spec = specs[split]
        actual = (spec.role, spec.problem_size, spec.n_instances, spec.seed)
        assert actual == expected, f"{task}/{split}: {CONTRACT_MSG}"


def test_frozen_op_max_len_budgets():
    assert {n: op_dataset.get_max_len(n) for n in (50, 100, 200)} == {50: 3.0, 100: 4.0, 200: 5.0}, CONTRACT_MSG


def test_frozen_aco_search_params_in_build_task():
    from experiments.infra.base import build_task

    for task, frozen in FROZEN_ACO_SEARCH_PARAMS.items():
        _, kwargs = build_task(task, eval_workers=None)
        for key, value in frozen.items():
            assert kwargs[key] == value, f"{task}.{key}: {CONTRACT_MSG}"


def test_frozen_search_seed_convention():
    from experiments.infra.base import TASKS, build_launch_plan
    from experiments.traceaad_v10_11.launch import build_plan

    args = argparse.Namespace(batch="contract_check", repeats=3, session_prefix="check")
    plan = build_launch_plan(args, module="m", method="check")
    assert [i.seed for i in plan] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], CONTRACT_MSG
    assert {i.task for i in plan} == set(TASKS)

    rows = build_plan("contract_check", "check")
    assert [r["seed"] for r in rows] == [0] * 5 + [1] * 5 + [2] * 5, CONTRACT_MSG
