from tests.experiments.test_training_monitor import append_records, candidate, write_json

import numpy as np

from experiments.infra.artifacts import pick_best_sample
from benchmarks.tasks import obp_scale
from benchmarks.generated_data_config import (
    get_generated_task_kwargs,
)
from benchmarks.online_bin_packing import OBPEvaluation
from benchmarks.online_bin_packing.template import task_description
from benchmarks.online_bin_packing.generate_weibull_instances import (
    generate_weibull_multiscale_dataset,
)


EXPECTED_TRAIN_SPECS = [
    {"n_instances": 1, "n_items": 1000, "capacities": [100, 500]},
    {"n_instances": 1, "n_items": 5000, "capacities": [100, 500]},
]

EXPECTED_EVAL_SPECS = [
    {"n_instances": 5, "n_items": 1000, "capacities": [100, 500]},
    {"n_instances": 5, "n_items": 5000, "capacities": [100, 500]},
    {"n_instances": 5, "n_items": 10000, "capacities": [100, 500]},
]


def test_obp_protocol_exposes_fixed_multicapacity_training_set():
    train_kwargs = get_generated_task_kwargs("online_bin_packing", "train")

    assert train_kwargs["seed"] == 2024
    assert train_kwargs["dataset_specs"] == EXPECTED_TRAIN_SPECS

    evaluator = OBPEvaluation(**train_kwargs)
    assert evaluator.dataset_metadata == (
        ("1k_100_instance_0", 1000, 100),
        ("1k_500_instance_0", 1000, 500),
        ("5k_100_instance_0", 5000, 100),
        ("5k_500_instance_0", 5000, 500),
    )


def test_obp_prompt_exposes_integer_dtype_and_float_output_contract():
    assert "integer dtype" in task_description
    assert "finite floating-point array" in task_description
    assert "cast integer-derived arrays" in task_description


def test_obp_protocol_exposes_separate_fixed_multiscale_test_set():
    train_kwargs = get_generated_task_kwargs("online_bin_packing", "train")
    eval_kwargs = get_generated_task_kwargs("online_bin_packing", "eval")

    assert eval_kwargs["seed"] == 2025
    assert eval_kwargs["dataset_specs"] == EXPECTED_EVAL_SPECS
    assert train_kwargs["seed"] != eval_kwargs["seed"]

    first = OBPEvaluation(**eval_kwargs)
    second = OBPEvaluation(**eval_kwargs)
    assert first.dataset_metadata == second.dataset_metadata
    assert len(first.dataset_metadata) == 30


def test_obp_test_evaluation_selects_the_fixed_held_out_scale():
    eval_kwargs = get_generated_task_kwargs("online_bin_packing", "eval")

    selected = obp_scale(eval_kwargs, n_items=10000, capacity=500)

    assert selected["seed"] == 2025
    assert selected["dataset_specs"] == [
        {"n_instances": 5, "n_items": 10000, "capacities": [500]}
    ]


def test_obp_scale_selection_preserves_the_canonical_fixed_instances():
    full = generate_weibull_multiscale_dataset(EXPECTED_EVAL_SPECS, seed=2025)
    selected = generate_weibull_multiscale_dataset(
        [{"n_instances": 5, "n_items": 10000, "capacities": [500]}],
        seed=2025,
    )
    train = generate_weibull_multiscale_dataset(EXPECTED_TRAIN_SPECS, seed=2024)

    assert np.array_equal(
        full["10k_500_instance_0"]["items"],
        selected["10k_500_instance_0"]["items"],
    )
    assert not np.array_equal(
        train["1k_100_instance_0"]["items"],
        full["1k_100_instance_0"]["items"],
    )


def test_obp_best_sample_can_be_truncated_at_the_formal_budget(tmp_path):
    run_dir = tmp_path / "eoh_run"
    write_json(run_dir / "summary.json", {"status": "finished"})
    append_records(run_dir / "events.jsonl", [candidate(1, -10), candidate(1001, -1)])
    best, records = pick_best_sample(run_dir, max_sample_order=1000)
    assert len(records) == 1
    assert best["sample_order"] == 1


def test_best_program_summary_supports_local_traceaad_layout(tmp_path):
    run_dir = tmp_path / "traceaad_local_run"
    write_json(run_dir / "summary.json", {"status": "finished"})
    append_records(run_dir / "events.jsonl", [candidate(42, -7, code="def f():\n    return 1\n")])
    best, records = pick_best_sample(run_dir, max_sample_order=1000)
    assert len(records) == 1
    assert best["sample_order"] == 42
    assert best["program"].startswith("def f")
