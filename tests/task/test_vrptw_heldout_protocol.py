from benchmarks.tasks import SCALES, heldout_task, split_of_scale
from experiments.infra.evaluate import parse_units


def test_vrptw_heldout_defaults_cover_same_and_larger_scales() -> None:
    assert SCALES["vrptw_construct"] == (50, 100, 200)
    assert split_of_scale("vrptw_construct", 100) == "eval_100"


def test_vrptw_units_parse_positive_problem_sizes() -> None:
    assert parse_units("vrptw_construct", "50,100,200") == ["eval_50", "eval_100", "eval_200"]


def test_vrptw_eval_kwargs_hold_test_split_fixed_across_scales() -> None:
    evaluation = heldout_task("vrptw_construct", "eval_200", timeout_seconds=120)
    assert evaluation.timeout_seconds == 120
    assert evaluation.problem_size == 200
    assert evaluation.n_instance == 16
    assert evaluation.seed == 2025
