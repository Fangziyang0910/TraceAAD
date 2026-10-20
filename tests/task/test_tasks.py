"""Benchmark data, solution validity and true objective values."""

import re

import numpy as np
import pytest

from benchmarks.fssp_gls.evaluation import makespan
from benchmarks.fssp_gls.evaluation import solve as flow_shop
from benchmarks.graph_colouring.evaluation import solve as color
from benchmarks.jssp_construct.evaluation import solve as schedule
from benchmarks.jssp_construct.evaluation import validate_schedule
from benchmarks.shared.seeded_data import generate_dataset
from benchmarks.tasks import CLASSES, CO_TASKS, FIXED_TASKS, training_task
from benchmarks.tsp_construct.evaluation import TSPEvaluation
from core.evaluate import InvalidEvaluationResult


@pytest.mark.parametrize("task", FIXED_TASKS)
def test_data_are_reproducible_and_training_is_separate_from_test(task):
    dataset = CLASSES[task].DATASET
    metadata, train = generate_dataset(dataset, "train", limit=2)
    repeated_metadata, repeated = generate_dataset(dataset, "train", limit=2)
    test_metadata, test = generate_dataset(dataset, "test", limit=2)
    assert metadata == repeated_metadata
    assert {r["group"] for r in metadata}.isdisjoint(r["group"] for r in test_metadata)
    for original, regenerated in zip(train, repeated):
        for key in original:
            np.testing.assert_array_equal(original[key], regenerated[key])
    assert not (
        {tuple(v.tobytes() for v in x.values()) for x in train}
        & {tuple(v.tobytes() for v in x.values()) for x in test}
    )


def test_job_shop_validates_the_returned_schedule_and_scores_original_inputs():
    data = {
        "processing_times": np.array([[3, 2], [2, 4]]),
        "machine_order": np.array([[0, 1], [1, 0]]),
    }
    feasible = np.array([[0, 3], [0, 3]])
    assert schedule(data, lambda t, m: feasible)[1] == 7
    assert schedule(data, lambda t, m: feasible + 10)[1] == 17

    def mutate(t, m):
        t[:] = 0
        m[:] = 0
        return feasible

    assert schedule(data, mutate)[1] == 7
    np.testing.assert_array_equal(data["processing_times"], [[3, 2], [2, 4]])
    with pytest.raises(InvalidEvaluationResult):
        validate_schedule(data["processing_times"], data["machine_order"], np.zeros((2, 2)))


def test_graph_coloring_checks_edges_even_if_the_program_modifies_its_input():
    graph = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    _, objective = color({"adjacency": graph}, lambda adjacency: np.array([10, 99, 10]))
    assert objective == 2

    def cheat(adjacency):
        adjacency[:] = 0
        return np.zeros(3, dtype=int)

    with pytest.raises(InvalidEvaluationResult):
        color({"adjacency": graph}, cheat)
    assert graph[0, 1] == 1


def test_flow_shop_makespan_uses_completion_recurrence():
    processing = np.array([[1, 3], [2, 1]])
    assert makespan(np.array([0, 1]), processing) == 5
    assert makespan(np.array([1, 0]), processing) == 6
    order, cost = flow_shop(
        {"processing_times": processing},
        lambda seq, t, m, n: (np.zeros_like(t), np.array([seq[-1]])),
        iterations=2,
    )
    assert sorted(order) == [0, 1]
    assert cost == makespan(order, processing) and cost > 0


@pytest.mark.parametrize("task", FIXED_TASKS)
def test_objective_does_not_use_reference_scores(task):
    evaluation = CLASSES[task](limit=1, safe_evaluate=False)
    evaluation.outer_settings = {name: 1 for name in evaluation.outer_settings}
    before = evaluation.evaluate_program(evaluation.template_program, None)
    for row in evaluation._rows:
        row["reference"] *= 100
    assert evaluation.evaluate_program(evaluation.template_program, None) == before


def test_tsp_checks_visited_nodes_even_if_the_program_changes_the_offered_nodes():
    evaluation = TSPEvaluation(n_instance=1, problem_size=4)

    def repeat_start(current, destination, unvisited, distances):
        unvisited[:] = 0
        return 0

    assert evaluation.evaluate(repeat_start) is None


def test_tsp_scores_coordinates_even_if_the_program_changes_the_distance_matrix():
    evaluation = TSPEvaluation(n_instance=1, problem_size=4)
    expected = evaluation.evaluate(lambda current, destination, unvisited, distances: unvisited[0])

    def mutate(current, destination, unvisited, distances):
        distances[:] = 0
        return unvisited[0]

    assert expected > 0
    assert evaluation.evaluate(mutate) == expected


@pytest.mark.parametrize("task", ["cvrp_aco", "op_aco"])
def test_aco_parallel_execution_preserves_instance_seeds_and_score(task):
    evaluation = CLASSES[task](n_ants=3, n_iterations=2, n_workers=1)
    evaluation._datasets = evaluation._datasets[:2]
    evaluation.n_instance = 2
    scope = {}
    exec(str(evaluation.template_program), scope)
    heuristic = scope["heuristics"]
    serial = evaluation.evaluate_program(str(evaluation.template_program), heuristic)
    evaluation.n_workers = 2
    assert np.isfinite(serial)
    assert evaluation.evaluate_program(str(evaluation.template_program), heuristic) == serial


@pytest.mark.parametrize("task", CO_TASKS)
def test_generation_text_gives_no_instance_count_or_size(task):
    # PROTOCOL.md: instance counts, sizes, splits, seeds and evaluation accounting stay out of generation prompts.
    evaluation = training_task(task)[0]
    text = evaluation.task_description + evaluation.template_program
    values = {evaluation.n_instance, evaluation.problem_size}
    dataset = getattr(evaluation, "DATASET", None)
    if dataset is not None:
        values |= set(dataset.COUNTS.values()) | set(dataset.DIMENSIONS.values())
    assert not [v for v in values if re.search(rf"\b{v}\b", text)]
    assert not re.search(r"\d+ (fixed |generated )*(train|test|evaluation )?instances", text, re.I)
    # Nor the evaluator, time limits or the fixed framework's settings.
    assert not re.search(r"time limit|\bCPU\b|\bseconds?\b|evaluator|settings|reference|local_passes", text, re.I)
