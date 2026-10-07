"""Independent feasibility oracles and the search/selection/held-out path."""

import hashlib
import itertools
import json

import numpy as np
import pytest

from benchmarks.ahd_suite import AHDEvaluation, fssp, graph, mdmkp, set_cover
from benchmarks.ahd_suite.dataset import PROTOCOL, TASKS, load, manifest, records
from benchmarks.ahd_suite.templates import FUNCTION_NAMES, TEMPLATES
from core import SecureEvaluator
from core.evaluate import InvalidEvaluationResult


def seed(task):
    namespace = {}
    exec(TEMPLATES[task], namespace)
    return namespace[FUNCTION_NAMES[task]]


def test_flowshop_recurrence_and_true_objective_after_perturbation():
    times = np.array([[2, 3], [4, 1], [1, 5]], dtype=float)

    def oracle(order):
        completion = np.zeros((4, 3))
        for i, job in enumerate(order, 1):
            for machine in range(1, 3):
                completion[i, machine] = max(completion[i-1, machine], completion[i, machine-1]) + times[job, machine-1]
        return completion[-1, -1]

    for order in itertools.permutations(range(3)):
        assert fssp.makespan(np.array(order), times) == oracle(order)
    result, cost = fssp.solve({"processing_times": times},
                            lambda seq, t, m, n: (np.zeros_like(t), np.array([seq[-1]])), iterations=2)
    assert sorted(result) == [0, 1, 2]
    assert cost == oracle(result) and cost > 0


@pytest.mark.parametrize("output", [None, (np.ones((3, 2)), np.array([])),
                                   (np.ones((3, 2)), np.array([0, 0])),
                                   (np.ones((2, 3)), np.array([0])),
                                   (np.full((3, 2), np.nan), np.array([0]))])
def test_flowshop_rejects_invalid_perturbations(output):
    with pytest.raises(InvalidEvaluationResult):
        fssp.solve({"processing_times": np.ones((3, 2))}, lambda *args: output, iterations=1)


def test_mdmkp_moves_agree_with_binary_feasibility_oracle():
    data = {"A_leq": np.array([[2, 3, 1, 4], [1, 2, 3, 2]]), "b_leq": np.array([6, 5]),
            "A_geq": np.array([[1, 1, 2, 1]]), "b_geq": np.array([2]),
            "cost_vector": np.array([5, -2, 7, 4])}
    for bits in itertools.product((0, 1), repeat=4):
        x = np.array(bits, dtype=np.int8)
        if not mdmkp.feasible(data, x):
            continue
        expected = set()
        for remove in [-1, *np.flatnonzero(x)]:
            for add in [-1, *np.flatnonzero(1-x)]:
                if remove == add == -1:
                    continue
                trial = x.copy()
                if remove >= 0: trial[remove] = 0
                if add >= 0: trial[add] = 1
                if mdmkp.feasible(data, trial): expected.add((remove, add))
        assert set(map(tuple, mdmkp.eligible_moves(data, x))) == expected


def test_mdmkp_retains_best_feasible_solution_after_a_worse_move():
    data = {"A_leq": np.array([[1, 1]]), "b_leq": np.array([2]),
            "A_geq": np.array([[1, 1]]), "b_geq": np.array([1]),
            "cost_vector": np.array([8, -2]), "initial_solution": np.array([1, 0], dtype=np.int8)}
    x, value = mdmkp.solve(data, lambda profits, a, b, g, d, x, moves: -seed("mdmkp_search")(profits, a, b, g, d, x, moves), steps=1)
    assert value == 8 and x.tolist() == [1, 0]


@pytest.mark.parametrize("adjacency,optimal", [(np.ones((3, 3), dtype=bool) ^ np.eye(3, dtype=bool), 3),
    (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool), 2)])
def test_graph_final_solution_survives_failed_color_reduction(adjacency, optimal):
    colors, k = graph.solve({"adjacency": adjacency}, seed("graph_colouring"), repair_steps=4)
    assert graph.valid(adjacency, colors) and k == optimal


def test_covering_solution_and_cost_against_enumerated_optimum():
    coverage = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
    costs = np.array([2., 2., 3.])
    feasible_costs = [costs[np.array(bits, dtype=bool)].sum() for bits in itertools.product((0, 1), repeat=3)
                      if coverage[:, np.array(bits, dtype=bool)].any(axis=1).all()]
    chosen, value = set_cover.solve({"coverage": coverage, "costs": costs}, seed("set_cover_construct"))
    assert coverage[:, chosen].any(axis=1).all()
    assert value == costs[chosen].sum() == min(feasible_costs)


def test_candidate_mutations_do_not_change_covering_solver_inputs():
    data = {"coverage": np.array([[1, 0], [0, 1]], dtype=bool), "costs": np.array([2., 3.])}
    original = {key: value.copy() for key, value in data.items()}

    def hostile(costs, coverage, selected, uncovered):
        value = np.ones(len(costs))
        coverage[:] = False
        costs[:] = 0
        selected[:] = True
        uncovered[:] = False
        return value

    _, value = set_cover.solve(data, hostile)
    assert value == 5
    assert all(np.array_equal(data[k], v) for k, v in original.items())


def miniature_data(root):
    data = {"coverage": np.array([[1, 0], [0, 1]], dtype=bool), "costs": np.array([2., 3.])}
    rows = []
    for split in ("train", "val", "test"):
        path = root / f"{split}.npz"
        np.savez_compressed(path, **data)
        rows.append({"id": split, "group": split, "split": split, "scale": 2,
                     "dimensions": {"elements": 2, "sets": 2}, "content": split,
                     "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "reference": 6., "reference_kind": "test reference"})
    (root / "manifest.json").write_text(json.dumps({"protocol": PROTOCOL, "tasks": {"set_cover_construct": rows}}))


def test_eval_isolates_module_state_and_measures_calls(tmp_path):
    from traceaad.common.evaluation import ProgramEvaluator
    miniature_data(tmp_path)
    evaluation = AHDEvaluation("set_cover_construct", data_root=tmp_path, safe_evaluate=False)
    evaluation._rows *= 2
    evaluation._instances *= 2
    code = ('import numpy as np\n_calls = 0\n'
            'def score_sets(costs, coverage, selected, uncovered):\n'
            '    global _calls\n    _calls += 1\n    assert _calls <= 4\n'
            '    return np.ones(len(costs))\n')
    result = ProgramEvaluator(evaluation, (123,)).evaluate(code, "isolation")
    assert result["failure"] is None
    assert result["calls"] == 8
    assert result["fitness"] == pytest.approx(-100/6)


def test_invalid_scores_are_reported_as_invalid_output(tmp_path):
    from traceaad.common.evaluation import ProgramEvaluator
    miniature_data(tmp_path)
    evaluation = AHDEvaluation("set_cover_construct", data_root=tmp_path, safe_evaluate=False)
    result = ProgramEvaluator(evaluation).evaluate('def score_sets(*args):\n    return [float("nan")]\n', "bad")
    assert result["failure"]["kind"] == "invalid_output"


def test_portable_data_hash_is_checked(tmp_path):
    miniature_data(tmp_path)
    row = records("set_cover_construct", "train", tmp_path)[0]
    (tmp_path / row["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load(row, tmp_path)


@pytest.mark.parametrize("task", TASKS)
def test_packaged_template_runs_through_secure_evaluation(task):
    evaluation = AHDEvaluation(task, limit=1, safe_evaluate=False)
    outcome = SecureEvaluator(evaluation).evaluate_program_with_details(evaluation.template_program)
    assert outcome.failure_kind is None, outcome.error
    assert np.isfinite(outcome.result)


def test_packaged_splits_have_no_base_group_or_content_overlap():
    for task, rows in manifest()["tasks"].items():
        for field in ("group", "content"):
            groups = {split: {r[field] for r in rows if r["split"] == split} for split in ("train", "val", "test")}
            assert all(groups[a].isdisjoint(groups[b]) for a, b in itertools.combinations(groups, 2)), task
        assert len({json.dumps(r["dimensions"], sort_keys=True) for r in rows}) == 1


def test_offline_search_selection_and_heldout_use_the_new_task(tmp_path):
    from experiments.infra.evaluate import evaluate_run
    from tests.support import TokenLLM, text_candidate
    from traceaad.v10_20 import Config, TraceAADV1020
    from traceaad.common.storage import write_json
    data = tmp_path / "data"
    data.mkdir()
    miniature_data(data)
    evaluation = AHDEvaluation("set_cover_construct", data_root=data, safe_evaluate=False)
    selection = AHDEvaluation("set_cover_construct", "val", data_root=data, safe_evaluate=False)
    run = tmp_path / "run"
    llm = TokenLLM(text_candidate(code=evaluation.template_program))
    config = Config(budget=1, roots=1, final_candidates=1)
    method = TraceAADV1020(evaluation=evaluation, selection_evaluation=selection,
                           llm=llm, run_dir=run, task="set_cover_construct", config=config)
    method.run()
    assert len(llm.calls) == 1
    write_json(run / "run_config.json", {"task": "set_cover_construct", "method_params": {"evaluation_seeds": [730241]}})
    # Real task data are used for held-out; a frozen valid heuristic must transfer.
    results = evaluate_run(run, ["test_2000"], condition="traceaad")
    assert results[0]["fitness"] is not None and not results[0]["failures"]
