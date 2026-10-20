"""Keep the baseline decisions that matter when comparing research methods."""

from unittest.mock import patch

import numpy as np
import pytest

from baselines.eoh.population import Population
from baselines.mcts_ahd.mcts import MCTS, MCTSNode
from baselines.mcts_ahd.mcts_ahd import MCTS_AHD
from baselines.reevo.prompt import ReEvoPrompt
from baselines.reevo.reevo import ReEvo
from baselines.shinka_evo import ShinkaArchive, ShinkaProgram
from core import Function
from tests.support import TinyEvaluation, TokenLLM


def function(value, score):
    item = Function(name="score", args="x", body=f"    return {value}")
    item.score, item.algorithm = score, f"return {value}"
    return item


def test_eoh_parent_sampling_uses_rank_weights():
    population = Population(pop_size=3)
    for value in (3, 2, 1):
        population.register_function(function(value, value), increment_generation=False)
    with patch("baselines.eoh.population.random.choices") as choose:
        choose.return_value = [population.population[0]]
        population.selection()
    expected = np.array([1 / 4, 1 / 5, 1 / 6])
    np.testing.assert_allclose(choose.call_args.kwargs["weights"], expected)


def test_reevo_reflection_orders_actual_worse_and_better_programs():
    worse, better = function(1, 3), function(2, 1)
    messages = ReEvoPrompt.get_short_term_reflection_prompt("minimize", [better, worse])
    text = messages[-1]["content"]
    assert text.index("return 1") < text.index("return 2")
    with pytest.raises(ValueError):
        ReEvoPrompt.get_short_term_reflection_prompt("minimize", [better, better])


def test_reevo_pairs_tied_parents_only_when_every_score_is_tied():
    method = ReEvo(TokenLLM(), TinyEvaluation(), pop_size=4, init_pop_size=2, max_sample_nums=10)
    try:
        method._population._population = [function(1, 2), function(2, 2), function(3, 2)]
        pairs = method._select_parent_pairs()
        assert method._selection_tied and len(pairs) == 4
        assert all(a is not b and a.score == b.score for a, b in pairs)
        for prompt in (ReEvoPrompt.get_short_term_reflection_prompt("minimize", pairs[0], True),
                       ReEvoPrompt.get_crossover_prompt("minimize", "", pairs[0], True)):
            text = prompt[-1]["content"]
            assert "[Code 1]" in text and "[Code 2]" in text
            assert "orse" not in text and "better than" not in text

        method._population._population = [function(1, 1), function(2, 2), function(3, 2)]
        pairs = method._select_parent_pairs()
        assert not method._selection_tied
        assert all(a.score != b.score for a, b in pairs)
    finally:
        method._evaluation_executor.shutdown()


def test_mcts_keeps_a_weak_child_in_the_tree_even_if_the_population_rejects_it():
    method = MCTS_AHD(TokenLLM(), TinyEvaluation(), pop_size=1, max_sample_nums=2)
    try:
        strongest, parent_function, child = function(10, -10), function(1, -1), function(0, 0)
        method._population._population = [strongest]
        tree = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = MCTSNode(
            "parent",
            str(parent_function),
            -1,
            individual=parent_function,
            parent=tree.root,
            depth=1,
            visit=1,
            Q=1,
            raw_info=parent_function,
        )
        tree.root.add_child(parent)
        parent.subtree.append(parent)
        method._sample_evaluate_register = lambda *args, **kwargs: child
        method.expand(tree, [], parent, "m1")
        assert parent.children[-1].individual is child
        assert method._population.population == [strongest]
    finally:
        method._evaluation_executor.shutdown()


def test_shinka_parent_weight_combines_quality_with_children_count():
    archive = ShinkaArchive(parent_selection_lambda=1)
    programs = []
    for value in (1, 2, 3):
        f = function(value, -value)
        p = ShinkaProgram.create(
            f, str(f), combined_score=-value, correct=True, program_id=str(value), island_idx=0
        )
        archive.add_program(p)
        programs.append(p)
    before = {p.id: probability for p, probability in archive.weighted_parent_probabilities(0)}
    assert before["3"] > before["2"] > before["1"]
    programs[2].children_count = 10
    after = {p.id: probability for p, probability in archive.weighted_parent_probabilities(0)}
    assert after["3"] < after["2"]


def test_baseline_freezes_the_training_winner_without_extra_selection(tmp_path):
    from baselines.shared.profiler import ProfilerBase
    from experiments.infra.result_storage import selected_program
    from traceaad.storage import read_json, write_json

    task = TinyEvaluation()
    write_json(tmp_path / "run_config.json", {"task": "toy", "method": "eoh", "budget": 2})
    profiler = ProfilerBase(tmp_path)
    profiler.record_parameters(TokenLLM(), task, None)
    for value in (1, 2):
        candidate = function(value, -value)
        profiler.register_function(candidate, str(candidate))
    profiler.finish()
    summary = read_json(tmp_path / "summary.json")
    assert summary["status"] == "finished" and summary["selection_evaluations"] == 0
    winner = selected_program(tmp_path)
    assert winner["verified"] and winner["id"] == 2 and winner["fitness"] == -2
    assert read_json(tmp_path / "selection.json")["finalists"] == []
    assert not (tmp_path / "evaluations.jsonl").exists()
