import math

import pytest

from experiments.decision_model.data import delivered_request, expected_gain, expected_request_cost, fit_record, gain_level, outcome, questions, request_state
from experiments.decision_model.collect import Collector, METHODS, develop_block, prompt_identity, save_manifest
from experiments.decision_model.prepare import choose_states, compatible, reward_support
from experiments.decision_model.augment_data import replace_observations
from experiments.decision_model.export_replays import empirical_gold


def node(index, score, code, parent=None):
    return {"id": index, "fitness": score, "score": score, "code": code, "valid": True,
            "parent_id": parent, "action": "Refine" if parent else "Init", "depth": int(parent is not None)}


def test_a_weaker_parent_improvement_is_not_a_frontier_gain():
    labels, reward = outcome(10, 8, [9])
    assert labels == {"new_valid": True, "improve_parent": True, "frontier_gain": 0}
    assert reward == 0


def test_failed_or_duplicate_candidates_have_no_reward():
    assert outcome(-10, -11, [])[0] == {"new_valid": False, "improve_parent": False, "frontier_gain": 0}


def test_signed_scores_use_the_same_minimized_objective():
    labels, reward = outcome(-10, -11, [-11.22], score_scale=11)
    assert labels["improve_parent"] and reward == pytest.approx(0.02)
    assert labels["frontier_gain"] in (3, 4)  # floating arithmetic at the bin boundary


def test_request_cannot_see_future_program_or_attempt():
    programs = {1: node(1, 10, "old code"), 2: node(2, 8, "reference"),
                4: node(4, 1, "secret future code", parent=1)}
    attempts = {4: {"id": 4, "action": "Refine", "parent_id": 1, "program_id": 4,
                    "status": "valid", "repair_of": None}}
    state = request_state("task", "description", "interface", programs, attempts, 1, 2,
                          "Crossover", 2, 1000)
    assert state["frontier_score_before"] == 8
    assert state["recent_attempts"] == []
    assert state["parent"]["previous_tries"] == 0
    assert "secret future code" not in str(state)


def test_gain_probabilities_are_required_to_be_a_distribution():
    support = [0, 0.001, 0.005, 0.01, 0.03]
    assert expected_gain({"probabilities": {str(i): float(i == 2) for i in range(5)}}, support) == 0.005
    with pytest.raises(ValueError, match="invalid gain probabilities"):
        expected_gain({"probabilities": {str(i): 0.3 for i in range(5)}}, support)
    with pytest.raises(ValueError):
        gain_level(math.nan)


def test_sampling_does_not_use_the_later_outcome():
    import random
    rows = [{"state_id": str(i), "state": {"decision_phase": "ordinary", "attempts_used": i,
             "frontier_score_before": 1, "parent": {"code": str(i), "fitness": 1}},
             "gold": {"frontier_gain": i % 5}} for i in range(20)]
    selected = choose_states(rows, 8, random.Random(5))
    for row in rows:
        row["gold"]["frontier_gain"] = 4
    again = choose_states(rows, 8, random.Random(5))
    assert [r["state_id"] for r in selected] == [r["state_id"] for r in again]


def test_fitting_history_never_truncates_programs():
    original = {"state": {"parent": {"code": "complete-parent"}, "reference": {"code": "complete-reference"},
                           "formation": ["old history"], "recent_attempts": ["recent history"]}}
    length = lambda r: 40 + sum(map(len, r["state"]["formation"] + r["state"]["recent_attempts"]))
    fitted = fit_record(original, length, 41)
    assert fitted["state"]["parent"] == original["state"]["parent"]
    assert fitted["state"]["reference"] == original["state"]["reference"]
    assert original["state"]["formation"] == ["old history"]
    with pytest.raises(ValueError, match="complete parent/reference"):
        fit_record(original, length, 40)


def test_repair_consumes_budget_and_weak_new_program_gets_developed():
    from types import SimpleNamespace
    original = node(5, 1, "original")
    calls = []
    class Method:
        attempts = 5
        config = SimpleNamespace(budget=9)
        progress = SimpleNamespace(repair_id=None)
        programs = {5: original}
        archive = programs
        prompts = SimpleNamespace(build=lambda action, parent, **kw: {"action": action})
        def _attempt(self, request, parent, reference=None):
            calls.append((request["action"], parent["id"]))
            self.attempts += 2 if self.attempts == 5 else 1
            self.archive[self.attempts] = node(self.attempts, 2 - 0.1 * (self.attempts - 7), "new", parent=parent["id"])
    method = Method()
    develop_block(method, 5, None, "Explore", 5)
    assert method.attempts == 9
    assert calls == [("Explore", 5), ("Refine", 7), ("Refine", 8)]


def test_prompt_policy_includes_inherited_repair_material():
    old, current = prompt_identity("v1022"), prompt_identity("v1023")
    assert set(old) < set(current)
    assert {name: current[name] for name in old} == old
    assert "traceaad.v10_23.prompts" in current
    assert METHODS["v1023"].PromptBuilder.ANALYSIS["Explore"] != METHODS["v1022"].PromptBuilder.ANALYSIS["Explore"]


def test_decision_input_describes_the_actual_generation_policy():
    programs = {1: node(1, 10, "parent"), 2: node(2, 9, "reference")}
    old = request_state("task", "description", "interface", programs, {}, 1, 2, "Explore", 2, 1000)
    current = request_state("task", "description", "interface", programs, {}, 1, 2, "Explore", 2, 1000,
                            prompt_policy="v1023")
    assert current["generation_policy"] == "TraceAAD V10.23"
    assert current["action_instruction"] == METHODS["v1023"].PromptBuilder.EXPLORE
    assert current["analysis_instruction"] == METHODS["v1023"].PromptBuilder.ANALYSIS["Explore"]
    assert current["parent"] == old["parent"]
    assert current["action_instruction"] != old["action_instruction"]


def test_collection_cannot_resume_with_changed_prompts(tmp_path):
    path = tmp_path / "manifest.json"
    design = {"states_sha256": "fixed", "prompt_policy": "v1022",
              "prompt_sources_sha256": prompt_identity("v1022")}
    save_manifest(path, design)
    save_manifest(path, design)
    with pytest.raises(ValueError, match="different collection design"):
        save_manifest(path, {**design, "prompt_policy": "v1023"})
    with pytest.raises(ValueError, match="different collection design"):
        save_manifest(path, {**design, "prompt_sources_sha256": {"changed": "source"}})


def test_source_acceptance_requires_the_measured_prompt_policy():
    from traceaad.common.config import REVISION
    config = {"revision": REVISION, "method": "v1023", "task": "tsp_construct", "objective": "min",
              "evaluation_execution": {"function_seconds": 2, "timeout_seconds": 20},
              "method_params": {"evaluation_seeds": [730241]}}
    assert compatible(config, "v1023")
    assert not compatible(config, "v1022")
    assert not compatible({**config, "revision": "different-evaluator"}, "v1023")


def test_request_includes_only_its_automatic_repair():
    first = {"id": 5, "action": "Explore", "status": "timeout", "program_id": 5}
    repair = {"id": 6, "action": "Repair", "repair_of": 5, "status": "valid", "program_id": 6}
    programs = {5: {"id": 5, "valid": False}, 6: {"id": 6, "valid": True, "fitness": 7}}
    delivery = delivered_request(first, repair, programs, 1000)
    assert delivery == {"fitnesses": [7], "candidate_cost": 2, "attempt_ids": [5, 6]}
    assert outcome(10, 8, delivery["fitnesses"])[1] == 1
    assert set(questions(2)) == {"frontier_gain", "repair_used"}


def test_unfinished_repair_is_not_a_zero_reward_label():
    first = {"id": 5, "action": "Explore", "status": "timeout", "program_id": 5}
    programs = {5: {"id": 5, "valid": False}}
    assert delivered_request(first, None, programs, 1000) is None
    assert delivered_request(first, None, programs, 5)["candidate_cost"] == 1


def test_a_later_ordinary_improvement_is_not_the_first_requests_return():
    first = {"id": 5, "action": "Refine", "status": "valid", "program_id": 5}
    following = {"id": 6, "action": "Refine", "status": "valid", "program_id": 6}
    programs = {5: {"id": 5, "valid": True, "fitness": 9},
                6: {"id": 6, "valid": True, "fitness": 1}}
    assert delivered_request(first, following, programs, 1000) == {
        "fitnesses": [9], "candidate_cost": 1, "attempt_ids": [5]}


def test_expected_cost_respects_the_actual_remaining_candidate_budget():
    answer = {"probabilities": {"true": 0.25, "false": 0.75}}
    assert expected_request_cost(answer, 1000) == 1.25
    assert expected_request_cost(answer, 1) == 1
    with pytest.raises(ValueError):
        expected_request_cost({"probabilities": {"true": 0.9, "false": 0.9}}, 1000)


def test_collection_cannot_grant_more_budget_than_the_frozen_state_has():
    collector = object.__new__(Collector)
    with pytest.raises(ValueError, match="fewer than four"):
        collector.run(({"cutoff": 999, "search_budget": 1000}, "Refine", 0))


def test_repeated_labels_preserve_uncertainty_and_actual_gains():
    trials = [{"gold": {"2": {"frontier_gain": g, "repair_used": repaired}}}
              for g, repaired in [(0, False), (3, True), (0, True), (0, False)]]
    gold = empirical_gold(trials, 2)
    assert gold["frontier_gain"]["probabilities"]["3"] == 0.25
    assert gold["repair_used"]["probabilities"]["true"] == 0.5
    support = reward_support([{"state": {"task": "task"}, "measurement": {"normalized_gains": [0, 0.012, 0, 0]}}])
    assert support["task"][0] == 0 and support["task"][3] == 0.012


def test_independent_repeats_replace_a_selected_historical_outcome():
    observed = [{"run_id": "train", "state_id": "before-2"}, {"run_id": "train", "state_id": "before-3"}]
    paired = [{"run_id": "train", "state_id": f"before-2/{action}/h2"} for action in ("Refine", "Explore", "Crossover")]
    assert replace_observations(observed, paired) == [observed[1]] + paired
    assert replace_observations(observed, [{"run_id": "other", "state_id": "before-2/Refine/h2"}])[:2] == observed
