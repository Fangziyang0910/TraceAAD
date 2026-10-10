import pytest
import json

from experiments.decision_model.evaluate_policy import ACTIONS, aggregate, fit_baseline, greedy_weights, matched_outcomes, policy_delivery
from experiments.decision_model import evaluate_policy
from experiments.decision_model.search import check_model_conditions
from experiments.decision_model.collect import prompt_identity
from traceaad.common.config import REVISION


def state(action, task="task", attempts=0):
    return {"task": task, "proposed_action": action, "decision_phase": "ordinary", "attempts_used": attempts,
            "parent": {"fitness": 1, "previous_tries": 0}, "frontier_score_before": 1, "score_scale": 1}


def test_candidate_cost_can_change_the_preferred_action():
    gains = dict(zip(ACTIONS, (0.03, 0.04, 0.01)))
    costs = dict(zip(ACTIONS, (1, 2, 1)))
    weights = greedy_weights({a: gains[a] / costs[a] for a in ACTIONS})
    assert weights["Refine"] == 1
    assert policy_delivery(weights, gains, costs) == {"gain": 0.03, "candidate_cost": 1}


def test_zero_signal_has_no_artificial_winner():
    assert greedy_weights(dict.fromkeys(ACTIONS, 0)) == dict.fromkeys(ACTIONS, 1 / 3)
    with pytest.raises(ValueError):
        greedy_weights(dict.fromkeys(ACTIONS, float("nan")))


def test_state_rule_falls_back_without_seeing_held_out_outcomes():
    rows = [{"state": state("Explore"), "measurement": {"normalized_gains": [0, 0.02], "candidate_costs": [2, 1]}}]
    estimate = fit_baseline(rows)
    assert estimate(state("Explore", attempts=500)) == pytest.approx((0.01, 1.5))
    assert estimate(state("Crossover", task="unseen")) == pytest.approx((0.01, 1.5))


def test_aggregation_divides_total_gain_by_total_cost_before_task_average():
    decisions = []
    for task, gain, cost in [("a", 0.02, 1), ("a", 0.02, 2), ("b", 0, 1)]:
        policies = {name: {"gain": gain, "candidate_cost": cost} for name in ("model", "fixed", "constant", "state_rule")}
        decisions.append({"task": task, "run_id": task, "policies": policies})
    report = aggregate(decisions)
    assert report["equal_task_macro_gain_per_candidate"]["model"] == pytest.approx(0.04 / 3 / 2)
    assert not report["offline_improvement"]


def test_matched_trials_require_corresponding_gain_and_cost_measurements():
    comparison = {"actions": {a: {"2": [0, 0.02]} for a in ACTIONS},
                  "candidate_costs": {a: {"2": [1, 2]} for a in ACTIONS}}
    gains, costs = matched_outcomes(comparison, 2)
    assert gains == dict.fromkeys(ACTIONS, 0.01) and costs == dict.fromkeys(ACTIONS, 1.5)
    comparison["candidate_costs"]["Explore"]["2"] = [1]
    with pytest.raises(ValueError):
        matched_outcomes(comparison, 2)


def test_version_name_alone_cannot_authorize_a_changed_generation_prompt():
    metadata = {"prompt_policy": "v1023", "revision": REVISION, "prompt_sources_sha256": prompt_identity("v1023")}
    check_model_conditions(metadata)
    with pytest.raises(ValueError, match="prompt sources differ"):
        check_model_conditions({**metadata, "prompt_sources_sha256": {"changed": "same version name"}})
    with pytest.raises(ValueError, match="evaluation conditions differ"):
        check_model_conditions({**metadata, "revision": "different evaluator"})


def test_evaluation_rejects_changed_prompt_sources_before_loading_model(tmp_path, monkeypatch):
    metadata = {"decision_horizon": 2, "revision": REVISION, "prompt_policy": "v1023",
                "prompt_sources_sha256": {"prompt": "original"}, "score_scales": {},
                "trained_questions": ["frontier_gain", "repair_used"]}
    paired, training = tmp_path / "paired", tmp_path / "training"
    paired.mkdir()
    training.mkdir()
    (paired / "metadata.json").write_text(json.dumps(metadata))
    (training / "metadata.json").write_text(json.dumps({**metadata, "prompt_sources_sha256": {"prompt": "changed"}}))
    (paired / "calibration.jsonl").touch()
    (paired / "comparisons.jsonl").touch()
    monkeypatch.setattr("sys.argv", ["evaluate_policy", "unused-model", str(paired), str(tmp_path / "report.json"),
                                    "--training-data", str(training)])
    with pytest.raises(ValueError, match="prompt_sources_sha256"):
        evaluate_policy.main()
