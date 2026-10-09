"""Evaluate same-state action choices by observed gain per candidate consumed."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

if __package__:
    from .data import ACTIONS, expected_gain, expected_request_cost, questions
else:
    from data import ACTIONS, expected_gain, expected_request_cost, questions

FIXED = {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}


def state_key(state):
    parent = state["parent"]
    return (state["task"], state["proposed_action"], state["decision_phase"], state["attempts_used"] // 250,
            (parent["fitness"] - state["frontier_score_before"]) / state["score_scale"] <= 0.005,
            0 if parent["previous_tries"] == 0 else 1 if parent["previous_tries"] < 4 else 2)


def observation(row):
    measured = row["measurement"]
    if "normalized_gains" in measured:
        gains, costs = measured["normalized_gains"], measured["candidate_costs"]
        if not gains or len(gains) != len(costs):
            raise ValueError("mismatched gain and cost trials")
        if any(not math.isfinite(v) or v < 0 for v in gains) or any(not math.isfinite(v) or not 1 <= v <= 2 for v in costs):
            raise ValueError("invalid observed trials")
        return sum(gains) / len(gains), sum(costs) / len(costs)
    return measured["normalized_gain"], measured.get("candidate_cost", 1)


def fit_baseline(rows):
    groups, cells, observations = defaultdict(list), defaultdict(list), []
    for row in rows:
        value = observation(row)
        if not math.isfinite(value[0]) or value[0] < 0 or not math.isfinite(value[1]) or not 1 <= value[1] <= 2:
            raise ValueError("invalid observed request")
        observations.append(value)
        state = row["state"]
        groups[state["task"], state["proposed_action"]].append(value)
        cells[state_key(state)].append(value)
    if not observations:
        raise ValueError("empty baseline training data")
    prior = tuple(sum(v[i] for v in observations) / len(observations) for i in (0, 1))

    def estimate(state, use_state=True):
        values = groups[state["task"], state["proposed_action"]]
        group = tuple((sum(v[i] for v in values) + 5 * prior[i]) / (len(values) + 5) for i in (0, 1))
        values = cells[state_key(state)] if use_state else []
        return tuple((sum(v[i] for v in values) + 20 * group[i]) / (len(values) + 20) for i in (0, 1))

    return estimate


def greedy_weights(values):
    if set(values) != set(ACTIONS) or not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError("invalid action values")
    best = max(values.values())
    tied = [a for a in ACTIONS if values[a] == best]
    return {a: (1 / len(tied) if a in tied else 0) for a in ACTIONS}


def matched_outcomes(comparison, horizon):
    gains, costs = {}, {}
    for action in ACTIONS:
        gain = comparison["actions"][action][str(horizon)]
        cost = (comparison["candidate_costs"][action]["2"] if horizon == 2 else [1] * len(gain))
        if not gain or len(gain) != len(cost) or any(not math.isfinite(v) or v < 0 for v in gain):
            raise ValueError("invalid matched gains")
        if any(not math.isfinite(v) or not 1 <= v <= horizon for v in cost):
            raise ValueError("invalid matched candidate costs")
        gains[action], costs[action] = sum(gain) / len(gain), sum(cost) / len(cost)
    return gains, costs


def policy_delivery(weights, gains, costs):
    if set(weights) != set(ACTIONS) or any(not 0 <= v <= 1 for v in weights.values()) or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("invalid action mixture")
    return {"gain": sum(weights[a] * gains[a] for a in ACTIONS),
            "candidate_cost": sum(weights[a] * costs[a] for a in ACTIONS)}


def aggregate(decisions):
    if not decisions:
        raise ValueError("no matched held-out states")
    names = tuple(decisions[0]["policies"])

    def summarize(items):
        return {name: {"gain_per_candidate": sum(r["policies"][name]["gain"] for r in items) /
                sum(r["policies"][name]["candidate_cost"] for r in items),
                "mean_gain_per_request": sum(r["policies"][name]["gain"] for r in items) / len(items),
                "mean_candidate_cost": sum(r["policies"][name]["candidate_cost"] for r in items) / len(items)}
                for name in names}

    by_task = {task: summarize([r for r in decisions if r["task"] == task]) for task in sorted({r["task"] for r in decisions})}
    by_source = {run: summarize([r for r in decisions if r["run_id"] == run]) for run in sorted({r["run_id"] for r in decisions})}
    macro = {name: sum(v[name]["gain_per_candidate"] for v in by_task.values()) / len(by_task) for name in names}
    return {"equal_task_macro_gain_per_candidate": macro, "by_task": by_task, "by_source": by_source,
            "offline_improvement": macro["model"] > max(macro[name] for name in ("fixed", "constant", "state_rule"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-data", type=Path, help="actual training data for the competing statistical rules")
    parser.add_argument("--split", choices=("calibration", "test"), default="calibration")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((args.dataset / "metadata.json").read_text())
    horizon = metadata.get("decision_horizon", 1)
    if horizon not in (1, 2):
        parser.error("ordinary action selection needs first-candidate or automatic-repair outcomes")
    rows = list(map(json.loads, (args.dataset / f"{args.split}.jsonl").read_text().splitlines()))
    requests = {(r["run_id"], r["state_id"].split('/')[0], r["state"]["proposed_action"]): r
                for r in rows if r["state"]["development_budget_candidates"] == horizon}
    comparisons = list(map(json.loads, (args.dataset / "comparisons.jsonl").read_text().splitlines()))
    training_path = args.training_data or args.dataset
    training_metadata = json.loads((training_path / "metadata.json").read_text())
    if training_metadata["prompt_policy"] != metadata["prompt_policy"] or training_metadata["score_scales"] != metadata["score_scales"]:
        raise ValueError("training and matched evaluation conditions differ")
    training = [r for r in map(json.loads, (training_path / "train.jsonl").read_text().splitlines())
                if r["state"]["development_budget_candidates"] == horizon]
    if {r["run_id"] for r in training} & {r["run_id"] for r in rows}:
        raise ValueError("baseline training and held-out sources overlap")
    estimate = fit_baseline(training)
    from unsloth import FastDecisionModel
    import torch

    model, processor = FastDecisionModel.from_pretrained(args.adapter, dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit, max_seq_length=metadata["max_seq_length"])
    FastDecisionModel.for_inference(model)
    model.eval().requires_grad_(False)
    decisions = []
    for comparison in comparisons:
        if comparison["split"] != args.split:
            continue
        task, predictions, state_values, constant_values = comparison["task"], {}, {}, {}
        for action in ACTIONS:
            row = requests[comparison["run_id"], comparison["state_id"], action]
            state = row["state"]
            if state["generation_policy"] != f"TraceAAD V10.{metadata['prompt_policy'][3:]}":
                raise ValueError("generation policy does not match dataset metadata")
            with torch.no_grad():
                prediction = FastDecisionModel.predict(model, processor, state, questions(horizon))
            gain = expected_gain(prediction["frontier_gain"], training_metadata["reward_support"][str(horizon)][task])
            cost = expected_request_cost(prediction["repair_used"], state["remaining_candidates"]) if horizon == 2 else 1
            predictions[action] = {"gain": gain, "candidate_cost": cost, "gain_per_candidate": gain / cost}
            sg, sc = estimate(state)
            cg, cc = estimate(state, use_state=False)
            state_values[action], constant_values[action] = sg / sc, cg / cc
        gains, costs = matched_outcomes(comparison, horizon)
        model_weights = greedy_weights({a: predictions[a]["gain_per_candidate"] for a in ACTIONS})
        policies = {"model": model_weights, "fixed": FIXED,
                    "mixed": {a: 0.5 * model_weights[a] + 0.5 * FIXED[a] for a in ACTIONS},
                    "constant": greedy_weights(constant_values), "state_rule": greedy_weights(state_values),
                    "matched_sample_best": greedy_weights({a: gains[a] / costs[a] for a in ACTIONS})}
        decisions.append({"run_id": comparison["run_id"], "state_id": comparison["state_id"], "task": task,
            "predictions": predictions, "model_action_weights": model_weights, "observed_gains": gains,
            "observed_costs": costs, "policies": {name: policy_delivery(weights, gains, costs) for name, weights in policies.items()}})
    report = {"split": args.split, "states": len(decisions), "horizon": horizon,
              "baseline_training_data": str(training_path), **aggregate(decisions), "decisions": decisions,
              "scope": "Matched repeated outcomes at frozen states; sample-best is descriptive and uses observed outcomes. "
                       "States within one source are dependent. Full-search utility remains unmeasured.",
              "status": "needs_full_search_validation"}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ("split", "states", "equal_task_macro_gain_per_candidate", "offline_improvement")}, indent=2))


if __name__ == "__main__":
    main()
