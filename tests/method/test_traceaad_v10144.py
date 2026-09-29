"""V10.14-4's discovery, region allocation and measured-feedback contracts."""

import pytest

from tests.support import TinyEvaluation, TokenLLM, text_candidate
from traceaad.v10_14_4 import Config, TraceAADV10144
from traceaad.v10_14_4.frontier import Frontier, distance


def test_probability_profiles_use_magnitude_and_discrete_profiles_use_choices():
    assert distance([0.5, 0.5], [0.51, 0.49]) == pytest.approx(0.01)
    assert distance([1, 2, 3], [1, 4, 3]) == 1 / 3
    assert distance([], [1]) is None


def test_main_region_weights_reach_distinct_families():
    anchors = {
        1: {"id": 1, "artifact_id": "a", "fitness": 10., "profile": [0, 0], "origin_region": None},
        2: {"id": 2, "artifact_id": "b", "fitness": 9., "profile": [1, 1], "origin_region": None},
    }
    frontier = Frontier(Config(regions=2), anchors)
    frontier.freeze()
    assert {frontier.region_for(a) for a in anchors.values()} == {0, 1}
    frontier.regions[0]["main_used"] = 100
    frontier.regions[1]["main_used"] = 0
    ids, weights = frontier.main_weights()
    assert dict(zip(ids, weights))[1] > dict(zip(ids, weights))[0]


def test_initialization_reserves_regions_for_later_discovery():
    anchors = {i: {"id": i, "artifact_id": str(i), "fitness": float(i),
                   "profile": [i, i], "parent_id": None, "origin_region": None}
               for i in range(1, 9)}
    frontier = Frontier(Config(regions=8, initial_regions=4), anchors)
    frontier.freeze()
    assert len(frontier.regions) == 4
    child = {"id": 9, "artifact_id": "9", "fitness": 9., "profile": [9, 9],
             "parent_id": None, "origin_region": None}
    anchors[9] = child
    assert frontier.admit_region(child)
    assert len(frontier.regions) == 5 and frontier.region_for(child) == 4


def test_bounded_discovery_is_paid_and_followed_by_development(tmp_path):
    responses = [text_candidate(i, code=f"def score(x):\n    return {i}\n") for i in range(50)]
    config = Config(budget=40, max_evaluations=40, init_proposals=1, discovery_fraction=.12)
    method = TraceAADV10144(evaluation=TinyEvaluation(), llm=TokenLLM(*responses),
                              run_dir=tmp_path, config=config)
    result = method.run()
    assert result["status"] == "search_complete"
    assert method.ledger.candidates == 40
    assert method.discovery_count == 4
    pivots = [a for a in method.facts.tables["attempt"].values() if a["scope"] == "Pivot"]
    assert len(pivots) == 4 and all(a["parent_id"] is None for a in pivots)
    for pivot in pivots:
        assert any(a["parent_id"] == pivot["id"] and a["scope"] == "Refine"
                   for a in method.facts.tables["attempt"].values())
    restored = TraceAADV10144(evaluation=TinyEvaluation(), llm=TokenLLM(),
                                run_dir=tmp_path, config=config)
    assert restored.discovery_count == 4 and restored.develop_queue == []


def test_prompt_reports_measured_transition_without_claiming_equivalence(tmp_path):
    method = TraceAADV10144(evaluation=TinyEvaluation(), llm=TokenLLM(), run_dir=tmp_path,
                              config=Config(budget=3, max_evaluations=3, init_proposals=1))
    parent = {"id": 1, "artifact_id": "a", "fitness": 1., "profile": [1, 2],
              "scenes": [{"scene": "early", "decision": [1]}]}
    child = {"id": 2, "artifact_id": "b", "parent_id": 1, "fitness": 2.,
             "profile": [1, 3], "scenes": [{"scene": "early", "decision": [3]}], "idea": ""}
    method.facts.add("artifact", {"id": "b", "code": "def score(x):\n return 2", "source_sha256": "b"})
    method.facts.tables["anchor"][1] = parent
    method.facts.tables["anchor"][2] = child
    feedback = method.prompts.probe_feedback(parent, child)
    assert feedback["distance"] == .5
    assert feedback["example_changes"][0]["before"] == [1]
    prompt = method.prompts.build(child)["prompt"]
    assert "Last measured transition" in prompt and '"fitness_change": 1.0' in prompt
    assert "not global equivalence" in prompt


def test_finalists_keep_two_best_then_cover_other_regions(tmp_path):
    method = TraceAADV10144(evaluation=TinyEvaluation(), llm=TokenLLM(), run_dir=tmp_path,
                              config=Config(budget=5, max_evaluations=5, init_proposals=1,
                                            regions=3, final_candidates=4))
    for i, fitness, profile in [(1, 10., [0, 0]), (2, 9., [0, 0]),
                                (3, 8., [0, 0]), (4, 7., [1, 0]), (5, 6., [1, 1])]:
        code = f"def score(x):\n return {i}\n"
        artifact_id = method.facts.artifact(code, method.environment)
        method.facts.add("anchor", {"id": i, "artifact_id": artifact_id, "fitness": fitness,
                                     "profile": profile, "parent_id": None, "origin_region": None,
                                     "evaluation_ids": [i]})
    method.frontier.freeze()
    method._freeze_finalists()
    assert method.finalists == [1, 2, 4, 5]
