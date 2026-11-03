"""Maturity identity, finite protection, request accounting and recovery."""

import json
import random

import pytest

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v18_1 import Config, Program, TraceAADV181
from traceaad.v18_1._basic_selection import sample_parent as basic_sample_parent
from traceaad.v18_1.storage import rows
from traceaad.v18_1.selection import age_distribution, sample_parent


def audit(value, *, relation="new", birth_id=None):
    return json.dumps({
        "relation": relation, "birth_id": birth_id,
        "mechanism": f"decision rule {value}",
        "evidence": "The returned decision computation differs.",
        "code_evidence": f"return {value}",
    })


def search(directory, *answers, **options):
    return TraceAADV181(
        evaluation=TinyEvaluation(), llm=TokenLLM(*answers), run_dir=directory,
        config=Config(**{"budget": 4, "roots": 1, **options}),
    )


def node(pid, birth, age, fitness):
    return Program(
        pid, f"def score(x): return {pid}", "rule", None, "Refine", pid,
        status="valid", fitness=fitness,
        request={"maturity": {"birth_id": birth, "age": age}},
    )


def birth(bid, age=0, used=0):
    return {"birth_id": bid, "max_age": age, "protected_used": used, "confirmed": True}


def test_generation_budget_excludes_audits_and_duplicate_does_not_rebirth(tmp_path):
    method = search(
        tmp_path, response(1), audit(1), response(2), audit(2, relation="existing", birth_id=1),
        response(2), response(3), audit(3), age_fraction=1,
    )
    summary = method.run()
    assert summary["status"] == "finished"
    assert summary["budget_used"] == 4 and summary["model_calls"] == 7
    assert summary["best"]["fitness"] == -3
    assert [p.request["maturity"]["age"] for p in method.programs.values()] == [0, 1, 1, 0]
    assert [p.request["maturity"]["birth_id"] for p in method.programs.values()] == [1, 1, 1, 4]
    assert method.maturity_state["births"]["1"]["protected_used"] == 3
    assert summary["v18_1"]["birth_checks"] == 3
    assert len(list(rows(tmp_path / "calls.jsonl.gz"))) == 7


def test_repair_counts_one_refinement_and_no_additional_protection(tmp_path):
    method = search(
        tmp_path, response(1), audit(1), response("1/0"), response(2),
        audit(2, relation="existing", birth_id=1), budget=3, age_fraction=1,
    )
    summary = method.run()
    assert summary["budget_used"] == 3 and summary["model_calls"] == 5
    assert method.programs[3].request["maturity"]["age"] == 1
    assert method.maturity_state["births"]["1"]["protected_used"] == 1
    assert "maturity" not in method.programs[2].request


def test_failed_reply_spends_protection_and_exhaustion_survives_resume(tmp_path):
    method = search(tmp_path, response(1), audit(1), "invalid JSON", protection_attempts=1, age_fraction=1)
    method._roots()
    method._roots()
    method._search()
    assert method.maturity_state["births"]["1"]["protected_used"] == 1
    resumed = search(tmp_path, response(2), audit(2, relation="existing", birth_id=1),
                     response(3), audit(3, relation="existing", birth_id=1),
                     protection_attempts=1, age_fraction=1)
    summary = resumed.run()
    assert summary["v18_1"]["protected_attempts"] == 1
    assert resumed.programs[2].request["selection"]["channel"] == "ordinary"


def test_uncertain_audit_does_not_reset_age_or_grant_new_protection(tmp_path):
    method = search(tmp_path, response(1), audit(1), response(2), "not JSON", budget=2)
    summary = method.run()
    record = method.programs[2].request["maturity"]
    assert record["birth_id"] == 1 and record["age"] == 1
    assert record["relation"] == "uncertain"
    assert summary["v18_1"]["birth_check_failures"] == 1


def test_birth_tokenizer_failure_does_not_abort_search(tmp_path, monkeypatch):
    from traceaad.v18_1.runtime.llm import TokenizationError

    method = search(tmp_path, response(1), response(2), budget=2)
    original = method.llm.count_prompt_tokens

    def count(prompt):
        if "Independently audit" in prompt:
            raise TokenizationError("tokenizer offline")
        return original(prompt)

    monkeypatch.setattr(method.llm, "count_prompt_tokens", count)
    summary = method.run()
    assert summary["status"] == "finished" and summary["budget_used"] == 2
    assert summary["v18_1"]["birth_check_failures"] == 2
    assert summary["v18_1"]["birth_checks"] == 0
    assert summary["v18_1"]["pending_birth_id"] is None


def test_unknown_root_without_a_valid_audit_has_no_protection(tmp_path):
    method = search(tmp_path, response(1), "bad JSON", budget=1)
    summary = method.run()
    assert summary["status"] == "finished"
    assert not method.maturity_state["births"]["1"]["confirmed"]
    assert age_distribution(list(method.archive.values()), method.programs,
                            method.maturity_state["births"], method.config)[0] == {}


def test_same_age_uses_historical_age_instead_of_current_mature_score():
    nodes = [node(1, 1, 0, 120), node(2, 1, 1, 110), node(3, 1, 2, 80),
             node(4, 4, 0, 130), node(5, 4, 1, 100)]
    births = {"1": birth(1, 2), "4": birth(4, 1)}
    probabilities, ranks, peers = age_distribution(nodes, {p.id: p for p in nodes}, births, Config())
    assert ranks[5] == pytest.approx(2 / 3)
    assert peers[5] == 1 and probabilities[5] > probabilities[3]
    assert set(probabilities) == {3, 5}


def test_duplicate_population_does_not_multiply_age_channel_share():
    a, b, duplicate = node(1, 1, 0, 10), node(2, 2, 0, 10), node(3, 1, 0, 10)
    nodes = [a, b, duplicate]
    probabilities, _, _ = age_distribution(nodes, {p.id: p for p in nodes},
                                            {"1": birth(1), "2": birth(2)}, Config())
    assert probabilities == {1: 0.5, 2: 0.5}


def test_mature_or_exhausted_births_cannot_reenter_via_younger_versions():
    nodes = [node(1, 1, 0, 1), node(2, 1, 8, 2), node(3, 3, 0, 3)]
    probabilities, _, _ = age_distribution(nodes, {p.id: p for p in nodes},
                                            {"1": birth(1, 8), "3": birth(3, used=12)}, Config())
    assert probabilities == {}


def test_reappearing_existing_mechanism_keeps_spent_credit(tmp_path):
    method = search(tmp_path, response(1), audit(1), response(2), audit(2), budget=2)
    method.run()
    method.maturity_state["births"]["1"]["protected_used"] = 12
    candidate = node(3, 9, 0, 0)
    candidate.request = {}
    method._assign_birth(candidate, method.programs[2], None, {
        "relation": "existing", "birth_id": 1, "mechanism": "old mechanism",
        "evidence": "Recovered old decision rule", "code_evidence": "return 3",
    })
    assert candidate.request["maturity"]["age"] == 1
    assert method.maturity_state["births"]["1"]["protected_used"] == 12


def test_disabled_selection_matches_original_random_stream():
    nodes = [node(1, 1, 0, 10), node(2, 2, 0, 20)]
    programs = {p.id: p for p in nodes}
    control, actual = random.Random(4), random.Random(4)
    for _ in range(20):
        expected, _ = basic_sample_parent(nodes, programs, control)
        chosen, _ = sample_parent(nodes, programs, {}, Config(same_age=False), actual)
        assert chosen.id == expected.id
    assert actual.getstate() == control.getstate()


def test_protected_channel_can_select_low_absolute_quality():
    nodes = [node(i, i, 0, float(i)) for i in range(1, 21)]
    programs = {p.id: p for p in nodes}
    births = {str(p.id): birth(p.id, used=12 if p.id < 20 else 0) for p in nodes}
    chosen, record = sample_parent(nodes, programs, births, Config(age_fraction=1), random.Random(0))
    assert chosen.id == 20 and record["channel"] == "same_age"
    assert record["probability"] == 1


def test_resume_preserves_generation_prompts_audits_and_choices(tmp_path):
    replies = [response(1), audit(1), response(2), audit(2, relation="existing", birth_id=1),
               response(3), audit(3, relation="existing", birth_id=1)]
    control = search(tmp_path / "control", *replies, budget=3, age_fraction=1)
    expected = control.run()
    partial = search(tmp_path / "resume", *replies[:4], budget=3, age_fraction=1)
    partial._roots()
    partial._roots()
    partial._search()
    resumed = search(tmp_path / "resume", *replies[4:], budget=3, age_fraction=1)
    actual = resumed.run()
    assert partial.llm.calls + resumed.llm.calls == control.llm.calls
    assert actual["best"]["id"] == expected["best"]["id"]
    assert actual["v18_1"] == expected["v18_1"]


def test_pending_birth_audit_resumes_without_reevaluation(tmp_path, monkeypatch):
    partial = search(tmp_path, response(1), budget=1)
    monkeypatch.setattr(partial, "_ensure_birth", lambda: None)
    partial._roots()
    assert partial.maturity_state["pending_birth_id"] == 1
    resumed = search(tmp_path, audit(1), budget=1)
    summary = resumed.run()
    assert summary["status"] == "finished" and summary["evaluation_calls"] == 1
    assert len(resumed.llm.calls) == 1 and summary["v18_1"]["pending_birth_id"] is None


def test_received_audit_is_committed_before_parsing_and_not_repeated(tmp_path, monkeypatch):
    import traceaad.v18_1.search as search_module

    method = search(tmp_path, response(1), audit(1), budget=1)
    parser = search_module.parse_birth_response

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(search_module, "parse_birth_response", interrupted)
    with pytest.raises(KeyboardInterrupt):
        method._roots()
    monkeypatch.setattr(search_module, "parse_birth_response", parser)
    resumed = search(tmp_path, budget=1)
    summary = resumed.run()
    assert summary["model_calls"] == 2 and not resumed.llm.calls
    assert summary["v18_1"]["confirmed_births"] == 1
    assert len(list(rows(tmp_path / "calls.jsonl.gz"))) == 2


def test_unknown_audit_transport_is_not_retried_on_resume(tmp_path, monkeypatch):
    method = search(tmp_path, response(1), budget=1)
    original = method.llm.draw_sample_with_details

    def interrupted(prompt, **kwargs):
        if "Independently audit" in prompt:
            raise KeyboardInterrupt
        return original(prompt, **kwargs)

    monkeypatch.setattr(method.llm, "draw_sample_with_details", interrupted)
    with pytest.raises(KeyboardInterrupt):
        method._roots()
    resumed = search(tmp_path, budget=1)
    summary = resumed.run()
    assert summary["model_calls"] == 2 and not resumed.llm.calls
    assert summary["v18_1"]["uncertain_audit_calls"] == 1
    calls = list(rows(tmp_path / "calls.jsonl.gz"))
    assert calls[-1]["uncertain"] and calls[-1]["stage"] == "birthcheck"


def test_large_registry_is_retrieved_within_context_and_preserves_parent(tmp_path):
    method = search(tmp_path, birth_context_tokens=1400)
    for pid in range(1, 181):
        program = node(pid, pid, 0, float(pid))
        method.programs[pid] = program
        method.maturity_state["births"][str(pid)] = {
            **birth(pid), "born_program_id": pid,
            "mechanism": "rule " * 20, "evidence": "decision " * 20,
        }
    program = node(181, 1, 1, 0)
    prompt, tokens, checked, omitted = method._audit_prompt(program, method.programs[180], None)
    assert tokens <= 1400 and 180 in checked and omitted > 0
    assert "Candidate Code" in prompt and checked


@pytest.mark.parametrize("options", [{"age_fraction": -1}, {"age_fraction": float("nan")},
                                     {"protection_attempts": 0}, {"max_refinement_age": True}])
def test_config_rejects_invalid_policy(options):
    with pytest.raises(ValueError):
        Config(**options)
