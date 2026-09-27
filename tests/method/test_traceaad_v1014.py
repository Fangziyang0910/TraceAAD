"""Observable V10.14 invariants, including real execution of a closed square."""

from dataclasses import replace
import json

import pytest

from core import Evaluation, SecureEvaluator
from tests.support import FakeLLM, TinyEvaluation, response
from traceaad.v10_14 import Config, TraceAADV1014
from traceaad.v10_14.edits import Edit, SourceError, apply_edits, changed_symbols, close_trace, parse_response
from traceaad.v10_14.evaluation import SeededEvaluation, protocol_identity
from traceaad.v10_14.frontier import Frontier
from traceaad.v10_14.prompts import ContextError
from traceaad.v10_14.state import Ledger


def program(a, b, *, invalid_without_a=False):
    failure = "    if b and not a:\n        raise ValueError('B requires A')\n" if invalid_without_a else ""
    return f"def score(x):\n    a = {a}\n    b = {b}\n{failure}    return 100 + 4*a + 10*b - 7*a*b\n"


def full(code):
    return "```python\n" + code.rstrip("\n") + "\n```"


def method(tmp_path, responses=(), **settings):
    values = dict(budget=40, max_evaluations=40, init_proposals=1)
    values.update(settings)
    return TraceAADV1014(evaluation=TinyEvaluation(), llm=FakeLLM(*responses),
                        run_dir=tmp_path, config=Config(**values))


def square_search(tmp_path, invalid=False, feedback=True):
    sources = [program(a, b, invalid_without_a=invalid) for a, b in [(0, 0), (1, 0), (1, 1)]]
    m = method(tmp_path, [*(full(p) for p in sources), response(111)], comparison_feedback=feedback)
    m._initialize()  # one root proposal
    m._initialize()  # bootstrap A
    m._initialize()  # freeze the frontier
    parent = m.anchors[2]
    m._begin("main", 1, parent, 0)
    m._attempt(anchor=parent, request=m.prompts.build(parent))
    m._finish_session()
    region, closure = m._closure_options()[0]
    m._begin("recheck", 2, m.anchors[3], region, closure)
    return m


def test_exact_square_and_overlap_rejection():
    p00, p10, p11 = [program(a, b) for a, b in [(0, 0), (1, 0), (1, 1)]]
    missing, check = close_trace(p00, p10, p11)
    assert missing == program(0, 1)
    assert len(set(check["snapshots"])) == 4
    with pytest.raises(ValueError):
        apply_edits("abcde", [Edit("abc", "A"), Edit("bcd", "B")])
    with pytest.raises(ValueError):
        apply_edits("x x", [Edit("x", "y")])
    with pytest.raises(ValueError):
        apply_edits("abc", [Edit("a", "z"), Edit("z", "q")])
    # B overwrites A itself; there is no exchangeable square.
    with pytest.raises(ValueError):
        close_trace(p00, p10, program(2, 0))


@pytest.mark.parametrize("text,finish", [
    (response(1), "length"),
    (response(1) + response(2), "stop"),
    ("```python\ndef score(x):\n    return 1", "stop"),
    ("<think>unfinished", "stop"),
    ('{"mode":"edit","edits":[]}', "stop"),
])
def test_parser_rejects_incomplete_or_ambiguous_delivery(text, finish):
    with pytest.raises((ValueError, SyntaxError)):
        parse_response(text, finish, TinyEvaluation().template_program)


def test_parser_preserves_source_and_failed_payload_for_repair():
    code = 'def score(x):\n    # exactly retained\n    return x + 1'
    parsed, note = parse_response(full(code), "stop", TinyEvaluation().template_program)
    assert parsed == code
    assert note == ""
    broken = 'def score(x):\n    return ('
    with pytest.raises(SourceError) as error:
        parse_response(full(broken), "stop", TinyEvaluation().template_program)
    assert error.value.code == broken


def test_atomic_edit_output_and_interface_check():
    base = program(0, 0)
    payload = json.dumps({"mode": "edit", "edits": [{"search": "a = 0", "replacement": "a = 1"}]})
    code, _ = parse_response(payload, "stop", TinyEvaluation().template_program, base=base, mode="edit")
    assert code == program(1, 0)
    with pytest.raises(SourceError, match="signature"):
        parse_response("def score(x, y):\n    return x", "stop", TinyEvaluation().template_program)


def test_ledger_reserves_whole_ticket_and_charges_failures():
    ledger = Ledger(10, 10, search_limit=10)
    caps = {"trial": 2, "recheck": 1}
    assert not ledger.can_reserve("trial", 3, 1, caps)
    assert not ledger.can_reserve("recheck", 2, 1, caps)
    ledger.reserve("main", 1, 1, caps)
    ledger.consume_candidate()  # invalid output: no evaluator call
    ledger.release()
    assert (ledger.candidates, ledger.evaluations) == (1, 0)
    ledger.reserve("trial", 2, 1, caps)
    ledger.consume_candidate()
    ledger.consume_evaluation()
    released = ledger.release()
    assert released["candidates"] == 1
    assert ledger.channel_used["trial"] == 1
    assert ledger.can_reserve("trial", 1, 1, caps)
    assert not ledger.can_reserve("trial", 2, 1, caps)


def test_bounded_initialization_counts_invalid_and_duplicate_attempts(tmp_path):
    m = method(tmp_path, ["bad output", response(1), response(1), response(2)],
               budget=4, max_evaluations=4, init_proposals=8)
    summary = m.run()
    attempts = list(m.facts.tables["attempt"].values())
    assert summary["budget_used"] == 4
    assert m.init_index == 2  # small budget shrinks proposals in advance
    assert m.bootstrap_index == 1
    assert len([a for a in attempts if a["channel"] == "init"]) == 3
    assert attempts[0]["status"] == "delivery_failed"
    assert attempts[2]["status"] == "duplicate"
    assert m.ledger.evaluations == 2  # duplicate successful source uses its fixed-panel cache


def test_all_invalid_initialization_stops_without_free_retries(tmp_path):
    m = method(tmp_path, ["invalid", "invalid"], budget=4, init_proposals=2)
    result = m.run()
    assert result["status"] == "no_valid_root"
    assert m.ledger.candidates == 2
    assert m.ledger.evaluations == 0


def test_recheck_executes_missing_program_and_injects_comparison(tmp_path):
    m = square_search(tmp_path)
    m._step_session()
    comparison = m.facts.tables["comparison"][1]
    assert comparison["delta_old"] == 4
    assert comparison["delta_current"] == -3
    assert comparison["interaction"] == -7
    p01 = m.anchors[4]
    assert p01["fitness"] == 110 and p01["parent_id"] == 3
    assert p01["evaluation_ids"] != m.anchors[3]["evaluation_ids"]
    m._step_session()
    req = list(m.facts.tables["request"].values())[-1]
    assert "comparison:1" in req["evidence_ids"]
    assert '"delta_current": -3.0' in req["prompt"]
    assert '"uncertainty"' in req["prompt"]
    assert m.facts.tables["attempt"][5]["parent_id"] == 4
    m._finish_session()
    assert m.ledger.channel_used["recheck"] == 2
    assert m.facts.tables["session"][4]["direct_recheck_gain"] == 3
    assert m.facts.tables["session"][4]["followup_gain"] == 1
    assert m.facts.tables["session"][4]["global_gain"] == 4


def test_invalid_counterfactual_does_not_get_zero_or_a_followup(tmp_path):
    m = square_search(tmp_path, invalid=True)
    calls_before = m.ledger.calls
    m._step_session()
    record = m.facts.tables["comparison"][1]
    assert record["status"] == "invalid"
    assert "delta_current" not in record
    assert m.session is None
    assert m.ledger.channel_used["recheck"] == 1
    assert m.ledger.calls == calls_before
    assert "B requires A" in m.facts.tables["attempt"][4]["error"]


def test_feedback_ablation_keeps_identical_candidate_and_start(tmp_path):
    on = square_search(tmp_path / "on", feedback=True)
    off = square_search(tmp_path / "off", feedback=False)
    for m in (on, off):
        m._step_session()
        m._step_session()
    assert on.ledger.evaluations == off.ledger.evaluations
    assert on.facts.code(on.anchors[4]) == off.facts.code(off.anchors[4])
    assert on.facts.tables["attempt"][5]["parent_id"] == off.facts.tables["attempt"][5]["parent_id"] == 4
    assert "comparison:1" not in list(off.facts.tables["request"].values())[-1]["evidence_ids"]


def test_failed_completion_is_not_rescued_by_first_code_block(tmp_path):
    class TruncatedLLM(FakeLLM):
        def draw_sample_with_details(self, prompt, **kwargs):
            result = super().draw_sample_with_details(prompt, **kwargs)
            result["finish_reason"] = "length"
            return result
    m = TraceAADV1014(evaluation=TinyEvaluation(), llm=TruncatedLLM(response(10)), run_dir=tmp_path,
                     config=Config(budget=1, max_evaluations=1, init_proposals=1))
    m.run()
    assert m.ledger.candidates == 1 and m.ledger.evaluations == 0


def test_trial_keeps_regressed_working_anchor_and_never_renews_ticket(tmp_path):
    m = method(tmp_path, [response(100), response(100), response(70), response(90), response(101)])
    for _ in range(3):
        m._initialize()
    parent = m.anchors[1]
    m._begin("trial", 3, parent, 0)
    ticket_id = m.session["id"]
    m._step_session()
    assert m.anchors[m.session["working"]]["fitness"] == 70
    m._step_session()
    m._step_session()
    assert m.session["id"] == ticket_id
    assert m.session["step"] == 3
    assert m.ledger.reservation["candidates"] == 0
    m._finish_session()
    assert m.facts.tables["session"][ticket_id]["net_gain"] == 1
    assert m.ledger.channel_used["trial"] == 3


def test_trial_repair_reads_failed_source_and_costs_one_of_three_steps(tmp_path):
    broken = "def score(x):\n    return 1/0"
    m = method(tmp_path, [response(100), response(100), full(broken), response(80), response(101)])
    for _ in range(3):
        m._initialize()
    m._begin("trial", 3, m.anchors[1], 0)
    m._step_session()
    assert m.session["repair"] == 3
    m._step_session()
    req = list(m.facts.tables["request"].values())[-1]
    assert broken in req["prompt"] and "division by zero" in req["prompt"]
    assert m.facts.tables["attempt"][4]["repair_of"] == 3
    m._step_session()
    assert m.ledger.channel_used["trial"] == 3


def test_required_source_and_comparison_are_never_silently_cut(tmp_path):
    m = square_search(tmp_path)
    m._step_session()
    builder = m.prompts
    builder.config = replace(m.config, evidence_tokens=1)
    with pytest.raises(ContextError, match="comparison"):
        builder.build(m.anchors[4], comparison=m.facts.tables["comparison"][1])
    builder.config = replace(m.config, max_input_tokens=1)
    with pytest.raises(ContextError, match="source"):
        builder.build(m.anchors[4])


def test_request_records_only_evidence_actually_visible_and_drops_donor(tmp_path):
    m = square_search(tmp_path)
    m.prompts.config = replace(m.config, evidence_tokens=1)
    packet = m.prompts.build(m.anchors[3], donor=m.anchors[1], reference_mode="Transfer")
    assert not packet["evidence_ids"]
    assert packet["donor_id"] is None
    assert packet["reference_mode"] == "None"
    assert m.facts.code(m.anchors[3]) in packet["prompt"]


def test_completed_search_resume_has_no_extra_calls_or_initialization(tmp_path):
    m = method(tmp_path, [response(1), response(2), response(3)], budget=3)
    result = m.run()
    resumed = method(tmp_path, budget=3)
    assert resumed.run()["ledger"] == result["ledger"]
    assert not resumed.llm.calls
    with pytest.raises(ValueError, match="changed"):
        method(tmp_path, budget=4)


def test_resume_after_completed_initial_candidate_does_not_reissue_root(tmp_path):
    m = method(tmp_path, [response(1)])
    m._begin("init", 1)
    m._attempt(request=m.prompts.build(scope="Init"))
    # Crash after candidate commit and before session close.
    resumed = method(tmp_path, [response(2)] * 40)
    resumed.run()
    assert resumed.init_index == 1
    assert len([a for a in resumed.anchors.values() if a["parent_id"] is None]) == 1


def test_uncertain_external_result_blocks_replay(tmp_path, monkeypatch):
    m = method(tmp_path, [response(1)])
    original = m.facts.add

    def crash(table, data):
        if table == "evaluation":
            raise OSError("crash before saving outcome")
        return original(table, data)

    monkeypatch.setattr(m.facts, "add", crash)
    with pytest.raises(OSError):
        m.run()
    with pytest.raises(RuntimeError, match="uncertain"):
        method(tmp_path)


def test_frontier_weights_ignore_duplicate_states_and_freeze_references():
    def anchor(i, score, profile, artifact=None):
        return {"id": i, "fitness": score, "profile": profile, "artifact_id": artifact or str(i), "origin_region": None}
    anchors = {0: anchor(0, 100, [0, 0]), 1: anchor(1, 90, [1, 1])}
    f = Frontier(Config(), anchors)
    f.freeze()
    refs = [r["reference"][:] for r in f.regions]
    before = f.main_weights()
    for i in range(2, 12):
        anchors[i] = anchor(i, 100, [0, 0], "0")
        f.admit(anchors[i])
    assert f.main_weights() == before
    assert [r["reference"] for r in f.regions] == refs
    f.regions[0]["stagnation"] = 10000
    assert f.main_weights()[1][0] == .25


def test_seeded_execution_is_repeatable_and_restores_host_rng():
    import random
    import numpy as np
    evaluator = SecureEvaluator(SeededEvaluation(TinyEvaluation()))
    code = "import random\nimport numpy as np\nv=random.random()\ndef score(x):\n    return v + random.random() + np.random.random()"
    random.seed(9)
    np.random.seed(9)
    py_state, np_state = random.getstate(), np.random.get_state()
    first = evaluator.evaluate_program(TinyEvaluation().template_program, source=code, seed=42)
    second = evaluator.evaluate_program(TinyEvaluation().template_program, source=code, seed=42)
    assert first["score"] == second["score"]
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])


class SelectionEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program=TinyEvaluation().template_program,
                         task_description="A separate selection objective", safe_evaluate=False)
    def evaluate_program(self, program_str, callable_func):
        return -abs(callable_func(2)-2)


def test_final_selection_is_frozen_separate_and_can_choose_training_nonbest(tmp_path):
    m = TraceAADV1014(evaluation=TinyEvaluation(), selection_evaluation=SelectionEvaluation(),
        llm=FakeLLM(response(1), response(2), response(3)), run_dir=tmp_path,
        config=Config(budget=3, max_evaluations=3, init_proposals=1))
    summary = m.run()
    assert summary["status"] == "finished"
    assert summary["best"]["fitness"] == 2
    assert summary["best"]["selection_fitness"] == 0
    assert m.ledger.evaluations == 3
    assert summary["selection_evaluations"] == 3
    assert "return 2" in (tmp_path / "best_program.py").read_text()
    from experiments.infra.artifacts import pick_best_sample
    assert "return 2" in pick_best_sample(tmp_path)[0]["program"]
    assert len(m.llm.calls) == 3


def test_protocol_identity_rejects_changed_data_and_selection_reuse(tmp_path):
    evaluation = TinyEvaluation()
    evaluation.seed = 1
    p1 = protocol_identity(evaluation, (1,), [], "search")[0]
    evaluation.seed = 2
    assert p1 != protocol_identity(evaluation, (1,), [], "search")[0]
    with pytest.raises(ValueError, match="distinct"):
        TraceAADV1014(evaluation=evaluation, selection_evaluation=evaluation,
                        llm=FakeLLM(), run_dir=tmp_path)


def test_prior_comparison_remains_available_on_the_exact_background(tmp_path):
    m = square_search(tmp_path)
    m._step_session()
    packet = m.prompts.build(m.anchors[4])
    assert "comparison:1" in packet["evidence_ids"]
    assert packet["evidence_tokens"] <= m.config.evidence_tokens


def test_recheck_resume_after_counterfactual_commit_completes_comparison(tmp_path):
    m = square_search(tmp_path)
    source = m.session["closure"]["source"]
    m._attempt(anchor=m.anchors[3], source=source)
    assert not m.facts.tables["comparison"]
    resumed = method(tmp_path, [response(111)])
    before = resumed.ledger.evaluations
    resumed._step_session()
    assert resumed.ledger.evaluations == before
    assert resumed.facts.tables["comparison"][1]["delta_current"] == -3
    resumed._step_session()
    assert "comparison:1" in list(resumed.facts.tables["request"].values())[-1]["evidence_ids"]


def test_session_commit_resume_does_not_issue_an_extra_root(tmp_path, monkeypatch):
    m = method(tmp_path, [response(1)])
    m._begin("init", 1)
    m._attempt(request=m.prompts.build(scope="Init"))

    def crash():
        raise OSError("after session fact before checkpoint")

    monkeypatch.setattr(m, "_save", crash)
    with pytest.raises(OSError):
        m._finish_session()
    resumed = method(tmp_path, [response(2)] * 40)
    resumed.run()
    assert resumed.init_index == 1
    assert len([a for a in resumed.anchors.values() if a["parent_id"] is None]) == 1


def test_main_repair_is_a_separate_paid_event_and_edit_uses_failed_source(tmp_path):
    m = method(tmp_path, [response(100), response(101)], output_mode="full")
    for _ in range(3):
        m._initialize()
    anchor = m.anchors[2]
    broken = "def score(x):\n    value = 1 / 0\n    return value"
    m.llm.responses = iter([full(broken)])
    m._begin("main", 1, anchor, 0)
    m._step_session()
    assert m.session["step"] == 1
    assert m.ledger.calls == 3  # no automatic free repair
    m._finish_session()
    m._begin("main", 1, anchor, 0)
    m.prompts.config = replace(m.config, output_mode="edit")
    m.llm.responses = iter([json.dumps({"mode": "edit", "edits": [
        {"search": "value = 1 / 0", "replacement": "value = 102"}]} )])
    m._step_session()
    assert m.anchors[4]["fitness"] == 102
    assert m.facts.tables["attempt"][4]["repair_of"] == 3
    assert m.ledger.channel_used["main"] == 2


def test_repeated_evaluation_consumes_each_seed_and_keeps_panel(tmp_path):
    m = method(tmp_path, [response(1), response(2)], budget=2,
               max_evaluations=4, evaluation_seeds=(10, 11))
    m.run()
    assert m.ledger.evaluations == 4
    assert m.anchors[1]["scores"] == [1, 1]
    assert [e["seed"] for e in m.facts.tables["evaluation"].values()] == [10, 11, 10, 11]


def test_token_cap_does_not_start_a_session_it_cannot_pay(tmp_path):
    m = method(tmp_path, [response(1)], budget=2, max_total_tokens=1)
    m.run()
    assert m.ledger.calls == 0
    assert not m.llm.calls


def test_template_imports_preserve_future_import_order():
    template = "import numpy as np\ndef score(x):\n    pass"
    code = '"module docs"\nfrom __future__ import annotations\ndef score(x):\n    return np.sqrt(x)'
    parsed, _ = parse_response(full(code), "stop", template)
    assert parsed.index("from __future__") < parsed.index("import numpy")
    compile(parsed, "<test>", "exec")


def test_actual_syntax_deviation_does_not_discard_a_valid_candidate(tmp_path):
    m = method(tmp_path, [response(1), response(2), "def score(x):\n    if x:\n        return 3\n    return 0"])
    for _ in range(3):
        m._initialize()
    m._begin("main", 1, m.anchors[2], 0)
    m._attempt(anchor=m.anchors[2], request=m.prompts.build(m.anchors[2], scope="Tune"))
    assert m.anchors[3]["fitness"] == 3
    assert m.facts.tables["attempt"][3]["contract_deviation"]


def test_new_local_evidence_can_reactivate_a_challenger_once():
    anchors = {1: {"id": 1, "fitness": 100, "profile": [0], "artifact_id": "a"},
               2: {"id": 2, "fitness": 95, "profile": [1], "artifact_id": "b"}}
    f = Frontier(Config(regions=1), anchors)
    f.freeze()
    f.regions[0]["tried"].append("b")
    f.regions[0]["challenger"] = None
    assert f.reactivate(anchors[2], 1)
    assert f.regions[0]["challenger"] == 2
    f.regions[0]["tried"].append("b")
    assert not f.reactivate(anchors[2], 1)
    assert "b" in f.regions[0]["tried"]


def test_revision_scope_names_changed_functions_not_every_function():
    before = "def helper(x):\n    return x\n\ndef score(x):\n    return helper(x)\n"
    after = before.replace("return x", "return x + 1")
    assert changed_symbols(before, after) == ["helper"]
