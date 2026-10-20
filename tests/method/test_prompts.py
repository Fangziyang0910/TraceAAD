"""Check which search evidence reaches each operator, without fixing its prose."""

import pytest

from benchmarks.tasks import CO_TASKS, training_task
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad import Config, Program, TraceAAD
from traceaad.prompts import OP_OBJECTIVE, PromptBuilder


@pytest.mark.parametrize("action", ["Refine", "Explore", "Crossover"])
def test_operator_generates_one_child_with_its_parent_and_reference(tmp_path, action):
    method = TraceAAD(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(
            response(1), response(2), response(3, analysis="new computation", design="new rule")
        ),
        run_dir=tmp_path,
        config=Config(budget=3, roots=2),
    )
    method._roots()
    method._roots()
    parent, reference = method.programs.values()
    reference = reference if action == "Crossover" else None
    request = method.prompts.build(action, parent, reference=reference)
    child = method._create_program(request, parent=parent, reference=reference)
    assert len(method.llm.calls) == 3
    assert child.action == action and child.fitness == -3
    assert child.analysis == "new computation" and child.idea == "new rule"
    assert child.parent_id == parent.id
    assert child.reference_id == (reference.id if reference else None)
    assert parent.code.strip() in request["prompt"]
    if reference:
        assert reference.code.strip() in request["prompt"]


@pytest.mark.parametrize("action", ["Refine", "Explore", "Crossover"])
def test_operator_context_contains_the_selected_history_and_real_feedback(action):
    programs = {}
    for id, score, fields in [
        (1, -1, {}),
        (2, -2, {"parent_id": 1, "action": "Refine"}),
        (3, None, {"parent_id": 2, "action": "Explore"}),
        (4, -5, {"parent_id": 3, "action": "Repair", "repair_of": 3}),
        (5, -1, {}),
    ]:
        programs[id] = Program(
            id=id,
            generation=id,
            code=f"def score(x): return {id}",
            idea=f"design {id}",
            analysis="private reasoning",
            fitness=score,
            status="valid" if score is not None else "runtime_error",
            **{"action": "Init", **fields},
        )
    programs[3].failure = {"kind": "runtime_error", "error": "ValueError: bad output"}
    prompts = PromptBuilder(TokenLLM(), "test", TinyEvaluation(), programs, Config())
    request = prompts.build(action, programs[2], reference=programs[5])
    assert request["child_ids"] == [3]
    assert "private reasoning" not in request["prompt"]
    assert "design 3" in request["prompt"] and "bad output" in request["prompt"]
    if action == "Explore":
        assert request["progress_ids"] == [2, 4]
    else:
        assert request["history_edge_ids"] == [2]
    repair = prompts.repair(programs[3])["prompt"]
    assert programs[3].code in repair and "bad output" in repair


@pytest.mark.parametrize("task", CO_TASKS)
def test_task_text_is_the_one_every_method_sees(task):
    evaluation = training_task(task)[0]
    task_section = PromptBuilder(TokenLLM(), task, evaluation, {}, Config()).common[0]
    assert task_section == "[Task]\n" + evaluation.task_description.replace(OP_OBJECTIVE, "").strip()
