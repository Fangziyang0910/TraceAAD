"""V14 revisit rejection, divergence after stagnation, and the development period."""

import json

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v14 import Config, TraceAADV14


def search(directory, *answers, **options):
    return TraceAADV14(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=directory,
        config=Config(**{"budget": 6, "roots": 1, **options}),
    )


def commented(value):
    code = f"def score(x):\n    # unchanged\n    return {value}\n"
    return json.dumps({"analysis": "", "design": "a claimed change", "code": code})


def test_identical_code_is_not_evaluated_and_the_request_is_asked_again(tmp_path):
    method = search(tmp_path, response(1), commented(1), response(2), budget=3, diverge=False)
    summary = method.run()
    first, copy, revised = method.programs.values()
    assert copy.status == "revisit" and copy.failure == {"kind": "revisit", "of": first.id}
    assert revised.repair_of == copy.id and revised.parent_id == first.id
    assert revised.action == copy.action and revised.idea == "return 2"
    assert "identical to the current algorithm" in method.llm.calls[-1][0]
    assert summary["budget_used"] == 3 and method.evaluation_calls == 2
    assert summary["v14"]["revisits"] == 1
    restored = search(tmp_path, budget=3, diverge=False)
    assert restored._identities == method._identities


def test_stagnation_starts_a_new_design_that_is_developed_before_it_competes(tmp_path):
    answers = [response(v) for v in (5, 1, 2, 3, 4, 6, 7, 8)]
    method = search(tmp_path, *answers, budget=8, stagnation_window=2, mature_steps=2)
    summary = method.run()
    actions = [p.action for p in method.programs.values()]
    assert actions[3:6] == ["Diverge", "Mature", "Mature"]
    assert all(a in ("Refine", "Explore", "Crossover") for a in actions[1:3] + actions[6:])
    diverge_prompt = method.llm.calls[3][0]
    assert "[Designs Developed So Far]" in diverge_prompt
    assert "def score(x):\n    return" not in diverge_prompt
    developed = method.programs[5]
    assert developed.parent_id == 4 and "[Development Period]" in method.llm.calls[4][0]
    assert method.programs[6].parent_id == 5
    (episode,) = method.v14["episodes"]
    assert episode["members"] == [4, 5, 6] and episode["new_best"] and episode["best_fitness"] == -6
    assert summary["best"]["fitness"] == -8 and summary["v14"]["episodes_with_new_best"] == 1


def test_resume_inside_a_development_period(tmp_path):
    answers = [response(v) for v in (5, 1, 2, 3, 4, 6, 7, 8)]
    control = search(tmp_path / "control", *answers, budget=8, stagnation_window=2, mature_steps=2)
    expected = control.run()
    first = search(tmp_path / "resumed", *answers[:5], budget=8, stagnation_window=2, mature_steps=2)
    first._roots()
    first._roots()
    for _ in range(4):
        first._search()
    assert first.v14["episode"] is not None
    resumed = search(tmp_path / "resumed", *answers[5:], budget=8, stagnation_window=2, mature_steps=2)
    actual = resumed.run()
    assert first.llm.calls + resumed.llm.calls == control.llm.calls
    assert actual["best"]["id"] == expected["best"]["id"]
    assert resumed.v14["episodes"] == control.v14["episodes"]
