"""V15 exposure policy: equal-effort allocation across lineages and code shown only for the edited program."""

import json

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v14 import Config as V14Config, TraceAADV14
from traceaad.v15 import Config, TraceAADV15


def search(directory, *answers, **options):
    return TraceAADV15(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=directory,
        config=Config(**{"budget": 6, "roots": 1, **options}),
    )


def same(value):
    code = f"def score(x):\n    # unchanged\n    return {value}\n"
    return json.dumps({"analysis": "", "design": "a claimed change", "code": code})


def test_with_both_rules_off_the_search_is_v14(tmp_path):
    answers = [response(5), response(3), same(5), response(4)] + [response(v) for v in (6, 1, 2, 7, 8, 9, 10, 11)]
    options = dict(budget=12, roots=2, stagnation_window=2, mature_steps=2, seed=1)
    v14 = TraceAADV14(
        evaluation=TinyEvaluation(), llm=TokenLLM(*answers), run_dir=tmp_path / "v14", config=V14Config(**options)
    )
    v15 = search(tmp_path / "v15", *answers, allocation="fitness", exposure="code", **options)
    v14.run()
    v15.run()
    assert [c[0] for c in v15.llm.calls] == [c[0] for c in v14.llm.calls]
    assert [(p.action, p.parent_id) for p in v15.programs.values()] == [
        (p.action, p.parent_id) for p in v14.programs.values()
    ]


def test_information_exposure_shows_code_only_for_the_program_being_edited(tmp_path):
    answers = [response(v, design=f"rule {v}") for v in (3, 5, 4, 6, 7, 8, 9)]
    method = search(
        tmp_path, *answers, budget=7, roots=3, allocation="fitness", diverge=False, operators={"Crossover": 1.0}
    )
    method.run()
    prompts = [c[0] for c in method.llm.calls]
    assert "rule 3" in prompts[1] and "    return 3" not in prompts[1]  # the second root sees the first by Design
    child = next(p for p in method.programs.values() if p.action == "Crossover")
    reference, prompt = method.programs[child.reference_id], prompts[child.id - 1]
    assert "[Reference Design]" in prompt and f"rule {-reference.fitness:g}" in prompt
    assert reference.code.splitlines()[-1] not in prompt and method.programs[child.parent_id].code.strip() in prompt
    assert "Code diff" not in "".join(prompts)
    output = prompt[prompt.rindex("[Output Format]") :]
    assert output.index('"code"') < output.index('"design"')


def test_challengers_are_developed_and_compared_with_the_champion_at_equal_effort(tmp_path):
    # Roots A (score -5) and B (-3); one development step, so checkpoints are 2, 4, 8.
    answers = [response(v) for v in (5, 3, 4, 7, 8)]
    method = search(tmp_path, *answers, budget=5, roots=2, mature_steps=1, diverge=False)
    method.run()
    a, b, b1, a1, a2 = method.programs.values()
    assert (b1.action, b1.parent_id) == ("Mature", b.id)  # B develops first
    assert (a1.action, a1.parent_id) == ("Mature", a.id)  # B is behind A after 2 attempts; A develops
    assert a2.action in ("Refine", "Explore", "Crossover") and a2.parent_id in (a.id, a1.id)
    assert b1.request["allocation"] == {"lineage": b.id, "champion": a.id, "effort": 1, "checkpoint": None}
    assert a2.request["allocation"]["lineage"] == a.id and a2.request["allocation"]["effort"] == 2


def test_a_challenger_that_keeps_pace_continues_and_can_take_over(tmp_path):
    answers = [response(v) for v in (5, 3, 5.0, 4, 6, 4.5)]
    method = search(tmp_path, *answers, budget=6, roots=2, mature_steps=1, diverge=False)
    method.run()
    a, b, b1, b2, b3, a1 = method.programs.values()
    assert b1.action == "Mature"  # B ties A's first score after two attempts, so it stays active
    assert b2.request["allocation"]["checkpoint"] == 2 and b3.request["allocation"]["lineage"] == b.id
    assert method._lineage_of(b3) == b.id and b3.fitness == -6  # B now holds the best program
    assert (a1.action, a1.parent_id) == ("Mature", a.id)  # A is now the challenger and develops


def test_a_new_lineage_starts_after_stagnation_without_code_and_is_developed(tmp_path):
    answers = [response(v) for v in (5, 4, 3, 6, 7)]
    method = search(tmp_path, *answers, budget=5, roots=1, mature_steps=1, stagnation_window=2)
    summary = method.run()
    a, a1, a2, c, c1 = method.programs.values()
    assert [p.action for p in (a1, c, c1)] == ["Mature", "Diverge", "Mature"]
    assert a2.parent_id in (a.id, a1.id) and c.parent_id is None and c1.parent_id == c.id
    assert "def score(x):\n    return" not in method.llm.calls[3][0]
    assert summary["v15"]["lineages"] == 2 and summary["v15"]["champion_lineage"] == c.id


def test_resume_continues_the_same_allocation(tmp_path):
    # Development of B, a new lineage after stagnation, then development of A and the new champion.
    answers = [response(v) for v in (5, 3, 5.0, 4, 6, 4.5, 7, 2)]
    options = dict(budget=8, roots=2, mature_steps=1, stagnation_window=3)
    control = search(tmp_path / "control", *answers, **options)
    expected = control.run()
    assert [p.action for p in control.programs.values()][2:7] == ["Mature", "Refine", "Diverge", "Mature", "Mature"]
    first = search(tmp_path / "resumed", *answers[:5], **options)
    for _ in range(3):
        first._roots()
    for _ in range(3):
        first._search()
    resumed = search(tmp_path / "resumed", *answers[5:], **options)
    actual = resumed.run()
    assert [c[0] for c in first.llm.calls + resumed.llm.calls] == [c[0] for c in control.llm.calls]
    assert actual["best"]["id"] == expected["best"]["id"] and actual["v15"] == expected["v15"]
