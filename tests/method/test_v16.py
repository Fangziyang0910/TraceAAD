"""V16: a challenger behind the champion at a checkpoint continues when its recent rate reaches the champion at
twice the checkpoint; the edited lineage is shown by changes, and a Crossover reference comes from another lineage."""

import json

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad import Program
from traceaad.v15 import Config as V15Config, TraceAADV15
from traceaad.v16 import Config, TraceAADV16
from traceaad.v16.prompts import PromptBuilderV16
from traceaad.v16.search import projected


def search(directory, *answers, kind=TraceAADV16, config=Config, **options):
    return kind(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=directory,
        config=config(**{"budget": 5, "roots": 2, "mature_steps": 1, "diverge": False, **options}),
    )


def test_the_rate_starts_at_half_the_checkpoint_or_the_first_valid_attempt(tmp_path):
    method = search(tmp_path, *(response(v) for v in (9, 1, 2, 3, 5, 6, 7, 8)), budget=8, roots=8)
    method.run()
    attempts = list(method.programs.values())  # scores -9, -1, -2, -3, -5, -6, -7, -8
    assert projected(attempts[1:5], 4) == -5 - 1.5 * 4  # window from age 2: (-2 - -5) / 2 per attempt
    assert projected(attempts[1:2], 1) == -1  # no window: compared at the checkpoint only


def test_a_fast_challenger_behind_the_champion_continues(tmp_path):
    # Roots A (-5) and B (-3); B reaches -4 at its checkpoint 2, behind A's -5. Its rate of 1 per attempt
    # projects -6 at age 4, ahead of A's -5, so B continues; V15 pauses it and develops A.
    answers = [response(v) for v in (5, 3, 4, 7, 8)]
    v16 = search(tmp_path / "v16", *answers)
    v15 = search(tmp_path / "v15", *answers, kind=TraceAADV15, config=V15Config)
    v16.run()
    v15.run()
    a, b, b1, b2 = list(v16.programs.values())[:4]
    assert b1.action == "Mature" and v16._lineage_of(b2) == b.id
    assert b2.request["allocation"] == {
        "lineage": b.id, "champion": a.id, "effort": 2, "checkpoint": 2, "projected": True
    }
    assert [p.action for p in v15.programs.values()][3] == "Mature"  # A develops under V15


def test_a_slow_challenger_pauses_and_without_projection_the_search_is_v15(tmp_path):
    slow = [response(v) for v in (5, 3, 3.2, 7, 8)]
    method = search(tmp_path / "slow", *slow)
    method.run()
    a, _, _, a1 = list(method.programs.values())[:4]
    assert (a1.action, a1.parent_id) == ("Mature", a.id)  # -3.2 - 0.2 * 2 = -3.6 does not reach -5
    answers = [response(v) for v in (5, 3, 4, 7, 8, 6, 2, 9)]
    v16 = search(tmp_path / "off", *answers, budget=8, projection=False, changes=False, foreign_reference=False)
    v15 = search(tmp_path / "v15", *answers, budget=8, kind=TraceAADV15, config=V15Config)
    v16.run()
    v15.run()
    assert [c[0] for c in v16.llm.calls] == [c[0] for c in v15.llm.calls]
    assert [(p.action, p.parent_id) for p in v16.programs.values()] == [
        (p.action, p.parent_id) for p in v15.programs.values()
    ]


def test_the_edited_lineage_is_shown_by_changes_and_whole_programs_by_design():
    programs = {}
    for id, score, fields in [
        (1, -1, {"idea": "root rule"}),
        (2, -2, {"parent_id": 1, "action": "Refine", "change": "change 2"}),
        (3, None, {"parent_id": 2, "action": "Explore", "change": "change 3"}),
        (4, -5, {"parent_id": 3, "action": "Repair", "repair_of": 3, "change": "change 3"}),
        (5, -1, {"idea": "other rule"}),
    ]:
        programs[id] = Program(
            id=id,
            generation=id,
            code=f"def score(x): return {id}",
            analysis=None,
            fitness=score,
            status="valid" if score is not None else "runtime_error",
            **{"action": "Init", "idea": f"whole {id}", **fields},
        )
    programs[3].failure = {"kind": "runtime_error", "error": "ValueError: bad output"}
    prompts = PromptBuilderV16(TokenLLM(), "test", TinyEvaluation(), programs, Config())
    refine = prompts.build("Refine", programs[2])["prompt"]
    assert "Design: root rule" in refine and "Change: change 2" in refine and "Change: change 3" in refine
    assert refine.count("whole 2") == 1 and refine.index("whole 2") > refine.index("[Current Algorithm]")
    assert "whole 3" not in refine and "already evaluated" in refine
    assert '"change"' in refine[refine.rindex("[Output Format]") :]
    crossover = prompts.build("Crossover", programs[2], reference=programs[5])["prompt"]
    assert "Design: other rule" in crossover and "Keep the current algorithm's main decision rule" in crossover
    assert crossover.count("other rule") == 1  # an initial reference is described once
    assert '"change"' not in prompts.initial([programs[1]])["prompt"]


def test_a_reply_states_its_change_and_a_repair_keeps_it(tmp_path):
    def reply(value, change):
        code = f"def score(x):\n    return {value}\n"
        return json.dumps({"analysis": "", "code": code, "design": f"rule {value}", "change": change})

    broken = json.dumps({"analysis": "", "code": "def score(x):\n    return x[0]\n", "design": "rule", "change": "index"})
    fixed = json.dumps({"analysis": "", "code": "def score(x):\n    return 6\n"})
    method = search(tmp_path, response(5), broken, fixed, reply(7, "return 7"), budget=4, roots=1)
    method.run()
    root, failed, repaired, child = method.programs.values()
    assert root.change == "" and failed.change == repaired.change == "index"
    assert child.change == "return 7"
    assert "Change: index" in method.llm.calls[3][0]


def test_a_crossover_reference_comes_from_another_lineage(tmp_path):
    answers = [response(v) for v in (5, 3, 4, 6, 7, 8, 9)]
    method = search(
        tmp_path, *answers, budget=7, roots=2, allocation="fitness", operators={"Crossover": 1.0}
    )
    method.run()
    children = [p for p in method.programs.values() if p.action == "Crossover"]
    assert len(children) == 5
    for child in children:
        reference = method.programs[child.reference_id]
        assert method._lineage_of(reference) != method._lineage_of(method.programs[child.parent_id])
