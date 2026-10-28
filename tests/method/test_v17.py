"""V17: by default V16's search with V15's same-age comparison, where a sampled Crossover imports another lineage's
improving change. With the progress rule, a lineage other than the champion continues while it improves on its own
record, within the attempts the champion spends beyond the stagnation window, a stalled exploration starts a new
lineage first, and a paused lineage resumes by importing a new change of a better lineage."""

import json

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v16 import Config as V16Config, TraceAADV16
from traceaad.v17 import Config, TraceAADV17


def search(directory, *answers, kind=TraceAADV17, config=Config, **options):
    return kind(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=directory,
        config=config(**{"budget": 5, "roots": 2, "mature_steps": 1, "diverge": False, **options}),
    )


def reply(value, change, design):
    code = f"def score(x):\n    return {value}\n"
    return json.dumps({"analysis": "", "code": code, "design": design, "change": change})


def reasons(method):
    return [(p.request.get("allocation") or {}).get("reason") for p in method.programs.values()]


def test_a_lineage_improving_on_its_own_record_continues_when_exploration_opens(tmp_path):
    # Roots A (-5) and B (-3) are developed once each; A1 (-6) holds the best. A then spends the window of 2
    # attempts and two more without a new best, so exploration opens once. B improved at its last attempt and
    # continues, although at its checkpoint it is behind A, where V16 pauses it.
    answers = [response(v) for v in (5, 3, 6, 4, 5.4, 5.3, 5.2, 5.1, 4.5, 5.0)]
    method = search(
        tmp_path, *answers, budget=10, stagnation_window=2, progress_rule=True, imports=False, operators={"Refine": 1.0}
    )
    method.run()
    programs = list(method.programs.values())
    assert reasons(method)[2:] == ["young", "young", *["champion"] * 4, "active", "champion"]
    assert method._lineage_of(programs[8]) == programs[1].id
    assert programs[8].request["allocation"] == {
        "lineage": 2, "champion": 1, "effort": 2, "reason": "active", "stall": 0, "account": {"champion": 4, "other": 1}
    }


def test_a_paused_lineage_resumes_by_importing_a_new_change_of_a_better_lineage(tmp_path):
    # B's development makes it worse, and one more attempt without improvement pauses it. A then sets a new best
    # with a large gain; when exploration opens again, B imports that change into its best program.
    answers = [response(v) for v in (5, 3, 6, 2, 6.5, 6.4, 6.3, 2.5, 8, 7.9, 7.8, 4)]
    method = search(tmp_path, *answers, budget=12, stagnation_window=1, progress_rule=True, operators={"Refine": 1.0})
    method.run()
    programs = list(method.programs.values())
    assert reasons(method)[7] == "active" and reasons(method)[11] == "resume"
    resumed = programs[11]
    assert (resumed.action, resumed.parent_id, resumed.reference_id) == ("Import", 2, 9)
    assert [p.id for p in method._components()] == [9, 12]  # gains 0.23 and 0.33 reach the median 0.22
    assert method._summary("finished")["v17"]["resumed_improved"] == 1


def test_a_crossover_imports_a_component_once_and_a_repeated_reply_is_asked_again(tmp_path):
    # Same-age comparison (the default): B is developed first, then pauses behind A. B's development step gains
    # more than A's, so it is the only component. A's sampled Crossover imports it; the reply reproduces A1 and is
    # asked again as an Import. The next Crossover takes a reference design.
    answers = [
        reply(5, "", "design A"),
        reply(3, "", "design B"),
        reply(4, "B: return 3 becomes 4", "design B1"),
        reply(6, "A: return 6", "design A1"),
        reply(6, "imported", "design A2"),
        reply(7, "imported again", "design A3"),
        reply(8, "crossed", "design A4"),
    ]
    method = search(tmp_path, *answers, budget=7, stagnation_window=50, operators={"Crossover": 1.0})
    method.run()
    _, b, b1, _, rejected, retry, crossed = method.programs.values()
    assert (rejected.status, rejected.action, rejected.reference_id) == ("revisit", "Import", b1.id)
    assert (retry.action, retry.repair_of, retry.reference_id) == ("Import", rejected.id, b1.id)
    prompt = method.llm.calls[4][0]
    assert "[Component From Another Algorithm]" in prompt and "Change: B: return 3 becomes 4" in prompt
    assert "-    return 3\n+    return 4" in prompt and "Score of that algorithm: -3 → -4" in prompt
    assert "design B" not in prompt and '"change"' in prompt[prompt.rindex("[Output Format]") :]
    assert "[Your Task: Import]" in method.llm.calls[5][0] and "[Note]" in method.llm.calls[5][0]
    assert crossed.action == "Crossover" and method._lineage_of(method.programs[crossed.reference_id]) == b.id


def test_without_the_progress_rule_and_imports_the_search_is_v16(tmp_path):
    answers = [response(v) for v in (5, 3, 4, 7, 8, 6, 2, 9)]
    v17 = search(tmp_path / "v17", *answers, budget=8, progress_rule=False, imports=False, projection=True)
    v16 = search(tmp_path / "v16", *answers, budget=8, kind=TraceAADV16, config=V16Config)
    v17.run()
    v16.run()
    assert [c[0] for c in v17.llm.calls] == [c[0] for c in v16.llm.calls]
    assert [(p.action, p.parent_id, p.reference_id) for p in v17.programs.values()] == [
        (p.action, p.parent_id, p.reference_id) for p in v16.programs.values()
    ]


def test_without_imports_the_search_is_v16_without_projection(tmp_path):
    answers = [response(v) for v in (5, 3, 4, 7, 8, 6, 2, 9)]
    v17 = search(tmp_path / "v17", *answers, budget=8, imports=False)
    v16 = search(tmp_path / "v16", *answers, budget=8, kind=TraceAADV16, config=V16Config, projection=False)
    v17.run()
    v16.run()
    assert [c[0] for c in v17.llm.calls] == [c[0] for c in v16.llm.calls]
    assert [p.request.get("allocation") for p in v17.programs.values()] == [
        p.request.get("allocation") for p in v16.programs.values()
    ]


def test_a_stalled_exploration_diverges_first_and_otherwise_continues_an_active_lineage(tmp_path):
    # A1 (-6) holds the best. When exploration opens, the window of 4 has passed without a new best, so a new
    # lineage D starts. At the next exploration attempt the last Diverge is recent: D is developed and active,
    # and continues. One window after the Diverge, exploration starts another lineage although D is active.
    answers = [response(v) for v in (5, 6, 5.9, 5.8, 5.7, 5.6, 5.5, 4, 4.5, 5.4, 5.3, 4.6, 5.2, 3)]
    method = search(
        tmp_path, *answers, budget=14, roots=1, stagnation_window=4, diverge=True, progress_rule=True,
        operators={"Refine": 1.0},
    )
    method.run()
    assert [r or "init" for r in reasons(method)] == [
        "init", "young", *["champion"] * 5, "diverge", "young", "champion", "champion", "active", "champion", "diverge"
    ]
    new, again = method.programs[8], method.programs[14]
    assert new.action == again.action == "Diverge" and new.request["allocation"]["lineage"] is None
    assert method._lineage_of(method.programs[12]) == new.id
