"""V12.1: an Explore version that is not a test of its Design is completed; one that does not carry out
its Design is not inherited."""

import json
import pytest
from tests.method.test_traceaad_linking import code, embed, proposal, start
from tests.support import TinyEvaluation, TokenLLM, response
from functools import partial
from traceaad.v12_1 import Config

BaseConfig = partial(Config, linking=False)
from traceaad.v12_1 import Config, DrawConfig, ResampleConfig, TraceAADV121
from traceaad.v12_1.verification import parse_dropped, parse_verdict

DESIGN = "savings route builder"
MERGE = "The program does not merge routes by savings."
DROP = "The program removes the start's distance rule."


def verdict(missing=""):
    return json.dumps({"computes": "It merges routes.", "missing": missing})


def contained(dropped=""):
    return json.dumps(
        {"computations": "The start ranks by distance.", "dropped": dropped}
    )


CARRIED, MISSING, KEPT, DROPPED = (
    verdict(),
    verdict(MERGE),
    contained(),
    contained(DROP),
)


def method(tmp_path, *answers, config=Config, **options):
    return TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=tmp_path,
        config=config(**{"budget": 10, "roots": 1, **options}),
        embed=embed,
    )


def explore(search, parent):
    request = search.prompts.build("Explore", parent)
    request.update(
        parent_id=parent.id,
        exploration={"id": len(search.facts.explorations) + 1, "step": 0},
    )
    return (search._create_program(request, parent=parent), request)


def versions(search, eid=1):
    return [
        (v["id"], v["implemented"], v["kept"])
        for v in search.facts.explorations[eid]["versions"]
    ]


@pytest.mark.parametrize(
    "text, reading, dropped",
    [
        (CARRIED, (True, None), (None, None)),
        (verdict("None."), (True, None), (None, None)),
        (MISSING, (False, MERGE), (None, None)),
        (contained('"none"'), (None, None), (True, None)),
        (DROPPED, (None, None), (False, DROP)),
        (
            json.dumps({"computations": ["ranks by distance: kept"], "dropped": ""}),
            (None, None),
            (True, None),
        ),
        ("Missing: x", (None, None), (None, None)),
    ],
)
def test_parse_reads_the_judgment_field(text, reading, dropped):
    assert parse_verdict(text) == reading and parse_dropped(text) == dropped


def test_a_reply_cut_at_the_output_limit_is_not_a_judgment():
    assert (
        TraceAADV121._judged(
            {
                "content": 'Reply "Dropped: none" or "Dropped: x"',
                "finish_reason": "length",
            }
        )
        == ""
    )
    assert TraceAADV121._judged({"content": KEPT, "finish_reason": "stop"}) == KEPT


def test_completion_replaces_random_development():
    with pytest.raises(TypeError, match="development_probability"):
        Config(development_probability=0.125)
    with pytest.raises(ValueError, match="completion"):
        Config(completion="again")


def test_version_missing_its_design_is_completed_in_place(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal(DESIGN, 9),
        code(0),
        MISSING,
        code(3),
        CARRIED,
        KEPT,
    )
    parent = start(search)
    explore(search, parent)
    search._search()
    verify, complete = search.llm.calls[3:5]
    assert verify[0].startswith(
        "[Task]\nReturn a numeric score.\n\n[Design]\nsavings route builder\n\n[Program]"
    )
    assert "return 0" in verify[0] and verify[1]["temperature"] == 0.0
    assert (
        "[Proposal]\nThe current algorithm was written for this proposal.\nAnalysis: weak spot\nDesign: "
        + DESIGN
        in complete[0]
    )
    assert (
        f"[Reading of the Code]\nRead against the Design: {MERGE}" in complete[0]
        and "return 0" in complete[0]
    )
    assert (
        "carries out the Design, and keep the rest of what it computes." in complete[0]
    )
    assert (
        "Design: <" not in complete[0]
        and "[An Earlier Program for This Proposal]" not in complete[0]
    )
    completed = search.archive[3]
    assert (
        completed.action == "Complete"
        and completed.parent_id == 2
        and (completed.idea == DESIGN)
    )
    search._search()
    contain = search.llm.calls[6][0]
    assert (
        "[Start Algorithm]\n```python\ndef score(x):\n    return 1" in contain
        and "return 3" in contain
    )
    record = search.facts.explorations[1]
    assert (
        record["developed"]
        and record["best_id"] == 3
        and (record["test_id"] == 3)
        and record["test_improved"]
    )
    assert versions(search) == [(2, False, None), (3, True, True)]
    refine = search.prompts.build("Refine", parent)["prompt"]
    assert (
        "; read against its Design: The program does not merge routes by savings; completed: score"
        in refine
    )
    formation = search.prompts.build("Refine", search.archive[3])["prompt"]
    assert f"Design: {DESIGN} (read against this Design: {MERGE})" in formation
    assert "Complete · score 0" in formation


def test_version_dropping_part_of_the_start_is_completed_in_the_start(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal(DESIGN, 9),
        code(0),
        CARRIED,
        DROPPED,
        code(3),
        CARRIED,
        KEPT,
    )
    parent = start(search)
    explore(search, parent)
    search._search()
    complete = search.llm.calls[5][0]
    assert (
        "[Current Algorithm]\nScore: -1" in complete
        and "This proposal was made for the current algorithm." in complete
    )
    assert (
        "[An Earlier Program for This Proposal]\nWritten from the current algorithm for this proposal.\nScore: 0"
        in complete
        and "return 0" in complete
    )
    assert f"from which it was written:\n{DROP}\n\n[Your Task: Complete]" in complete
    assert search.archive[3].parent_id == 1
    search._search()
    assert versions(search) == [(2, True, False), (3, True, True)]
    refine = search.prompts.build("Refine", parent)["prompt"]
    assert (
        "; besides its Design: The program removes the start's distance rule; completed: score"
        in refine
    )


def test_long_readings_and_completion_share_the_output_setting(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal(DESIGN, 9),
        code(0),
        CARRIED,
        DROPPED,
        code(3),
        CARRIED,
        KEPT,
        output_tokens=1234,
    )
    parent = start(search)
    search.prompts.common[0] += "\n" + "context " * 66000
    explore(search, parent)
    search._search()
    search._search()
    assert versions(search) == [(2, True, False), (3, True, True)]
    assert all(
        (search.prompts.common[0] in prompt for prompt, _ in search.llm.calls[1:])
    )
    assert {kwargs["max_tokens"] for _, kwargs in search.llm.calls} == {1234}
    assert search.facts.explorations[1]["test_improved"]


@pytest.mark.parametrize(
    "answers, value, test",
    [
        ((CARRIED, KEPT), 0, True),
        ((MISSING,), 2, False),
        ((CARRIED, DROPPED), 2, False),
    ],
)
def test_a_test_or_an_improving_version_is_not_completed(
    tmp_path, answers, value, test
):
    search = method(tmp_path, response(1), proposal(DESIGN, 9), code(value), *answers)
    parent = start(search)
    explore(search, parent)
    search._search()
    record = search.facts.explorations[1]
    assert not record["developed"] and len(search.llm.calls) == 3 + len(answers)
    assert record["test_id"] == (2 if test else None) and (not record["test_improved"])


def test_completion_stops_after_its_steps_and_chains_from_the_latest_version(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal(DESIGN, 9),
        code(0),
        MISSING,
        code(0.5),
        verdict("Still no merges."),
        code(0.25),
        MISSING,
    )
    parent = start(search)
    explore(search, parent)
    for _ in range(3):
        search._search()
    assert [search.programs[i].parent_id for i in (3, 4)] == [2, 3]
    assert "Read against the Design: Still no merges." in search.llm.calls[6][0]
    assert [v[1] for v in versions(search)] == [False, False, False]
    refine = search.prompts.build("Refine", parent)["prompt"]
    assert "still not carrying out the Design; completed again: score" in refine


def test_completion_with_identical_code_is_a_new_evaluated_individual(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal(DESIGN, 9),
        code(0),
        CARRIED,
        DROPPED,
        code(0),
        CARRIED,
        DROPPED,
        code(0),
        CARRIED,
        DROPPED,
    )
    parent = start(search)
    explore(search, parent)
    for _ in range(3):
        search._search()
    assert [search.programs[i].status for i in (2, 3, 4)] == ["valid"] * 3
    assert search.programs[2].code == search.programs[3].code == search.programs[4].code
    assert search.programs[2] is not search.programs[3]
    assert search.facts.explorations[1]["development_ids"] == [3, 4]
    assert len([e for e in search.facts.evaluations if e["role"] == "search"]) == 4


def test_resample_arm_sends_the_first_implementation_request_again(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal(DESIGN, 9),
        code(0),
        MISSING,
        code(3),
        CARRIED,
        KEPT,
        config=ResampleConfig,
    )
    parent = start(search)
    explore(search, parent)
    search._search()
    assert search.llm.calls[4][0] == search.llm.calls[2][0]
    assert (
        search.archive[3].parent_id == 1
        and search.programs[3].request["completion"] == "resample"
    )


def test_unfaithful_version_is_not_inherited(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal(DESIGN, 9),
        code(0),
        MISSING,
        proposal(DESIGN + " again", 9),
        code(2),
        completion_steps=0,
    )
    parent = start(search)
    explore(search, parent)
    search._search()
    assert search.programs[2].verdict["implemented"] is False
    assert search._implementations().keys() == {1} and search.archive.keys() == {1, 2}
    _, request = explore(search, parent)
    assert request["link"]["candidates"] == []


def test_randomized_arm_records_verdict_and_draw(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal(DESIGN, 9),
        code(0),
        MISSING,
        config=DrawConfig,
        completion_probability=0.0,
    )
    parent = start(search)
    explore(search, parent)
    search._search()
    record = search.facts.explorations[1]
    assert not record["developed"] and 0 <= record["completion_draw"] < 1
    assert versions(search) == [(2, False, None)]


def test_resume_reuses_saved_verdicts(tmp_path):
    search = method(
        tmp_path, response(1), proposal(DESIGN, 9), code(0), MISSING, code(3)
    )
    parent = start(search)
    explore(search, parent)
    search._search()
    resumed = method(tmp_path, CARRIED, KEPT)
    resumed._search()
    assert (
        len(resumed.llm.calls) == 2
        and resumed.programs[2].verdict["implemented"] is False
    )
    assert versions(resumed)[-1] == (3, True, True)


def test_verification_report_replays_earlier_runs_and_summarizes_completions(tmp_path):
    from experiments.traceaad_v12_1 import verification_report as report

    search = method(
        tmp_path / "online",
        response(1),
        proposal(DESIGN, 9),
        code(0),
        MISSING,
        code(3),
        CARRIED,
        KEPT,
    )
    parent = start(search)
    explore(search, parent)
    search._search()
    search._search()
    summary = report.summary(report.recorded(search.facts))
    assert summary["readings_if_worse"] == {"test": 0.0, "missing": 1.0, "dropped": 0.0}
    assert summary["completion"]["completed"] == {
        "explorations": 1,
        "ends_test": 1.0,
        "test_improved": 1.0,
        "best_beats_start": 1.0,
        "followed": 0.0,
    }
    base = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response(1), response(0)),
        run_dir=tmp_path / "base",
        config=BaseConfig(budget=10, roots=1),
    )
    parent = start(base)
    request = base.prompts.build("Explore", parent)
    request["parent_id"] = parent.id
    base._create_program(request, parent=parent)
    llm = TokenLLM(CARRIED, DROPPED)
    rows, cost = report.replay(base.facts, report.task_text(tmp_path / "base"), llm)
    assert llm.calls[0][0].startswith("[Task]\nReturn a numeric score.\n\n[Design]")
    assert llm.calls[1][0].startswith(
        "[Task]\nReturn a numeric score.\n\n[Start Algorithm]"
    )
    assert rows[0]["reading"] == "dropped" and "verify_input_tokens" in cost
    assert f"## Attempt 2 · dropped: {DROP}" in report.examples(base.facts, rows, 1)


class ReadingLLM(TokenLLM):
    """Answers by request: every version is worse than its start and never carries out its Design."""

    def __init__(self):
        super().__init__()
        self.made = 0

    def draw_sample_with_details(self, prompt, **kwargs):
        self.made += 1
        if "Does the program carry out" in prompt:
            answer = MISSING
        elif "Which earlier designs" in prompt:
            answer = json.dumps({"analysis": "", "same": []})
        elif '"code"' not in prompt:
            answer = proposal(f"{DESIGN} {self.made}", 9)
        else:
            answer = response(1) if self.made == 1 else code(-self.made)
        self.responses = iter([answer])
        return super().draw_sample_with_details(prompt, **kwargs)


def test_a_tiny_run_completes_each_exploration_twice(tmp_path):
    search = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=ReadingLLM(),
        run_dir=tmp_path,
        embed=embed,
        config=Config(budget=7, roots=1, operators={"Explore": 1}),
    )
    assert search.run()["status"] == "finished"
    actions = [a.action for a in sorted(search.programs.values(), key=lambda a: a.id)]
    assert actions == [
        "Init",
        "Explore",
        "Complete",
        "Complete",
        "Explore",
        "Complete",
        "Complete",
    ]
    assert [len(r["versions"]) for r in search.facts.explorations.values()] == [3, 3]
    assert {kwargs["max_tokens"] for _, kwargs in search.llm.calls} == {16384}


def test_completion_pairs_compare_arms_on_the_same_first_versions():
    from experiments.traceaad_v12_1.completion_pairs import summary

    def arm(beats, reading="test"):
        return {
            "status": "valid",
            "beats_start": beats,
            "reading": reading,
            "same_as": None,
        }

    rows = [
        {
            "task": "t",
            "reading": "missing",
            "first_beats_start": False,
            "develop_best_beats_start": True,
            "develop": arm(False, "missing"),
            "resample": arm(False, "missing"),
            "complete": arm(True),
        },
        {
            "task": "t",
            "reading": "test",
            "first_beats_start": False,
            "develop_best_beats_start": False,
            "develop": arm(False),
            "resample": arm(True),
        },
        {
            "task": "t",
            "reading": "test",
            "first_beats_start": True,
            "develop_best_beats_start": False,
        },
    ]
    result = summary(rows)["all"]
    assert result["readings_if_worse"] == {"test": 0.5, "missing": 0.5, "dropped": 0.0}
    assert result["worse_not_test"]["complete"]["test_beats_start"] == 1.0
    assert result["worse_test"]["resample"]["beats_start"] == 1.0
    assert result["paired_not_test"]["complete_vs_resample_beats_start"] == {
        "n": 1,
        "both": 0,
        "complete_only": 1,
        "resample_only": 0,
    }
    assert result["paired_not_test"]["complete_vs_resample_tests"] == {
        "n": 1,
        "both": 0,
        "complete_only": 1,
        "resample_only": 0,
    }


@pytest.mark.parametrize(
    "bad_proposal,finish",
    [
        ("{invalid json", "stop"),
        (
            json.dumps(
                {"analysis": "why", "design": "  ", "code": "def score(x): return 9"}
            ),
            "stop",
        ),
        (json.dumps({"analysis": "why", "design": "valid design"}), "length"),
        (json.dumps({"analysis": "why"}), "stop"),
    ],
)
def test_failed_proposal_stops_before_implementation_and_survives_resume(
    tmp_path, monkeypatch, bad_proposal, finish
):
    search = method(tmp_path, response(1), bad_proposal)
    start(search)
    original = search.llm.draw_sample_with_details
    monkeypatch.setattr(
        search.llm,
        "draw_sample_with_details",
        lambda *args, **kwargs: {**original(*args, **kwargs), "finish_reason": finish},
    )
    search._ordinary_search("Explore")
    assert len(search.programs) == 1 and search.budget_used == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert (
        events[-1]["kind"] == "delivery_failed"
        and "attempt" not in events[-1]
        and "program" not in events[-1]
    )
    assert events[-1]["call_ids"] == [2] and search.progress.repair_id is None
    resumed = method(tmp_path)
    assert resumed.budget_used == 2 and resumed._open_exploration() is None
    assert not resumed.llm.calls and len(resumed.programs) == 1


@pytest.mark.parametrize("action", ["Explore", "Complete"])
def test_invalid_code_keeps_explicit_design_and_analysis(tmp_path, action):
    design = "First line\nDesign: this is part of the design"
    bad = json.dumps(
        {
            "analysis": "completion analysis",
            "design": "ignore this",
            "code": "def score(x):\n    return (",
        }
    )
    answers = (
        [response(1)] + ([proposal(design)] if action == "Explore" else []) + [bad]
    )
    search = method(tmp_path, *answers, budget=2)
    parent = start(search)
    if action == "Explore":
        explore(search, parent)
        expected_design = " ".join(design.split())
        analysis = "weak spot"
    else:
        request = search.prompts._result(["Complete the design"], "Complete")
        request["design"] = design
        search._create_program(request, parent=parent)
        expected_design, analysis = (design, "completion analysis")
    attempt = search.programs[2]
    assert attempt.status == "invalid_source"
    assert attempt.idea == search.programs[2].idea == expected_design
    assert attempt.analysis == analysis
