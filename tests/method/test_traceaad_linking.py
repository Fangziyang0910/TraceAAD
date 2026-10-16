"""V12.0: Explore proposals linked to the best earlier implementation of the same computation."""

import json
import zlib
from dataclasses import replace
from traceaad.v12_1 import Program
import pytest
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v12_1 import Config, TraceAADV121
from functools import partial

BaseConfig = partial(Config, linking=False)
from traceaad.v12_1.linking import cut_proposal, parse_match, rank_candidates


def embed(text):
    vector = [0.0] * 64
    for word in text.lower().split():
        vector[zlib.crc32(word.encode()) % 64] += 1.0
    return vector


def proposal(design, value=None):
    return json.dumps(
        {
            "analysis": "weak spot",
            "design": design,
            **(
                {"code": f"def score(x):\n    return {value}\n"}
                if value is not None
                else {}
            ),
        }
    )


def code(value):
    return json.dumps({"code": f"def score(x):\n    return {value}\n"})


def same(*ids, analysis=""):
    return json.dumps({"analysis": analysis, "same": list(ids)})


def method(tmp_path, *answers, config=Config, embedder=embed, **options):
    return TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=tmp_path,
        config=config(**{"budget": 10, "roots": 1, **options}),
        embed=embedder,
    )


def start(search):
    search._roots()
    search._roots()
    return search.archive[1]


def explore(search, parent):
    request = search.prompts.build("Explore", parent)
    request["parent_id"] = parent.id
    return (search._create_program(request, parent=parent), request)


def test_base_arm_uses_one_call_without_linking(tmp_path):
    search = method(
        tmp_path, response(1), response(2), config=BaseConfig, embedder=None
    )
    parent = start(search)
    child, request = explore(search, parent)
    assert (
        len(search.llm.calls) == 2 and "link" not in request and (child.fitness == -2)
    )
    with pytest.raises(ValueError, match="embedding"):
        method(tmp_path / "other", config=Config, embedder=None)


def test_first_proposal_has_nothing_to_link_and_is_still_implemented(tmp_path):
    search = method(
        tmp_path, response(1), proposal("savings route builder", 9), code(2)
    )
    parent = start(search)
    child, request = explore(search, parent)
    assert child.fitness == -2 and child.idea == "savings route builder"
    propose, implement = search.llm.calls[1:]
    assert propose[0] == request["prompt"] and propose[1]["response_format"] == {
        "type": "json_object"
    }
    assert (
        "stop" not in propose[1]
        and '"code"' not in propose[0].split("[Output Format]")[1]
    )
    assert (
        "[Your Proposal]\nAnalysis: weak spot\nDesign: savings route builder"
        in implement[0]
    )
    assert "[An Earlier Program" not in implement[0] and "return 9" not in implement[0]
    assert request["link"]["candidates"] == [] and request["link"]["linked_id"] is None
    attempt = search.programs[2]
    assert [search.progress.model_calls, len(attempt.call_ids)] == [3, 2]


def test_repeated_proposal_is_implemented_with_the_best_matched_program(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal("savings route builder", 0),
        code(2),
        proposal("savings route builder with merges", 0),
        same(),
        code(3),
        proposal("savings route builder with local search", 0),
        same(3, 2, 99, analysis="The closest are [3] and [2]."),
        code(4),
    )
    parent = start(search)
    explore(search, parent)
    explore(search, parent)
    child, request = explore(search, parent)
    link = request["link"]
    assert set(link["candidates"]) == {2, 3} and link["matched"] == [2, 3]
    assert link["linked_id"] == 3 and link["shown"] and (link["display_draw"] < 1)
    match, implement = search.llm.calls[-2:]
    assert "[Design A]\nsavings route builder with local search" in match[0]
    assert (
        match[1]["temperature"] == 0.0
        and match[1]["max_tokens"] == search.config.output_tokens
    )
    assert (
        "[An Earlier Program for the Same Computation]" in implement[0]
        and "return 3" in implement[0]
    )
    assert (
        implement[0].index("[Your Proposal]")
        < implement[0].index("[An Earlier Program")
        < implement[0].index("[Your Task: Implement]")
    )
    assert child.fitness == -4 and search.programs[4].link["linked_id"] == 3
    saved = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert [c["stage"] for c in saved[-3:]] == ["propose", "match", "implement"]
    assert search.programs[4].call_ids == [c["request_id"] for c in saved[-3:]]


def test_display_draw_withholds_the_program_but_keeps_the_link(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal("savings route builder", 0),
        code(2),
        proposal("savings route builder again", 0),
        same(2),
        code(3),
        display_probability=0.0,
    )
    parent = start(search)
    explore(search, parent)
    _, request = explore(search, parent)
    assert request["link"]["linked_id"] == 2 and (not request["link"]["shown"])
    assert "[An Earlier Program" not in search.llm.calls[-1][0]


def test_embedding_failure_leaves_the_proposal_unlinked(tmp_path):

    def broken(text):
        raise ConnectionError("service down")

    search = method(
        tmp_path,
        response(1),
        proposal("a", 0),
        code(2),
        proposal("a b", 0),
        code(3),
        embedder=broken,
    )
    parent = start(search)
    explore(search, parent)
    child, request = explore(search, parent)
    assert child.fitness == -3 and request["link"]["error"].startswith(
        "ConnectionError"
    )
    assert request["link"]["matched"] == []


def test_implementation_keeps_the_earlier_program_and_its_own_material(tmp_path):
    search = method(tmp_path, response(1))
    parent = start(search)
    base = search.prompts.build("Explore", parent)
    earlier = replace(
        parent,
        id=7,
        idea="other",
        code="def score(x):\n" + "    y = 1\n" * 23000 + "    return y\n",
    )
    result = search.prompts.implement(base["prompt"], "Design: short", earlier)
    assert earlier.code.rstrip() in result["prompt"]
    assert base["prompt"].split("[Output Format]")[0].rstrip() in result["prompt"]
    assert result["input_tokens"] > 65536


def test_ranking_parsing_and_cutting():
    programs = {
        i: Program(
            id=i,
            generation=i,
            code="",
            idea=idea,
            analysis="",
            action="Init",
            status="valid",
            fitness=f,
        )
        for i, f, idea in [
            (1, 0.0, "x"),
            (2, 5.0, "savings a"),
            (3, 4.0, "savings a"),
            (4, 1.0, "lookahead"),
        ]
    }
    programs[5] = Program(
        id=5,
        generation=5,
        code="",
        idea="savings b",
        analysis="",
        action="Init",
        status="invalid_source",
    )
    ranked = rank_candidates("savings b", programs, 1, embed, 2)
    assert [p.id for p, _ in ranked] == [3, 4]
    assert parse_match(same(), [2]) == [] and parse_match("Same: 2", [2]) == []
    assert parse_match(same(2, 7, True, "5"), [2, 5]) == [2]
    assert (
        cut_proposal("Analysis: a\nDesign: b\n```python\nx\n```")
        == "Analysis: a\nDesign: b"
    )


def test_resume_requires_the_same_arm(tmp_path):
    search = method(tmp_path, response(1))
    start(search)
    with pytest.raises(ValueError, match="V12.1 resume"):
        method(tmp_path, config=BaseConfig, embedder=None)


@pytest.mark.parametrize(
    "implementation",
    [
        json.dumps(
            {"design": "unrelated lookahead", "code": "def score(x):\n    return 2\n"}
        ),
        "def score(x):\n    return 2\n",
        'Code:\n```python\ndef score(x):\n    note = """\nDesign: text inside code\n"""\n    return 2\n```',
    ],
)
def test_proposal_design_survives_implementation_text(tmp_path, implementation):
    search = method(
        tmp_path, response(1), proposal("savings route builder", 0), implementation
    )
    parent = start(search)
    child, request = explore(search, parent)
    assert child.idea == request["link"]["design"] == "savings route builder"
    assert child.fitness == -2
    assert search.programs[2].idea == child.idea
    resumed = method(tmp_path)
    assert resumed.archive[2].idea == child.idea


def test_invalid_implementation_keeps_proposal_design(tmp_path):
    search = method(
        tmp_path,
        response(1),
        proposal("savings", 0),
        "Design: unrelated\nCode:\n```python\ndef score(x):\n    return (\n```",
        budget=2,
    )
    parent = start(search)
    explore(search, parent)
    assert search.programs[2].idea == search.programs[2].idea == "savings"
    assert not search.programs[2].valid


def test_long_proposal_is_implemented_and_recorded(tmp_path):
    design = "long " * 66000 + "design end"
    search = method(tmp_path, response(1), proposal(design, 0), code(2))
    parent = start(search)
    search._ordinary_search("Explore")
    assert search.budget_used == 2 and search.progress.model_calls == 3
    assert design in search.llm.calls[-1][0]
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert calls[-1]["stage"] == "implement" and design in calls[-1]["prompt"]
    resumed = method(tmp_path)
    assert resumed.progress.model_calls == 3 and resumed.budget_used == 2
    assert resumed.progress.input_tokens == search.progress.input_tokens


def test_long_match_is_sent_and_its_program_is_used(tmp_path, monkeypatch):
    search = method(
        tmp_path,
        response(1),
        proposal("savings", 0),
        code(2),
        proposal("savings again", 0),
        same(2),
        code(3),
    )
    parent = start(search)
    explore(search, parent)
    match = "context " * 66000 + "match end"
    monkeypatch.setattr(
        search.prompts, "match", lambda *args: (match, search.prompts.count(match))
    )
    child, request = explore(search, parent)
    assert child.fitness == -3 and request["link"]["error"] is None
    assert request["link"]["shown"] and search.llm.calls[-2][0] == match
    assert "return 2" in search.llm.calls[-1][0]
    assert len(search.programs[3].call_ids) == 3


def test_resume_ignores_legacy_mechanism_label(tmp_path):
    search = method(tmp_path, response(1))
    start(search)
    checkpoint = tmp_path / "resume.json"
    saved = json.loads(checkpoint.read_text())
    saved["state"]["identity"]["mechanism"] = "legacy-label"
    checkpoint.write_text(json.dumps(saved))
    resumed = method(tmp_path)
    assert resumed.budget_used == search.budget_used
    assert "mechanism" not in resumed.identity
    resumed._save()
    assert "mechanism" not in json.loads(checkpoint.read_text())["state"]["identity"]


@pytest.mark.parametrize("stage", ["match", "implement"])
@pytest.mark.parametrize("transient", [False, True])
def test_stage_failure_preserves_prior_calls_and_costs(
    tmp_path, monkeypatch, stage, transient
):
    import core.llm as model

    search = method(
        tmp_path,
        response(1),
        proposal("savings", 0),
        code(2),
        proposal("savings again", 0),
        same(2),
    )
    parent = start(search)
    explore(search, parent)
    original = search.llm.draw_sample_with_details
    fail_at = len(search.llm.calls) + (2 if stage == "match" else 3)

    def draw(prompt, **kwargs):
        if len(search.llm.calls) + 1 == fail_at:
            raise ConnectionError("offline") if transient else ValueError("bad request")
        return {
            **original(prompt, **kwargs),
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    monkeypatch.setattr(search.llm, "draw_sample_with_details", draw)
    monkeypatch.setattr(model, "SERVICE_WAIT_SECONDS", 0)
    with pytest.raises(RuntimeError if transient else model.ModelCallError):
        explore(search, parent)
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    stages = ["propose", "match"] + (["implement"] if stage == "implement" else [])
    assert [c["stage"] for c in calls[3:]] == stages
    assert calls[-1]["error"] and all((not c["error"] for c in calls[:-1]))
    assert [c["request_id"] for c in calls] == list(range(1, len(calls) + 1))
    assert "stage" not in calls[0]
    resumed = method(tmp_path)
    assert resumed.budget_used == 2 and resumed.progress.service_failures == 1
    assert resumed.progress.model_calls == len(calls)
    assert resumed.progress.output_tokens == 3 * search.config.output_tokens + 3 * (
        len(stages) - 1
    )
    assert resumed.progress.input_tokens == search.progress.input_tokens
