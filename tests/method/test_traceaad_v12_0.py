"""V12.0: Explore proposals linked to the best earlier implementation of the same computation."""

import json
import zlib

import pytest

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v10_21.prompts import PromptBuilder as V1021Prompts
from traceaad.v12_0 import BaseConfig, Config, TraceAADV120
from traceaad.v12_0.linking import cut_proposal, parse_match, rank_candidates


def embed(text):
    vector = [0.0] * 64
    for word in text.lower().split():
        vector[zlib.crc32(word.encode()) % 64] += 1.0
    return vector


def proposal(design, value):
    # The fake model ignores stop sequences; the search cuts the reply before the code.
    return f"Analysis: weak spot\nDesign: {design}\nCode:\n```python\ndef score(x):\n    return {value}\n```"


def code(value):
    return f"Code:\n```python\ndef score(x):\n    return {value}\n```"


def method(tmp_path, *answers, config=Config, embedder=embed, **options):
    return TraceAADV120(evaluation=TinyEvaluation(), llm=TokenLLM(*answers), run_dir=tmp_path,
                        config=config(**{"budget": 10, "roots": 1, **options}), embed=embedder)


def start(search):
    search._roots()
    search._roots()
    return search.archive[1]


def explore(search, parent):
    request = search.prompts.build("Explore", parent)
    request["parent_id"] = parent["id"]
    return search._attempt(request, parent=parent), request


def test_base_arm_is_the_v1021_explore(tmp_path):
    search = method(tmp_path, response(1), response(2), config=BaseConfig, embedder=None)
    parent = start(search)
    baseline = V1021Prompts(search.llm, search.task, search.training.evaluation,
                            search.programs, search.attempts_table, search.config)
    assert search.prompts.build("Explore", parent) == baseline.build("Explore", parent)
    child, request = explore(search, parent)
    assert len(search.llm.calls) == 2 and "link" not in request and child["fitness"] == -2
    with pytest.raises(ValueError, match="embedding"):
        method(tmp_path / "other", config=Config, embedder=None)


def test_first_proposal_has_nothing_to_link_and_is_still_implemented(tmp_path):
    search = method(tmp_path, response(1), proposal("savings route builder", 9), code(2))
    parent = start(search)
    child, request = explore(search, parent)
    assert child["fitness"] == -2 and child["idea"] == "savings route builder"
    propose, implement = search.llm.calls[1:]
    assert propose[0] == request["prompt"] and propose[1]["stop"] == ["```", "\nCode:"]
    assert "[Your Proposal]\nAnalysis: weak spot\nDesign: savings route builder" in implement[0]
    assert "[An Earlier Program" not in implement[0] and "return 9" not in implement[0]
    assert request["link"]["candidates"] == [] and request["link"]["linked_id"] is None
    attempt = search.attempts_table[2]
    assert [search.progress.model_calls, len(attempt["call_ids"])] == [3, 2]


def test_repeated_proposal_is_implemented_with_the_best_matched_program(tmp_path):
    search = method(tmp_path, response(1), proposal("savings route builder", 0), code(2),
                    proposal("savings route builder with merges", 0), "Same: none", code(3),
                    proposal("savings route builder with local search", 0), "The closest are [3] and [2].\n**Same: 3, 2, 99**", code(4))
    parent = start(search)
    explore(search, parent)
    explore(search, parent)
    child, request = explore(search, parent)
    link = request["link"]
    # Programs 2 and 3 share the computation; 3 scores better; the current program is never listed.
    assert set(link["candidates"]) == {2, 3} and link["matched"] == [2, 3]
    assert link["linked_id"] == 3 and link["shown"] and link["display_draw"] < 1
    match, implement = search.llm.calls[-2:]
    assert "[Design A]\nsavings route builder with local search" in match[0]
    assert match[1]["temperature"] == 0.0 and match[1]["max_tokens"] == search.config.match_output_tokens
    assert "[An Earlier Program for the Same Computation]" in implement[0] and "return 3" in implement[0]
    assert implement[0].index("[Your Proposal]") < implement[0].index("[An Earlier Program") \
        < implement[0].index("[Your Task: Implement]")
    assert child["fitness"] == -4 and search.attempts_table[4]["link"]["linked_id"] == 3
    saved = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [c["stage"] for c in saved[-3:]] == ["propose", "match", "implement"]
    assert search.attempts_table[4]["call_ids"] == [c["request_id"] for c in saved[-3:]]


def test_display_draw_withholds_the_program_but_keeps_the_link(tmp_path):
    search = method(tmp_path, response(1), proposal("savings route builder", 0), code(2),
                    proposal("savings route builder again", 0), "Same: 2", code(3), display_probability=0.0)
    parent = start(search)
    explore(search, parent)
    _, request = explore(search, parent)
    assert request["link"]["linked_id"] == 2 and not request["link"]["shown"]
    assert "[An Earlier Program" not in search.llm.calls[-1][0]


def test_embedding_failure_leaves_the_proposal_unlinked(tmp_path):
    def broken(text):
        raise ConnectionError("service down")
    search = method(tmp_path, response(1), proposal("a", 0), code(2), proposal("a b", 0), code(3),
                    embedder=broken)
    parent = start(search)
    explore(search, parent)
    child, request = explore(search, parent)
    assert child["fitness"] == -3 and request["link"]["error"].startswith("ConnectionError")
    assert request["link"]["matched"] == []


def test_implementation_drops_the_earlier_program_before_its_own_material(tmp_path):
    search = method(tmp_path, response(1))
    parent = start(search)
    base = search.prompts.build("Explore", parent)
    earlier = {**parent, "id": 7, "idea": "other", "code": "def score(x):\n" + "    y = 1\n" * 400 + "    return y\n"}
    search.config.max_input_tokens = base["input_tokens"] + 40
    result = search.prompts.implement(base["prompt"], parent, "Design: short", earlier)
    assert result["trims"] == ["earlier_program"] and "y = 1" not in result["prompt"]
    assert result["input_tokens"] <= search.config.max_input_tokens


def test_ranking_parsing_and_cutting():
    programs = {i: {"id": i, "valid": True, "fitness": f, "idea": idea}
                for i, f, idea in [(1, 0.0, "x"), (2, 5.0, "savings a"), (3, 4.0, "savings a"), (4, 1.0, "lookahead")]}
    programs[5] = {"id": 5, "valid": False, "fitness": None, "idea": "savings b"}  # failed programs are not listed
    ranked = rank_candidates("savings b", programs, 1, embed, 2)
    assert [p["id"] for p, _ in ranked] == [3, 4]  # one entry per Design text, its best program
    assert parse_match("Same: none", [2]) == [] and parse_match("[2] looks alike", [2]) == []
    assert parse_match("Same: 5\n[5] differs.\nSame: 2, 7", [2, 5]) == [2]  # the last answer line counts
    assert cut_proposal("Analysis: a\nDesign: b\n```python\nx\n```") == "Analysis: a\nDesign: b"


def test_resume_requires_the_same_arm(tmp_path):
    search = method(tmp_path, response(1))
    start(search)
    with pytest.raises(ValueError, match="V12.0 resume"):
        method(tmp_path, config=BaseConfig, embedder=None)


def test_linking_report_summarizes_online_links_and_replays_base_runs(tmp_path):
    from experiments.traceaad_v12_0 import linking_report as report
    from traceaad.common.state import Facts
    online = method(tmp_path / "online", response(1), proposal("savings route builder", 0), code(2),
                    proposal("savings route builder again", 0), "Same: 2", code(3))
    parent = start(online)
    explore(online, parent)
    explore(online, parent)
    facts = Facts(tmp_path / "online")
    rows = [report.record(facts, a, a.get("link")) for a in report.proposals(facts)]
    summary = report.summary(rows)
    assert summary["proposals"] == 2 and summary["linked"] == 1
    assert rows[1]["linked"] == 2 and rows[1]["first_beats_parent"] and not rows[1]["copied_linked"]
    # E0: the base run's proposals are matched offline against the programs before each of them.
    base = method(tmp_path / "base", response(1), response(2), response(3), config=BaseConfig, embedder=None)
    parent = start(base)
    explore(base, parent)
    explore(base, parent)
    rows, cost = report.replay(tmp_path / "base", 20, TokenLLM("Same: 2"), embed)
    assert [r["candidates"] for r in rows] == [[], [2]] and rows[1]["linked"] == 2
    assert rows[1]["linked_beats_first"] is False and rows[1]["first_beats_linked"] and "match_input_tokens" in cost
    assert report.summary(rows)["first_beats_linked"] == {"all": 1.0, "shown": None, "hidden": 1.0}
