"""V21: Base's search with trajectory material and the analysis -> code -> design protocol."""
import ast
import json
import random

import pytest

from traceaad_base import Evaluation, Program, Task
import traceaad_base
import traceaad_v21_1
from traceaad_v21_1.context import COMMON, CROSSOVER, EXPLORE, FAILURES, REFINE, ancestors, attempts, card, context
from traceaad_v21_1.prompts import CANDIDATE_RESPONSE_FORMAT, COMMON_TEMPLATE, PromptBuilder
from traceaad_v21_1.representation import bare, changes

CONFIG = traceaad_base.Config()


def code(k, body="x"):
    return f"def solve(x):\n    a = {k}\n    return {body} + a\n"


def program(i, score, parent=0, *, source=None, operator="Refine", failure=None):
    evaluation = Evaluation("error", None, failure) if failure else Evaluation("success", score)
    return Program(i, source or code(i), f"design_{i}", operator, parent, None, evaluation, analysis=f"analysis_{i}")


def chain(n):
    archive = {1: program(1, 100., operator="Init")}
    for i in range(2, n + 1):
        archive[i] = program(i, 100. - i, i - 1)
    return archive


class Reply:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, **_):
        self.prompts.append(prompt)
        k = len(self.prompts)
        lines = "\n    ".join(f"v{i} = x + {i}" for i in range(k % 5 + 1))
        return json.dumps({"analysis": f"a{k}", "code": f"def solve(x):\n    {lines}\n    return {k % 7} + x\n",
                           "design": f"d{k}"})


def task():
    return Task("Return a finite score; lower is better.", "def solve(x):\n    return 1\n",
                lambda _, fn: float(fn(0)), score_definition="lower is better",
                entrypoint="solve", time_limit=5., evaluation_seed=1)


def test_output_order_puts_design_after_code():
    assert list(CANDIDATE_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]) == ["analysis", "code", "design"]
    assert COMMON_TEMPLATE.index('- "analysis"') < COMMON_TEMPLATE.index('- "code"') < COMMON_TEMPLATE.index('- "design"')


def test_real_clients_request_the_v21_field_order(tmp_path):
    class Client(Reply):
        request_options = {"temperature": 1.0}

    client = Client()
    traceaad_v21_1.TraceAAD(task(), client, config=traceaad_base.Config(evaluation_budget=1), run_dir=tmp_path / "run")
    assert client.request_options == {"temperature": 1.0, "response_format": CANDIDATE_RESPONSE_FORMAT}


def test_formation_keeps_recent_three_then_the_largest_score_moves():
    archive = chain(30)
    for i in range(6, 31):
        archive[i].evaluation = Evaluation("success", archive[i].evaluation.fitness - 50.)
    shown, path = ancestors(archive, archive[30], 8)
    ids = [p.id for p in shown]
    assert len(ids) == 8 and ids[-3:] == [27, 28, 29] and 6 in ids
    assert [p.id for p in path] == list(range(1, 30))
    text, _, _ = context(archive, archive[30], "Refine", None, CONFIG)
    assert "Omitted intermediate nodes before this record" in text


def test_attempts_merge_equal_source_and_show_best_plus_latest():
    archive = chain(2)
    archive[3] = program(3, 90., 2, source=code(77))
    archive[4] = program(4, 90., 2, source=code(77))
    archive[5] = program(5, 50., 2)
    archive[6] = program(6, 70., 2)
    archive[7] = program(7, 80., 2)
    archive[8] = program(8, None, 2, failure="ValueError: bad")
    archive[9] = program(9, None, 2, failure="ValueError: worse")
    archive[10] = program(10, None, 2, failure="ValueError: last")
    children, failures = attempts(archive, archive[2], CONFIG)
    assert [p.id for p in children] == [5, 6, 7]
    assert [p.id for p in failures] == [9, 10]


COMMENTED = 'def f(x):\n    """Doc."""\n    # why\n    y = x + 1  # note\n    return y\n'


def test_bare_source_drops_comments_and_docstrings_and_keeps_the_computation():
    text = bare(COMMENTED)
    assert "Doc" not in text and "why" not in text and "note" not in text
    assert ast.dump(ast.parse(text)) == ast.dump(ast.parse("def f(x):\n    y = x + 1\n    return y\n"))
    assert "pass" in bare('def f(x):\n    """Only."""\n')
    assert bare("def f(:\n") == "def f(:\n"


def test_small_changes_show_changed_lines_only():
    filler = "\n".join(f"    k{i} = {i}" for i in range(30))
    before = f"def f(x):\n{filler}\n    b = 2\n    return b\n"
    after = f"def f(x):\n{filler}\n    b = 3\n    return b\n"
    text = changes(before, after)
    assert "-    b = 2" in text and "+    b = 3" in text
    assert "k5 = 5" not in text and "k29 = 29" in text


def test_a_rewrite_is_stated_without_copying_either_program():
    before = "def f(x):\n    a = 1\n    b = a + x\n    return b\n"
    after = "def f(x):\n    for i in range(3):\n        x = x * 2\n    return sorted([x])[0]\n"
    text = changes(before, after)
    assert "most of the code was replaced" in text and "sorted" not in text and "a + x" not in text


def test_refine_shows_what_changed_on_offspring_only_and_the_current_code_undescribed():
    archive = chain(4)
    archive[5] = program(5, 90., 4)
    text, _, _ = context(archive, archive[4], "Refine", None, CONFIG)
    ancestors_part = text[text.index("[Ancestor History]"):text.index("[Evaluated Offspring]")]
    offspring_part = text[text.index("[Evaluated Offspring]"):text.index("[Current Algorithm]")]
    current_part = text[text.index("[Current Algorithm]"):]
    assert "Changed computation" not in ancestors_part and "Changed computation" in offspring_part
    for i in (1, 2, 3):
        assert f"design_{i}" in ancestors_part and f"analysis_{i}" in ancestors_part
    assert "design_5" in offspring_part and "analysis_5" in offspring_part
    assert "design_4" not in current_part and "analysis_4" not in current_part
    assert "Score: 96.0" in current_part and "def solve" in current_part


def test_identical_code_is_stated_as_a_fact():
    archive = chain(2)
    archive[3] = program(3, 98., 2, source=archive[2].code)
    text = card(archive, archive[3], change=True)
    assert "none; the code is identical to Algorithm 2" in text
    assert "Result relative to Algorithm 2: 98.0 -> 98.0; tied" in text


def test_explore_gets_only_the_current_code():
    archive = chain(4)
    archive[5] = program(5, 90., 4)
    archive[6] = program(6, None, 4, failure="ValueError: bad")
    text, instruction, view = context(archive, archive[4], "Explore", None, CONFIG)
    assert text.startswith("[Current Algorithm]") and "def solve" in text
    for word in ("Score", "Design", "Proposed", "Offspring", "Ancestor", "96.0", "design_", "analysis_"):
        assert word not in text
    assert instruction == EXPLORE
    assert view == {"ancestor_path": [], "ancestors": [], "children": [], "failed": []}


def test_crossover_has_no_ancestors_and_the_reference_carries_its_last_change():
    archive = chain(4)
    archive[5] = program(5, 90., 4)
    archive[6] = program(6, 95., 2, source="def solve(x):\n    a = 9\n    return x + a\n")
    text, _, view = context(archive, archive[4], "Crossover", archive[6], CONFIG)
    assert "[Ancestor History]" not in text and view["ancestors"] == []
    reference_part = text[text.index("[Reference Algorithm]"):text.index("[Current Algorithm]")]
    assert "Result relative to Algorithm 2: 98.0 -> 95.0; improved" in reference_part
    assert "Changed computation" in reference_part and "+    a = 9" in reference_part
    assert "Changed computation" in text[text.index("[Evaluated Offspring]"):text.index("[Reference Algorithm]")]


def test_explore_prompt_has_no_score_definition(tmp_path):
    model = Reply()
    scored = Task("Return a finite score.", "def solve(x):\n    return 1\n", lambda _, fn: float(fn(0)),
                  score_definition="lower is better", entrypoint="solve", time_limit=5., evaluation_seed=1)
    run = traceaad_v21_1.TraceAAD(scored, model, config=traceaad_base.Config(evaluation_budget=24, initial_size=3, seed=3),
                                run_dir=tmp_path / "run")
    run.run()
    explore = [p for p in model.prompts if EXPLORE in p]
    refine = [p for p in model.prompts if REFINE in p]
    assert explore and refine
    assert all("Score definition" not in p and "[Evaluated Offspring]" not in p for p in explore)
    assert all("Score definition" in p for p in refine)


def test_prompt_code_is_bare_and_the_record_stays_raw():
    archive = chain(2)
    archive[2].code = COMMENTED
    archive[3] = program(3, 1., operator="Init", source=COMMENTED.replace("f(x)", "g(x)").replace("+ 1", "* 2"))
    text, _, _ = context(archive, archive[2], "Crossover", archive[3], CONFIG)
    assert "# why" not in text and "Doc." not in text and "y = x + 1" in text and "y = x * 2" in text
    assert "# why" in archive[2].code


def test_initialization_shows_bare_code_and_repair_shows_raw_code():
    archive = {1: program(1, 5., operator="Init", source=COMMENTED)}
    builder = PromptBuilder(task(), CONFIG)
    assert "# why" not in builder.initialization(archive)
    assert "# why" in archive[1].code
    failed = program(2, None, 1, source=COMMENTED, failure="ValueError: bad")
    assert "# why" in builder.repair_program(failed)


def test_output_limits_both_descriptions_as_base_does():
    assert COMMON_TEMPLATE.count("using at most 100 English words or 200 Chinese characters") == 2


def test_reference_keeps_constant_variants_and_matches_base_random_selection():
    archive = {1: program(1, 5., operator="Init", source=code(1)), 2: program(2, 4., 1, source=code(2))}
    revised = traceaad_v21_1.TraceAAD(task(), Reply())
    revised.archive = archive
    assert revised.reference(archive[1]).id == 2
    archive = {i: program(i, float(i), operator="Init", source=code(i)) for i in range(1, 21)}
    base = traceaad_base.TraceAAD(task(), Reply())
    revised.archive = base.archive = archive
    picks = set()
    for seed in range(200):
        revised.rng = random.Random(seed)
        base.rng = random.Random(seed)
        actual = revised.reference(archive[20]).id
        assert actual == base.reference(archive[20]).id
        picks.add(actual)
    assert picks == set(range(1, 11))


def test_each_operator_gets_instructions_that_agree_with_it():
    archive = chain(2)
    _, instruction, _ = context({1: archive[1]}, archive[1], "Refine", None, CONFIG)
    assert instruction == "\n".join([COMMON, REFINE])
    archive[3] = program(3, 90., 2)
    archive[4] = program(4, None, 2, failure="ValueError: bad")
    _, instruction, _ = context(archive, archive[2], "Refine", None, CONFIG)
    assert instruction == "\n".join([COMMON, REFINE, FAILURES])
    _, instruction, _ = context(archive, archive[2], "Explore", None, CONFIG)
    assert instruction == EXPLORE
    _, instruction, _ = context(archive, archive[2], "Crossover", archive[1], CONFIG)
    assert instruction == "\n".join([COMMON, CROSSOVER, FAILURES])
    for text in (COMMON, REFINE, EXPLORE, CROSSOVER, FAILURES):
        for word in ("TSP", "route", "tour", "2-opt", "schedule", "colour", "pheromone", "time limit", "efficient"):
            assert word not in text


def test_failure_cards_hide_the_limit():
    archive = chain(2)
    archive[3] = Program(3, code(3), "d", "Refine", 2, None,
                         Evaluation("timeout", None, "Evaluation exceeded the total time limit of 160 seconds."), analysis="a")
    text, _, _ = context(archive, archive[2], "Refine", None, CONFIG)
    assert "160" not in text and "did not finish within the allowed time" in text


def test_search_uses_base_budget_and_resumes(tmp_path):
    config = traceaad_base.Config(evaluation_budget=24, initial_size=3, seed=3)
    model = Reply()
    v21 = traceaad_v21_1.TraceAAD(task(), model, config=config, run_dir=tmp_path / "v21")
    result = v21.run()
    assert result.evaluations == 24
    assert any("[Current Algorithm]" in prompt and "Proposed change" in prompt for prompt in model.prompts)
    crossover = [a for a in v21.attempts if a.operator == "Crossover"]
    assert crossover and all(a.reference_id is not None for a in crossover)
    resumed = traceaad_v21_1.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "v21", resume=True)
    assert resumed.evaluations_used == 24 and set(resumed.archive) == set(v21.archive)


@pytest.mark.parametrize("change", ["method", "reference_rule", "generation_protocol", "mechanisms"])
def test_changed_combination_is_rejected_before_log_repair(tmp_path, change):
    config = traceaad_base.Config(evaluation_budget=3, initial_size=2)
    search = traceaad_v21_1.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path)
    search.step()
    saved = search.store.read("run_config.json")
    saved[change] = "old-rule" if change != "mechanisms" else []
    search.store.write("run_config.json", saved)
    before = {}
    for name in ("calls.jsonl", "attempts.jsonl"):
        p = tmp_path / name
        p.write_bytes(p.read_bytes() + b'{"unfinished":')
        before[name] = p.read_bytes()
    with pytest.raises(ValueError, match=change):
        traceaad_v21_1.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path, resume=True)
    assert all((tmp_path / name).read_bytes() == data for name, data in before.items())
