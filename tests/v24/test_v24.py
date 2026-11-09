"""V24: V21 plus mechanism 3 (design notes revised every 10 evaluated attempts)."""
import json

from traceaad_base import Task
import traceaad_base
import traceaad_v24
from traceaad_v24.notes import INTERVAL, NEW_BEST, REFLECTION_FORMAT, attempt_card, parse, reflection_prompt

REFLECT = "You keep the design notes of this search."


class Reply:
    """Candidate replies for generation, note replies for reflection."""

    def __init__(self, fail_reflection=False):
        self.prompts, self.reflections = [], []
        self.fail_reflection = fail_reflection

    def complete(self, prompt, **_):
        if REFLECT in prompt:
            self.reflections.append(prompt)
            if self.fail_reflection:
                return "not json"
            k = len(self.reflections)
            return json.dumps({"analysis": f"r{k}", "notes": [{"idea": f"idea {k}.{i}", "lesson": f"lesson {k}.{i}"}
                                                               for i in range(k % 3 + 1)]})
        self.prompts.append(prompt)
        k = len(self.prompts)
        lines = "\n    ".join(f"v{i} = x + {i}" for i in range(k % 5 + 1))
        return json.dumps({"analysis": f"a{k}", "code": f"def solve(x):\n    {lines}\n    return {k % 7} + x\n",
                           "design": f"d{k}"})


def task(definition="lower is better"):
    return Task("Return a finite score.", "def solve(x):\n    return 1\n",
                lambda _, fn: float(fn(0)), score_definition=definition,
                entrypoint="solve", time_limit=5., evaluation_seed=1)


def search(tmp_path, model, budget=45, **kw):
    config = traceaad_base.Config(evaluation_budget=budget, initial_size=3, seed=2)
    return traceaad_v24.TraceAAD(task(), model, config=config, run_dir=tmp_path / "v24", **kw)


def counted(run):
    return [a for a in run.attempts if a.operator != "Init" and a.evaluations]


def test_reflection_reads_every_ten_evaluated_attempts_with_the_search_best(tmp_path):
    model = Reply()
    run = search(tmp_path, model)
    run.run()
    reflected = [a for a in run.attempts if "reflection" in a.record]
    assert reflected and len(model.reflections) == len(reflected)
    ids = [a.id for a in counted(run)]
    for n, attempt in enumerate(reflected):
        assert attempt.record["reflection"]["attempts"] == ids[n * INTERVAL:(n + 1) * INTERVAL]
    prompt = model.reflections[0]
    assert "Search best before" not in prompt
    assert prompt.count(NEW_BEST) <= INTERVAL
    assert "Implement the following" not in prompt and "No notes yet." in prompt
    assert "Score definition: lower is better" in prompt
    assert run.notes == reflected[-1].record["reflection"]["notes"]


def test_notes_lead_refine_and_crossover_context_and_skip_explore(tmp_path):
    model = Reply()
    run = search(tmp_path, model)
    run.run()
    first = next(a.id for a in run.attempts if "reflection" in a.record)
    later = [a for a in run.attempts if a.id > first and a.operator != "Init"]
    assert later
    by_attempt = {row["attempt_id"]: row["prompt"] for row in run.store.rows("calls.jsonl")
                  if row["stage"] == "generate"}
    for attempt in later:
        prompt = by_attempt[attempt.id]
        if attempt.operator == "Explore":
            assert "[Design Notes]" not in prompt and attempt.record["notes_from"] is None
        else:
            assert prompt.index("[Context]") < prompt.index("[Design Notes]") < prompt.index("[Current Algorithm]")
            assert attempt.record["notes_from"] is not None
    for attempt in run.attempts:
        if attempt.operator != "Init" and attempt.id <= first:
            assert "[Design Notes]" not in by_attempt[attempt.id]


def test_failed_reflection_keeps_notes_and_no_reflection_after_the_last_evaluation(tmp_path):
    model = Reply(fail_reflection=True)
    run = search(tmp_path, model, budget=3 + 2 * INTERVAL)
    run.run()
    assert run.notes == []
    assert all(a.record["reflection"]["status"] == "failed" for a in run.attempts if "reflection" in a.record)
    # The tenth attempt of the second window ends the budget, so it is not reflected.
    assert len(model.reflections) == 1


def test_resume_restores_notes_without_calling_the_model(tmp_path):
    model = Reply()
    run = search(tmp_path, model)
    run.run()
    again = Reply()
    resumed = search(tmp_path, again, resume=True)
    assert resumed.notes == run.notes and resumed.notes_from == run.notes_from
    assert resumed.pending == run.pending and again.prompts == again.reflections == []


def test_reflection_format_and_parse():
    assert list(REFLECTION_FORMAT["json_schema"]["schema"]["properties"]) == ["analysis", "notes"]
    good = '{"analysis": "x", "notes": [{"idea": "a", "lesson": "b "}, {"idea": " ", "lesson": ""}]}'
    assert parse(good) == ("x", [{"idea": "a", "lesson": "b"}])
    assert parse('{"analysis": "x", "notes": ["a"]}') is None
    assert parse('{"notes": []}') is None and parse("text") is None
    prompt = reflection_prompt(task("lower is better"), [{"idea": "i1", "lesson": "l1"}], ["card"])
    assert "1. Idea: i1\n   Lesson: l1" in prompt and "The latest 1 attempts" in prompt
    assert "worth using" not in prompt


def test_only_a_new_search_best_is_marked():
    from traceaad_base import Evaluation, Program
    archive = {1: Program(1, "def solve(x):\n    return 1\n", "d1", "Init", None, None, Evaluation("success", 5.)),
               2: Program(2, "def solve(x):\n    return 0\n", "d2", "Refine", 1, None, Evaluation("success", 4.),
                          analysis="a2")}
    assert NEW_BEST in attempt_card(archive, archive[2], 5.)
    assert NEW_BEST not in attempt_card(archive, archive[2], 3.)


def test_real_clients_get_a_separate_reflection_schema(tmp_path):
    class Client(Reply):
        request_options = {"temperature": 1.0}

        def for_role(self, schema):
            role = Client()
            role.request_options = {**self.request_options, "response_format": schema}
            return role

    run = search(tmp_path, Client(), budget=1)
    assert run.reflection_model.request_options["response_format"] == REFLECTION_FORMAT
    assert run.model.request_options["response_format"]["json_schema"]["name"] != "traceaad_design_notes"
