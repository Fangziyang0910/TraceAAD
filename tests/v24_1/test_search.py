"""V24.1 evidence retention, replay, operator contracts and experiment entry."""

from argparse import Namespace
import json
import re
import shutil

import pytest

from traceaad_base import Config, Evaluation, Task
from traceaad_base.models import Attempt
from traceaad_v21_3.context import context as v21_context
from traceaad_v21_3.search import TraceAAD as V21
from traceaad_v24_1 import TraceAAD
from traceaad_v24_1.context import EXPLORE
from traceaad_v24_1.evidence import build
from traceaad_v24_1.notes import (CHAT_RESERVE, REFLECTION_FORMAT, REFLECTION_TOKENS,
                                  experience_block, reflection_prompt)


def task():
    return Task("Return a finite value.", "def solve(x):\n    return x", lambda _, fn: fn(0), 2.,
                score_definition="lower is better", metadata={"n_instance": 16, "settings": {}})


def evaluate(code):
    namespace = {}
    try:
        exec(code, namespace)
        return Evaluation("success", float(namespace["solve"](0)))
    except Exception as exc:
        return Evaluation("error", failure=f"{type(exc).__name__}: {exc}")


class Generator:
    def __init__(self, start=0, failures=()):
        self.n, self.failures, self.prompts = start, set(failures), []

    def complete(self, prompt, **settings):
        self.prompts.append(prompt)
        self.n += 1
        code = ("def solve(x):\n    raise ValueError('broken')\n" if self.n in self.failures
                else f"def solve(x):\n    k = {self.n}\n    return x + k % 7\n")
        return json.dumps({"analysis": "Change the computation.", "code": code,
                           "design": "Compute a finite value."})


class Reflector:
    def __init__(self, failures=(), make_note=False, failure_kind="parse"):
        self.failures, self.make_note, self.prompts = set(failures), make_note, []
        self.failure_kind = failure_kind
        self.settings = []

    def count_prompt_tokens(self, text):
        # A deterministic test tokenizer; production must provide its serving tokenizer.
        return len(text.encode())

    def complete(self, prompt, **settings):
        self.prompts.append(prompt)
        self.settings.append(settings)
        if len(self.prompts) in self.failures:
            if self.failure_kind == "request":
                raise TimeoutError("reflection request timed out")
            return "not json"
        updates = []
        if self.make_note:
            source = json.loads(re.search(r"New events: (\[[^\n]*\])", prompt)[1])[0]
            updates = [{"id": None if '"id": 1' not in prompt else 1, "idea": "Remainder scoring",
                        "when": "The returned value is computed using a remainder.",
                        "lesson": "Parameter changes can leave the ranking unchanged.", "evidence_ids": [source]}]
        return json.dumps({"analysis": "A computational condition became clear.",
                           "updates": updates, "archive_ids": []})


def search(path, generator=None, reflector=None, *, config=None, resume=False, **options):
    result = TraceAAD(task(), generator or Generator(), reflection_model=reflector or Reflector(),
                      config=config or Config(evaluation_budget=45, initial_size=3, seed=2),
                      run_dir=path, resume=resume, **options)
    result.evaluator.evaluate = evaluate
    return result


@pytest.mark.parametrize("failure_kind", ["parse", "request"])
def test_failed_reflection_retains_events_and_success_only_consumes_the_shown_question(tmp_path, failure_kind):
    reflector = Reflector(failures=[1], make_note=True, failure_kind=failure_kind)
    run = search(tmp_path, reflector=reflector)
    run.run()
    reflections = [a for a in run.attempts if "reflection" in a.record]
    assert [a.id for a in reflections] == [13, 23, 33, 43]
    failed, succeeded = [a.record["reflection"] for a in reflections[:2]]
    assert failed["status"] == "failed" and failed["snapshot"]["unreviewed_events"] == list(range(4, 14))
    assert succeeded["status"] == "success" and succeeded["unreviewed_before"] == list(range(4, 24))
    assert succeeded["reviewed_event_ids"]
    assert succeeded["snapshot"]["unreviewed_events"] == [
        i for i in succeeded["unreviewed_before"] if i not in succeeded["shown_event_ids"]]
    assert succeeded["focus"]["anchor_attempt"] in succeeded["shown_event_ids"]
    for attempt in reflections[1:]:
        reflection = attempt.record["reflection"]
        assert reflection["status"] == "success"
        for update in reflection["updates"]:
            assert set(update["evidence_ids"]) <= set(reflection["shown_event_ids"])


def test_repeated_unchanged_computations_stay_archived_without_forcing_a_knowledge_update(tmp_path):
    class Unchanged(Generator):
        def complete(self, prompt, **settings):
            self.n += 1
            if "[Current Algorithm]" not in prompt:  # Initialization: distinct programs, so Crossover has references.
                source = f"def solve(x):\n    return x + {self.n}\n"
            else:  # Evolution returns the current algorithm unchanged.
                source = prompt.split("[Current Algorithm]")[1].split("```python\n")[1].split("```")[0]
            return json.dumps({"analysis": "Keep the computation.", "code": source, "design": "Add one."})
    reflector = Reflector()
    run = search(tmp_path, generator=Unchanged(), reflector=reflector,
                 config=Config(evaluation_budget=26, initial_size=3))
    run.run()
    assert len(run.events) == 26 and run.unreviewed_events == []
    assert not reflector.prompts
    reflected = [a for a in run.attempts if "reflection" in a.record]
    assert [a.id for a in reflected] == [13, 23]
    assert all(a.record["reflection"]["status"] == "observed" for a in reflected)
    assert all(a.record["observation"]["reason"] == "unchanged_computation"
               for a in run.attempts if a.operator != "Init")


def test_resume_restores_cadence_unread_entries_and_sources_without_requests(tmp_path):
    config = Config(evaluation_budget=40, initial_size=3, seed=2)
    first = search(tmp_path / "first", reflector=Reflector(make_note=True), config=config)
    for _ in range(17):
        first.step()
    saved = first.snapshot()
    # A stale mirror is never used as the recovery authority.
    first.store.write("experience.json", {"incorrect": "uncommitted"})
    shutil.copytree(tmp_path / "first", tmp_path / "resumed")
    generator, reflector = Generator(start=first.model_calls), Reflector(make_note=True)
    resumed = search(tmp_path / "resumed", generator, reflector, config=config, resume=True)
    assert resumed.snapshot() == saved and generator.prompts == reflector.prompts == []
    assert resumed.store.read("experience.json") == saved
    first.run()
    resumed.run()
    assert first.snapshot() == resumed.snapshot()
    for a, b in zip(first.attempts, resumed.attempts, strict=True):
        assert {k: v for k, v in a.record.items() if k != "aux"} == {
            k: v for k, v in b.record.items() if k != "aux"}
        # Request duration is wall time, not recoverable search state.
        assert [{k: v for k, v in row.items() if k != "seconds"} for row in a.record.get("aux", [])] == [
            {k: v for k, v in row.items() if k != "seconds"} for row in b.record.get("aux", [])]


def test_whole_question_deferral_survives_restore(tmp_path):
    full = search(tmp_path / "full")
    for _ in range(13):
        full.step()
    unread = list(range(4, 14))
    material = build(full.events, unread, [], lambda related, text: reflection_prompt(
        task(), related, text), len, full.input_limit)
    size = material.input_size
    context_tokens = REFLECTION_TOKENS + CHAT_RESERVE + size
    limited = search(tmp_path / "limited", reflection_context_tokens=context_tokens)
    for _ in range(13):
        limited.step()
    reflection = limited.attempts[-1].record["reflection"]
    assert reflection["status"] == "success" and reflection["reviewed_event_ids"]
    assert reflection["deferred_event_ids"] and limited.unreviewed_events
    assert set(reflection["reviewed_event_ids"]).isdisjoint(limited.unreviewed_events)
    resumed = search(tmp_path / "limited", resume=True, reflection_context_tokens=context_tokens)
    assert resumed.snapshot() == limited.snapshot()


def test_no_fragment_fits_without_a_request_and_events_are_not_discarded(tmp_path):
    reflector = Reflector()
    config = Config(evaluation_budget=25, initial_size=3)
    run = search(tmp_path, reflector=reflector, config=config,
                 reflection_context_tokens=REFLECTION_TOKENS + CHAT_RESERVE + 1)
    run.run()
    assert not reflector.prompts and run.unreviewed_events == list(range(4, 26))
    assert all(a.record["reflection"]["status"] == "deferred"
               for a in run.attempts if "reflection" in a.record)


def test_repairs_count_once_for_cadence_and_final_budget_does_not_trigger_reflection(tmp_path):
    reflector = Reflector()
    config = Config(evaluation_budget=24, initial_size=3)
    run = search(tmp_path, Generator(failures=[4]), reflector, config=config)
    run.run()
    counted = [a for a in run.attempts if a.operator != "Init" and a.evaluations]
    assert len(counted) == 20 and sum(len(a.evaluations) for a in counted) == 21
    assert len(reflector.prompts) == 1
    assert counted[0].repair_used and len(counted[0].evaluations) == 2
    assert run.events[counted[0].id].child.code == run.archive[counted[0].program_id].code
    assert "reflection" not in run.attempts[-1].record
    assert {a.id for a in counted[-10:]} <= set(run.unreviewed_events)
    resumed = search(tmp_path, config=config, resume=True)
    assert resumed.snapshot() == run.snapshot()


def test_unevaluated_format_failures_do_not_advance_the_reflection_interval(tmp_path):
    class BrokenFormat(Generator):
        def complete(self, prompt, **settings):
            if self.n in (3, 4):
                self.n += 1
                self.prompts.append(prompt)
                return "not json"
            return super().complete(prompt, **settings)
    reflector = Reflector()
    run = search(tmp_path, generator=BrokenFormat(), reflector=reflector)
    for _ in range(14):
        run.step()
    failed = next(a for a in run.attempts if not a.evaluations)
    assert len(failed.calls) == 2 and failed.id not in run.events
    reflected = [a for a in run.attempts if "reflection" in a.record]
    assert [a.id for a in reflected] == [14]
    reflection = reflected[0].record["reflection"]
    assert 1 <= len(reflection["reviewed_event_ids"]) <= 3
    assert set(reflection["reviewed_event_ids"]) <= set(reflection["shown_event_ids"])


def test_selection_evaluation_and_generation_budget_match_v21(tmp_path):
    config = Config(evaluation_budget=37, initial_size=3, seed=8)
    baseline = V21(task(), Generator(), config=config, run_dir=tmp_path / "v21")
    baseline.evaluator.evaluate = evaluate
    revised = search(tmp_path / "v24_1", config=config)
    a, b = baseline.run(), revised.run()
    assert a.evaluations == b.evaluations == a.model_calls == b.model_calls == 37
    assert a.best.code == b.best.code
    assert [(x.parent_id, x.operator, x.reference_id) for x in baseline.attempts] == [
        (x.parent_id, x.operator, x.reference_id) for x in revised.attempts]


@pytest.mark.parametrize("operator", ["Refine", "Crossover", "Explore"])
def test_no_experience_keeps_v21_material_with_positive_operator_guidance(tmp_path, operator):
    run = search(tmp_path)
    for _ in range(3):
        run.step()
    parent = run.archive[1]
    reference = run.archive[2] if operator == "Crossover" else None
    attempt = Attempt(4, parent.id, operator, reference.id if reference else None)
    expected, _, _, view = v21_context(run.archive, parent, operator, reference, run.config)
    prompt = run.evolution_prompt(attempt, parent, reference)
    assert expected in prompt and attempt.record["context"] == view
    assert "[Design Experience]" not in prompt and attempt.record["experience_ids"] == []
    assert not re.search(r"\b(do not|does not|not merely|cannot|never|avoid|without)\b", prompt, re.I)
    output = prompt.split("[Output]")[1]
    assert output.index('"analysis"') < output.index('"code"') < output.index('- "design"')
    if operator == "Refine":
        assert "keeping its main mechanism" in prompt
    elif operator == "Crossover":
        assert "reference computation" in prompt
    else:
        assert "Introduce a different computation" in prompt


@pytest.mark.parametrize("operator", ["Refine", "Crossover", "Explore"])
def test_operator_uses_conditions_without_changing_v21_material_or_exposing_sources(tmp_path, operator):
    run = search(tmp_path)
    for _ in range(3):
        run.step()
    run.experience.apply(json.dumps({"analysis": "A condition.", "updates": [{
        "id": None, "idea": "remainder", "when": "Remainder determines the return value.",
        "lesson": "Check whether a change affects the remainder.", "evidence_ids": [1]}], "archive_ids": []}),
                         {1: run.events[1]}, editable_ids=set())
    parent, reference = run.archive[1], run.archive[2] if operator == "Crossover" else None
    attempt = Attempt(4, parent.id, operator, reference.id if reference else None)
    prompt = run.evolution_prompt(attempt, parent, reference)
    text, instruction, scores, view = v21_context(run.archive, parent, operator, reference, run.config)
    assert attempt.record["context"] == view
    if operator == "Explore":
        assert prompt == run.prompts._render(text, EXPLORE, has_scores=scores)
        assert attempt.record["experience_ids"] == []
    else:
        assert experience_block(run.experience.active) + "\n\n" + text in prompt
        assert "Applicable computational conditions" in prompt and attempt.record["experience_ids"] == [1]
        assert "evidence_ids" not in prompt and "source_ids" not in prompt
        assert prompt.index("[Current Algorithm]") < prompt.index("[Instruction]")


def test_direct_client_gets_distinct_reflection_schema_and_cross_version_resume_is_rejected(tmp_path):
    class Client(Generator):
        request_options = {}
    client = Client()
    run = search(tmp_path, generator=client)
    direct = TraceAAD(task(), client, reflection_counter=len, run_dir=tmp_path / "direct")
    assert direct.reflection_model is not client
    assert direct.reflection_model.request_options["response_format"] == REFLECTION_FORMAT
    assert client.request_options["response_format"] != REFLECTION_FORMAT
    saved = run.store.read("run_config.json")
    run.store.write("run_config.json", {**saved, "method": "traceaad_v24"})
    calls = tmp_path / "calls.jsonl"
    calls.write_bytes(b'{"unfinished":')
    with pytest.raises(ValueError, match="resume settings: method"):
        search(tmp_path, resume=True)
    assert calls.read_bytes() == b'{"unfinished":'


def test_reflection_uses_separate_output_budget_and_requires_a_tokenizer(tmp_path):
    reflector = Reflector()
    run = search(tmp_path / "run", reflector=reflector)
    for _ in range(13):
        run.step()
    assert reflector.settings[0]["max_tokens"] == REFLECTION_TOKENS
    assert run.config.max_tokens == 16384
    assert run.input_limit == 49152 - REFLECTION_TOKENS - CHAT_RESERVE
    with pytest.raises(ValueError, match="requires count_prompt_tokens"):
        TraceAAD(task(), Generator(), run_dir=tmp_path / "missing")


def test_experiment_entry_and_launcher_register_both_aliases_and_report_reflection_cost(tmp_path, monkeypatch, capsys):
    from experiments.traceaad import run
    from experiments.traceaad.launch_host import build_plan
    from traceaad_base import Evaluator

    class Scheduled(Generator):
        model = "offline-test"
        def __init__(self, path, label):
            super().__init__()
            self.path, self.scheduler = path, {"endpoints": []}
            self.reflector = Reflector(make_note=True)
        def for_role(self, schema):
            return self.reflector

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_environment", lambda *_: None)
    monkeypatch.setattr(run, "load_task", lambda **_: task())
    monkeypatch.setattr(run, "ScheduledModel", Scheduled)
    def evaluator(task, *_):
        result = Evaluator(task)
        result.evaluate = evaluate
        return result
    monkeypatch.setattr(run, "ScheduledEvaluator", evaluator)
    run.main(["--method", "v24.1", "--task", "tsp_construct", "--run-name", "offline", "--batch", "offline",
              "--experiment", "traceaad_v24_1", "--budget", "26", "--initial-size", "3",
              "--scheduler-socket", "/fake", "--skip-heldout"])
    directory = tmp_path / "experiments_result/traceaad_v24_1/tsp_construct/offline"
    summary = json.loads((directory / "summary.json").read_text())
    native = json.loads((directory / "native/run_config.json").read_text())
    assert summary["method"] == native["method"] == "traceaad_v24_1"
    assert summary["evaluation_calls"] == summary["model_calls"] == 26
    assert summary["auxiliary"]["reflect"]["requests"] == 2
    assert native["reflection_settings"]["interval"] == 10
    assert run.METHODS["v24.1"] == run.METHODS["v24_1"]
    args = Namespace(tasks=None, method="v24.1", batch="new", experiment="traceaad_v24_1", repeats=3, backend="scheduled",
                     budget=1000, initial_size=5, max_tokens=16384, model_timeout=600, selection_temperature=1.0,
                     trajectory_reference_embedding=False,
                     scheduler_socket="/fake", seconds_per_instance=10., env_file=None, skip_heldout=False)
    plan = build_plan(args)
    assert len(plan) == 18 and all("--method" in row["command"] for row in plan)
    assert all("traceaad_v24_1" in row["run_name"] for row in plan)
    capsys.readouterr()


def test_cross_branch_feedback_and_new_knowledge_survive_display_budget_and_resume(tmp_path):
    class BranchGenerator(Generator):
        def complete(self, prompt, **settings):
            self.n += 1
            self.prompts.append(prompt)
            if self.n == 1:
                code = "def solve(x):\n    scale = 100.\n    return x + scale\n"
            elif self.n == 2:
                code = "def solve(x):\n    magnitude = 1000.\n    offset = 4.\n    return magnitude + offset\n"
            elif self.n <= 12:
                code = f"def solve(x):\n    scale = 100.\n    return x + scale / {self.n}\n"
            else:
                assert "Normalize the magnitude before combination." in prompt or "before adding the offset" in prompt
                code = ("def solve(x):\n    magnitude = 1000.\n    offset = 4.\n"
                        f"    return magnitude / {self.n} + offset\n")
            return json.dumps({"analysis": "Normalize the magnitude used in the decision.",
                               "code": code, "design": "Combine the normalized magnitude with the remaining term."})

    class BranchReflector(Reflector):
        def complete(self, prompt, **settings):
            self.prompts.append(prompt)
            new_ids = json.loads(re.search(r"New events: (\[[^\n]*\])", prompt)[1])
            source = max(new_ids)
            main = {"id": None, "idea": "Magnitude normalization", "when": "Different magnitudes enter a decision.",
                    "lesson": "Normalize the magnitude before combination.", "evidence_ids": [source]}
            if source <= 12:
                updates = [main, *[{**main, "idea": f"Population restart {i}",
                                    "when": "Population search restarts.", "lesson": "Reuse restart candidates."}
                                   for i in range(5)]]
            else:
                assert "Program 1" in prompt and "Program 2" in prompt
                assert "Experience available during generation: [1, 2, 3, 4, 5, 6]" in prompt
                updates = [
                    {**main, "id": 1, "when": "A magnitude and an additive offset enter the decision.",
                     "lesson": "Normalize the magnitude before adding the offset.", "evidence_ids": [12, source]},
                    {**main, "idea": "Additive offset", "when": "The normalized magnitude shares a decision with an offset.",
                     "lesson": "Keep the additive offset at its intended scale."},
                ]
            return json.dumps({"analysis": "The two branches reveal how normalization interacts with combination.",
                               "updates": updates, "archive_ids": []})

    def choose_branch(run):
        def select(attempt_id):
            return run.archive[1 if attempt_id <= 12 else 2], "Refine", None, {}
        run.select = select

    config = Config(evaluation_budget=24, initial_size=2)
    generator, reflector = BranchGenerator(), BranchReflector()
    run = search(tmp_path / "first", generator, reflector, config=config)
    choose_branch(run)
    for _ in range(23):
        run.step()
    assert len(reflector.prompts) == 2 and len(run.experience.active) == 7
    first, revised = run.attempts[11].record["reflection"], run.attempts[21].record["reflection"]
    assert first["status"] == revised["status"] == "success"
    assert first["updates"][0]["id"] == revised["updates"][0]["id"] == 1
    assert revised["updates"][0]["evidence_ids"] == [12, 22]
    assert run.events[12].parent.id == 1 and run.events[22].parent.id == 2
    assert "magnitude / 13 + offset" in run.events[13].child.code
    assert run.events[13].experience_ids == (1, 2, 3, 4, 5, 6)
    assert 7 in run.attempts[22].record["experience_ids"]
    assert len(run.attempts[22].record["experience_ids"]) == 6
    assert 22 in revised["reviewed_event_ids"] and 22 not in run.unreviewed_events
    saved = run.snapshot()
    shutil.copytree(tmp_path / "first", tmp_path / "resumed")
    resumed_generator, resumed_reflector = BranchGenerator(start=23), BranchReflector()
    resumed = search(tmp_path / "resumed", resumed_generator, resumed_reflector, config=config, resume=True)
    choose_branch(resumed)
    assert resumed.snapshot() == saved and resumed.events == run.events
    assert resumed_generator.prompts == resumed_reflector.prompts == []
    run.step()
    resumed.step()
    assert generator.prompts[-1] == resumed_generator.prompts[-1]
    assert resumed.snapshot() == run.snapshot()
    assert run.model_calls == run.evaluations_used == 24
    for prompt in reflector.prompts + [generator.prompts[-1]]:
        assert not re.search(r"\b(do not|does not|not merely|cannot|never|avoid|without)\b", prompt, re.I)


def test_old_experience_protocol_is_rejected_before_log_repair(tmp_path):
    run = search(tmp_path)
    config = run.store.read("run_config.json")
    config["reflection_settings"]["organization"] = "focused_relations"
    run.store.write("run_config.json", config)
    calls = tmp_path / "calls.jsonl"
    calls.write_bytes(b'{"unfinished":')
    with pytest.raises(ValueError, match="resume settings: reflection_settings"):
        search(tmp_path, resume=True)
    assert calls.read_bytes() == b'{"unfinished":'
