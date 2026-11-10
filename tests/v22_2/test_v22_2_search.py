"""V22.2 and V23.2 actual attempt, budget, first_result, prompt and recovery boundaries."""

import json
from argparse import Namespace
from dataclasses import replace

import pytest

from experiments.traceaad.launch_host import build_plan
from experiments.traceaad.run import METHODS, request_options
from traceaad_base import Config, Evaluation, Program, Task
from traceaad_base.models import Attempt
from traceaad_base.llm import CANDIDATE_RESPONSE_FORMAT as BASE_RESPONSE_FORMAT
from traceaad_base.search import TraceAAD as Base
from traceaad_v21_1.context import context
from traceaad_v21_1.prompts import CANDIDATE_RESPONSE_FORMAT as V21_RESPONSE_FORMAT
from traceaad_v22_2 import TraceAAD as V22_2
from traceaad_v23_2 import TraceAAD as V23_2


class Replies:
    request_options = {}

    def __init__(self, start=0, values=None):
        self.index = start
        self.values = values
        self.prompts = []

    def complete(self, prompt, **options):
        self.prompts.append(prompt)
        i = self.index
        self.index += 1
        value = self.values[i] if self.values is not None else 100.0 - i
        if value == "format":
            return "{}"
        if value == "dup100":
            code = "def solve(x):\n    return 100.0\n"
        elif value == "error":
            code = 'def solve(x):\n    raise ValueError("bad")\n'
        else:
            code = f"def solve(x):\n    return {float(value)}\n"
        return json.dumps({"analysis": f"change {i}", "code": code, "design": f"design {i}"})


def task():
    return Task(
        "Return a finite score.",
        "def solve(x):\n    return 1\n",
        lambda source, fn: float(fn(0)),
        time_limit=5.0,
        entrypoint="solve",
    )


def assert_same_selectors(a, b):
    assert a.nodes == b.nodes
    assert a.sources == b.sources
    assert a.processed == b.processed
    assert a.champion == b.champion
    assert a.evaluations_used == b.evaluations_used
    assert a.history == b.history
    assert a.settings == b.settings


@pytest.mark.parametrize("search_type", [V22_2, V23_2])
def test_resumed_search_matches_uninterrupted_actions_and_history(tmp_path, search_type):
    class Model(Replies):
        request_options = {"response_format": BASE_RESPONSE_FORMAT}

    config = Config(evaluation_budget=9, initial_size=2, seed=7)
    full = search_type(task(), Model(), config=config, run_dir=tmp_path / "full")
    result = full.run()
    split = search_type(task(), Model(), config=config, run_dir=tmp_path / "split")
    for _ in range(5):
        split.step()
    resumed = search_type(
        task(),
        Model(start=split.model_calls),
        config=config,
        run_dir=tmp_path / "split",
        resume=True,
    )
    assert_same_selectors(split.selector, resumed.selector)
    resumed.run()
    assert result.evaluations == resumed.evaluations_used == 9
    assert [a.record for a in full.attempts] == [a.record for a in resumed.attempts]
    assert_same_selectors(full.selector, resumed.selector)
    assert full.model.prompts[5:] == resumed.model.prompts
    assert full.best.code == resumed.best.code
    assert set(full.model.request_options["response_format"]["json_schema"]["schema"]["properties"]) == {
        "analysis",
        "code",
        "design",
    }


def test_format_repair_before_first_evaluation_counts_as_first_result(tmp_path):
    # Init: 100.0, 110.0. Evolution attempt 3: format error -> format repair -> 95.0 (first actual evaluation!)
    values = [100.0, 110.0, "format", 95.0]
    search = V22_2(task(), Replies(values=values), config=Config(evaluation_budget=3, initial_size=2), run_dir=tmp_path)
    search.run()
    last = search.attempts[-1]
    assert search.evaluations_used == 3 and search.model_calls == 4
    assert len(last.evaluations) == 1 and last.repair_used is True
    sel = last.record["selection"]
    assert sel["first_result"]["status"] == "success"
    assert sel["first_result"]["fitness"] == 95.0
    assert sel["first_result"]["usable_at_attempt_start"] is True
    assert sel["first_champion_gain"] == 5.0
    assert sel["champion_gain"] == sel["first_champion_gain"]


def test_evaluation_error_then_repair_sets_zero_first_gain_and_positive_final_gain(tmp_path):
    # Init: 100.0, 110.0. Attempt 3: first eval error -> program repair -> second eval 92.0
    values = [100.0, 110.0, "error", 92.0]
    search = V22_2(task(), Replies(values=values), config=Config(evaluation_budget=4, initial_size=2), run_dir=tmp_path)
    search.run()
    last = search.attempts[-1]
    sel = last.record["selection"]
    assert sel["evaluations"] == 2
    assert sel["first_result"]["status"] == "error"
    assert sel["first_result"]["usable_at_attempt_start"] is False
    assert sel["first_champion_gain"] == 0.0
    assert sel["final_result"]["status"] == "success"
    assert sel["final_result"]["usable_at_attempt_start"] is True
    assert sel["champion_gain"] == 8.0

    # Resuming from saved logs preserves both first_result and final_result
    restored = V22_2(task(), Replies(values=values), config=Config(evaluation_budget=4, initial_size=2), run_dir=tmp_path, resume=True)
    assert_same_selectors(search.selector, restored.selector)


def test_duplicate_source_at_first_evaluation_has_zero_gain_and_not_added_to_pool(tmp_path):
    values = [100.0, 110.0, "dup100"]
    search = V22_2(task(), Replies(values=values), config=Config(evaluation_budget=3, initial_size=2), run_dir=tmp_path)
    search.run()
    last = search.attempts[-1]
    sel = last.record["selection"]
    assert sel["first_result"]["status"] == "success"
    assert sel["first_result"]["usable_at_attempt_start"] is False
    assert sel["first_champion_gain"] == 0.0
    assert sel["champion_gain"] == 0.0
    assert len(search.selector.nodes) == 2


def test_no_evaluation_attempt_records_zero_evaluations_and_no_yield(tmp_path):
    class Failing(Replies):
        def complete(self, prompt, **options):
            if self.index >= 2:
                raise ConnectionError("No reply")
            return super().complete(prompt, **options)

    config = Config(evaluation_budget=8, initial_size=2, max_unevaluated=2)
    search = V22_2(task(), Failing(), config=config, run_dir=tmp_path)
    result = search.run()
    assert result.stop_reason == "consecutive_unevaluated_limit"
    assert result.evaluations == 2 and result.attempts == 4
    assert len(search.selector.history) == 2
    for rec in search.selector.history:
        assert rec.evaluations == 0
        assert rec.champion_gain == 0.0
    restored = V22_2(task(), Failing(), config=config, run_dir=tmp_path, resume=True)
    assert_same_selectors(search.selector, restored.selector)


@pytest.mark.parametrize("trajectory_references", [False, True])
def test_v22_2_and_v23_2_share_allocation_and_use_respective_prompts(tmp_path, trajectory_references):
    class Encoder:
        settings = {"model": "deterministic-test-encoder"}
        last_usage = {}

        def __call__(self, text):
            return [1.0, float(len(text) % 17 + 1)]

    config = Config(evaluation_budget=8, initial_size=2, seed=11,
                    refine_weight=0.0, explore_weight=0.0, crossover_weight=1.0)
    v22_2 = V22_2(task(), Replies(), config=config, run_dir=tmp_path / "v22_2",
                  reference_embedding=Encoder() if trajectory_references else None)
    v23_2 = V23_2(task(), Replies(), config=config, run_dir=tmp_path / "v23_2",
                  reference_embedding=Encoder() if trajectory_references else None)
    v22_2.run()
    v23_2.run()
    for a, b in zip(v22_2.attempts, v23_2.attempts, strict=True):
        assert (a.parent_id, a.operator, a.reference_id) == (b.parent_id, b.operator, b.reference_id)
        assert a.record.get("selection") == b.record.get("selection")
    assert_same_selectors(v22_2.selector, v23_2.selector)
    assert v22_2.model.prompts != v23_2.model.prompts
    assert all("context" in a.record for a in v22_2.attempts[2:])
    assert all("context" not in a.record for a in v23_2.attempts[2:])
    expected = "trajectory_embedding_diverse_v1" if trajectory_references else "quality_pool_uniform_v1"
    assert any(a.operator == "Crossover" for a in v22_2.attempts[2:])
    assert all(a.operator == ("Crossover" if a.reference_id is not None else "Explore")
               for a in v22_2.attempts[2:])
    assert all(a.record["selection"]["reference_selection"]["rule"] == expected for a in v22_2.attempts[2:])


def test_registry_and_launcher_support_dot_and_underscore_names():
    assert not {"v20", "v22", "v23"} & METHODS.keys()
    for key in ("v22.2", "v22_2"):
        assert METHODS[key][0] == "traceaad_v22_2"
        assert request_options(key)["response_format"] == V21_RESPONSE_FORMAT
    for key in ("v23.2", "v23_2"):
        assert METHODS[key][0] == "traceaad_v23_2"
        assert request_options(key)["response_format"] == BASE_RESPONSE_FORMAT

    args = Namespace(
        tasks=None,
        method="v22.2",
        batch="test",
        experiment="traceaad_v22_2",
        repeats=3,
        backend="scheduled",
        budget=1000,
        initial_size=5,
        max_tokens=16384,
        model_timeout=600,
        selection_temperature=0.1,
        trajectory_reference_embedding=False,
        scheduler_socket="/fake",
        seconds_per_instance=10.0,
        env_file=None,
        skip_heldout=False,
    )
    plan = build_plan(args)
    assert len(plan) == 18
    assert all(row["session"].endswith(f"_v22_2_r{row['repeat']}") for row in plan)
    assert all("traceaad_v22_2" in row["run_name"] for row in plan)
    assert all(row["command"][3] == "experiments.traceaad.run" for row in plan)
    assert all(row["command"][row["command"].index("--selection-temperature") + 1] == "0.1"
               for row in plan)


def test_resume_rejects_incompatible_settings(tmp_path):
    config = Config(evaluation_budget=6, initial_size=2)
    search = V22_2(task(), Replies(), config=config, run_dir=tmp_path)
    search.step()
    path = tmp_path / "run_config.json"
    saved = json.loads(path.read_text())
    saved["selection_settings"]["shrinkage"] = 8.0
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="selection_settings.shrinkage"):
        V22_2(task(), Replies(), config=config, run_dir=tmp_path, resume=True)


@pytest.mark.parametrize("operator", ["Refine", "Explore", "Crossover"])
def test_v23_2_uses_current_base_generation_without_trajectory(operator):
    archive = {
        i: Program(i, f"def solve(x): return {i}", f"design_{i}",
                   "Init" if i in (1, 5) else "Refine", 0 if i in (1, 5) else i - 1,
                   None, Evaluation("success", float(i)), analysis=f"analysis_{i}")
        for i in range(1, 6)
    }
    archive[4].evaluation = Evaluation("error", failure="ValueError: hidden offspring failure")
    archive[4].parent_id = 2
    search, base = V23_2(task(), Replies()), Base(task(), Replies())
    search.archive = base.archive = archive
    parent = archive[2]
    reference = archive[5] if operator == "Crossover" else None
    attempt = Attempt(6, parent.id, operator, reference.id if reference else None)
    prompt = search.evolution_prompt(attempt, parent, reference)
    assert prompt == base.evolution_prompt(attempt, parent, reference)
    assert parent.code in prompt and parent.design in prompt
    assert all(p.design not in prompt and p.analysis not in prompt
               for p in (archive[1], archive[3], archive[4]))
    assert parent.analysis not in prompt
    assert (archive[5].design in prompt) == (operator == "Crossover")
    assert search.prompts.initialization(archive) == base.prompts.initialization(archive)
    assert search.prompts.repair_program(archive[4]) == base.prompts.repair_program(archive[4])


@pytest.mark.parametrize("protocol", [None, "trajectory_context"])
def test_v23_2_rejects_old_generation_protocol_before_log_changes(tmp_path, protocol):
    config = Config(evaluation_budget=6, initial_size=2)
    search = V23_2(task(), Replies(), config=config, run_dir=tmp_path)
    search.step()
    path = tmp_path / "run_config.json"
    saved = json.loads(path.read_text())
    assert saved["generation_protocol"] == Base.generation_protocol
    if protocol is None:
        del saved["generation_protocol"]
    else:
        saved["generation_protocol"] = protocol
    path.write_text(json.dumps(saved))
    before = {}
    for name in ("attempts.jsonl", "calls.jsonl"):
        log = tmp_path / name
        log.write_bytes(log.read_bytes() + b'{"unfinished":')
        before[name] = log.read_bytes()
    model = Replies()
    with pytest.raises(ValueError, match="original source snapshot"):
        V23_2(task(), model, config=config, run_dir=tmp_path, resume=True)
    assert not model.prompts
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())
