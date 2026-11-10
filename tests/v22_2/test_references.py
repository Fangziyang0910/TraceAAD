"""Check reference-policy decisions, trajectory inputs, caching and actual search integration."""

import json

import pytest

from traceaad_base import Config, Evaluation, Program, Task
from traceaad_base.storage import Store
from traceaad_v22_2.references import RULE, TrajectoryReferences, trajectory_text
from traceaad_v22_2.selection import Action, Node, Selector
from traceaad_v23_2 import TraceAAD


def program(pid, fitness, *, parent=0, design=None):
    return Program(pid, f"def solve(x):\n    return x + {pid}\n", design or f"p{pid}",
                   "Refine" if parent else "Init", parent, None, Evaluation("success", fitness))


class Embedding:
    def __init__(self, vectors=None, model="test-embedding"):
        self.vectors = vectors or {}
        self.settings = {"model": model}
        self.last_usage = {}
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        self.last_usage = {"total_tokens": 7}
        label = text.splitlines()[1]
        return self.vectors.get(label, [1.0, float(len(label))])


def test_reference_is_trajectory_dissimilar_but_quality_filtered_and_not_self(tmp_path):
    archive = {i: program(i, float(i)) for i in range(1, 7)}
    embedding = Embedding({"p1": [1, 0], "p2": [0.9, 0.1], "p3": [-1, 0],
                           "p4": [1, 0], "p5": [0, 1], "p6": [-1, 0]})
    selector = TrajectoryReferences(embedding, Store(tmp_path))
    references = selector.select(archive)
    assert references[4] == 3
    assert references[3] == 1
    assert all(pid != ref and ref in {1, 2, 3} for pid, ref in references.items())
    assert selector.records[4]["cosine_similarity"] == pytest.approx(-1.0)
    assert selector.records[4]["rule"] == RULE


def test_duplicate_source_does_not_create_an_additional_reference(tmp_path):
    archive = {1: program(1, 1.0), 2: program(2, 2.0), 3: program(3, 3.0)}
    archive[2].code = archive[1].code + "\n# same source\n"
    selector = TrajectoryReferences(Embedding(), Store(tmp_path))
    assert selector.select(archive) == {1: None, 3: 1}


def test_no_partner_does_not_call_embedding(tmp_path):
    embedding = Embedding()
    selector = TrajectoryReferences(embedding, Store(tmp_path))
    assert selector.select({1: program(1, 1.0)}) == {1: None}
    assert embedding.calls == []


def test_embedding_reference_can_be_from_the_same_family():
    selector = Selector()
    selector.nodes = {
        1: Node(1, 1.0, "Init", None, None, 1, 0.0, 0.0, 1.0),
        2: Node(2, 2.0, "Refine", 1, None, 1, -1.0, -0.5, 1.0),
    }
    assert Action(2, "Crossover", 1) in selector.actions({1: None, 2: 1})


def test_trajectory_input_has_actual_ordered_changes_and_observed_result():
    archive = {i: program(i, float(i), parent=i - 1) for i in range(1, 6)}
    text = trajectory_text(archive, archive[5])
    assert "Current code" in text and "return x + 5" in text
    assert text.count("Operation: Refine") == 3
    assert text.count("Observed result: degraded") == 3
    assert text.index("+    return x + 3") < text.index("+    return x + 4") < text.index("+    return x + 5")
    assert "Initial program; no recorded formation step." in trajectory_text(archive, archive[1])


def test_cached_embeddings_survive_recreation_without_service_calls(tmp_path):
    archive = {i: program(i, float(i)) for i in range(1, 5)}
    original = TrajectoryReferences(Embedding(), Store(tmp_path))
    expected = original.select(archive)
    original.select(archive)
    assert len(original.embedding.calls) == 4
    second = TrajectoryReferences(Embedding(), Store(tmp_path))
    assert second.select(archive) == expected
    assert second.records == original.records
    assert second.embedding.calls == []
    calls = Store(tmp_path).rows("reference_embedding_calls.jsonl")
    assert len(calls) == 4 and all(row["usage"]["total_tokens"] == 7 for row in calls)


def test_embedding_failure_is_recorded_and_propagated(tmp_path):
    class Broken(Embedding):
        def __call__(self, text):
            raise ConnectionError("embedding unavailable")

    selector = TrajectoryReferences(Broken(), Store(tmp_path))
    with pytest.raises(ConnectionError, match="embedding unavailable"):
        selector.select({1: program(1, 1.0), 2: program(2, 2.0)})
    calls = Store(tmp_path).rows("reference_embedding_calls.jsonl")
    assert calls[-1]["status"] == "error"
    assert list((tmp_path / "reference_embeddings").glob("*.json")) == []


class Generator:
    def __init__(self, start=0):
        self.index = start

    def complete(self, prompt, **options):
        self.index += 1
        return json.dumps({"analysis": "Change the constant.", "design": f"candidate{self.index}",
                           "code": f"def solve(x):\n    return {10.0 - self.index}\n"})


def task():
    return Task("Minimize the score.", "def solve(x): return x", lambda _src, fn: fn(0), 5)


def test_embedding_policy_reaches_actual_crossover_and_replays_identically(tmp_path):
    config = Config(evaluation_budget=6, initial_size=3,
                    refine_weight=0.0, explore_weight=0.0, crossover_weight=1.0)
    full = TraceAAD(task(), Generator(), config=config, run_dir=tmp_path / "full",
                    reference_embedding=Embedding())
    full.run()
    split = TraceAAD(task(), Generator(), config=config, run_dir=tmp_path / "split",
                     reference_embedding=Embedding())
    for _ in range(4):
        split.step()
    restored = TraceAAD(task(), Generator(start=4), config=config, run_dir=tmp_path / "split", resume=True,
                        reference_embedding=Embedding())
    restored.run()
    assert [a.record for a in full.attempts] == [a.record for a in restored.attempts]
    assert full.evaluations_used == full.model_calls == 6
    for attempt in full.attempts[3:]:
        assert attempt.operator == "Crossover"
        record = attempt.record["selection"]
        assert record["reference_selection"]["rule"] == RULE
        assert record["reference_selection"]["reference_id"] == attempt.reference_id
    assert any(row["input"].count("Operation:") > 0
               for row in [json.loads(p.read_text()) for p in (tmp_path / "full" / "reference_embeddings").glob("*.json")])


def test_disabled_crossover_does_not_request_embeddings(tmp_path):
    embedding = Embedding()
    config = Config(evaluation_budget=3, initial_size=2, crossover_weight=0.0)
    search = TraceAAD(task(), Generator(), config=config, run_dir=tmp_path, reference_embedding=embedding)
    search.run()
    assert embedding.calls == []
    assert search.attempts[-1].record["selection"]["reference_selection"]["rule"] == RULE


def test_missing_reference_redirects_actual_crossover_to_explore(tmp_path):
    config = Config(evaluation_budget=2, initial_size=1,
                    refine_weight=0.0, explore_weight=0.0, crossover_weight=1.0)
    embedding = Embedding()
    search = TraceAAD(task(), Generator(), config=config, run_dir=tmp_path, reference_embedding=embedding)
    search.run()
    assert search.attempts[-1].operator == "Explore"
    assert search.attempts[-1].reference_id is None
    assert search.attempts[-1].record["selection"]["base_probability"] == 1.0
    assert embedding.calls == []


@pytest.mark.parametrize("change", ["disable", "model"])
def test_resume_rejects_changed_embedding_policy_before_log_mutation(tmp_path, change):
    config = Config(evaluation_budget=3, initial_size=2)
    search = TraceAAD(task(), Generator(), config=config, run_dir=tmp_path, reference_embedding=Embedding())
    search.step()
    calls = tmp_path / "calls.jsonl"
    calls.write_bytes(calls.read_bytes() + b'{"unfinished":')
    before = calls.read_bytes()
    embedding = None if change == "disable" else Embedding(model="different-model")
    with pytest.raises(ValueError, match="reference_rule|reference_settings"):
        TraceAAD(task(), Generator(), config=config, run_dir=tmp_path, resume=True,
                 reference_embedding=embedding)
    assert calls.read_bytes() == before
