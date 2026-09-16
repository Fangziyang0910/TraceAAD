import random

from llm4ad.base import Evaluation
from llm4ad.method.traceaad_v10_11 import TraceAADV1011
from llm4ad.method.traceaad_v10_11.core import read_journal
from llm4ad.method.traceaad_v10_11_rand_ctx import TraceAADV1011RandCtx
from llm4ad.method.traceaad_v10_11_rand_ctx.context import rank_softmax_sample


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program="def score(x):\n    pass",
                         task_description="Return a numeric score.", safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 7

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response if isinstance(response, dict) else {
            "content": response, "finish_reason": "stop", "usage": {"completion_tokens": 20},
        }


def response(value=1):
    return (
        "Idea: Return the requested constant.\n"
        "Code:\n"
        "```python\n"
        f"def score(x):\n    return {value}\n"
        "```"
    )


def method(path, llm=None, **kwargs):
    return TraceAADV1011RandCtx(
        evaluation=TinyEvaluation(),
        llm=llm or FakeLLM(),
        run_dir=path,
        **{"budget": 1000, "n_roots": 1, **kwargs},
    )


def add(tree, fitness, parent=None, code="def score(x):\n    return 1", operator=None):
    return tree.add(
        code=code,
        idea=f"idea {len(tree.nodes)}",
        fitness=fitness,
        evaluation_id=len(tree.nodes) + 1,
        parent_id=parent,
        operator="Init" if parent is None else (operator or "Refine"),
    )


class Numbered:
    def __init__(self, id, fitness):
        self.id, self.fitness = id, fitness


def test_rank_softmax_sample_is_deterministic_and_without_replacement():
    nodes = [Numbered(i, i) for i in range(30)]
    first = rank_softmax_sample(nodes, 8, random.Random(7))
    second = rank_softmax_sample([Numbered(i, i) for i in range(30)], 8, random.Random(7))
    assert [node.id for node in first] == [node.id for node in second]
    assert len({node.id for node in first}) == 8


def test_rank_softmax_sample_prefers_better_fitness():
    nodes = [Numbered(i, i) for i in range(100)]
    rng = random.Random(12345)
    drawn = [rank_softmax_sample(nodes, 1, rng)[0].fitness for _ in range(500)]
    assert sum(drawn) / len(drawn) > 75


def test_rank_softmax_sample_returns_all_candidates_when_few():
    nodes = [Numbered(0, 1), Numbered(1, 3), Numbered(2, 2)]
    chosen = rank_softmax_sample(nodes, 8, random.Random(0))
    assert sorted(node.id for node in chosen) == [0, 1, 2]


def test_rand_ctx_context_replaces_history_with_unordered_cards(tmp_path):
    m = method(tmp_path, budget=1)
    root = add(m.tree, 1)
    child = add(m.tree, 2, root.id, code="def score(x):\n    return x")
    references = [add(m.tree, 5), add(m.tree, 3), add(m.tree, 4)]
    prompt = m.builder.build(child, "Refine", references=references)
    assert "# Archive Algorithms" in prompt
    assert "3 independently evaluated algorithms" in prompt
    assert "Reference 1 | Fitness: 5" in prompt
    assert "Reference 2 | Fitness: 4" in prompt
    assert "Reference 3 | Fitness: 3" in prompt
    assert "idea 2" in prompt and "idea 3" in prompt and "idea 4" in prompt
    assert prompt.count("```python") == 3  # task stub, current algorithm, output contract
    assert "# Design History" not in prompt
    assert "Step 1 |" not in prompt


def test_rand_ctx_mechanism_is_isolated_from_v1011(tmp_path):
    m = method(tmp_path / "rand", budget=1)
    v = TraceAADV1011(evaluation=TinyEvaluation(), llm=FakeLLM(),
                      run_dir=tmp_path / "base", budget=1, n_roots=1)
    assert m.mechanism["method"] == "v1011rc"
    assert m.mechanism["context"] == "random_references"
    assert m.mechanism["n_references"] == 8
    assert m.mechanism["traj_gens"] == 0
    assert m.mechanism != v.mechanism


def test_rand_ctx_initialization_context_matches_v1011(tmp_path):
    m = method(tmp_path / "rand", budget=1)
    v = TraceAADV1011(evaluation=TinyEvaluation(), llm=FakeLLM(),
                      run_dir=tmp_path / "base", budget=1, n_roots=1)
    for engine in (m, v):
        root = add(engine.tree, 1)
        add(engine.tree, 2, root.id, code="def score(x):\n    return x")
    assert m.builder.build_initial() == v.builder.build_initial()


def test_rand_ctx_schedule_records_reference_ids_deterministically(tmp_path):
    def engine_with_populated_tree(path):
        engine = method(path, budget=1, seed=11)
        root = add(engine.tree, 10)
        for fitness in range(1, 10):
            add(engine.tree, fitness, parent=root.id)
        return engine

    first_engine = engine_with_populated_tree(tmp_path / "a")
    second_engine = engine_with_populated_tree(tmp_path / "b")
    first = first_engine._schedule()
    second = second_engine._schedule()
    assert first["reference_ids"] == second["reference_ids"]
    assert len(first["reference_ids"]) == 8
    assert first["parent_id"] not in first["reference_ids"]
    assert all(node_id in first_engine.tree.nodes and node_id != first["parent_id"]
               for node_id in first["reference_ids"])
    assert "# Archive Algorithms" in first["prompt"]


def test_rand_ctx_run_persists_reference_ids_in_events(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response(2), response(3)),
               budget=3, n_roots=2)
    m.run()
    events = read_journal(m.events_path)
    assert [event["status"] for event in events] == ["ok", "ok", "ok"]
    search = events[2]
    assert search["parent_id"] is not None
    assert len(search["reference_ids"]) == 1
    assert search["parent_id"] not in search["reference_ids"]
    assert events[0]["reference_ids"] == [] and events[1]["reference_ids"] == []
