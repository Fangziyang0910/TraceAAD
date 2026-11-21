"""EvoTrace V5 端到端搜索测试：去重计费、修复、崩溃恢复、收缩与剪枝反思、断点续跑一致性。"""

import json

import evotrace_v5.search as search_mod
from evotrace_v5 import Config, EvoTrace, Task


def _task():
    return Task(
        "Return a small positive number.",
        "def solve(x):\n    return x\n",
        lambda _src, fn: float(fn(1)),
        time_limit=5.0,
        example_code=None,
    )


def _cand(code, design="d", analysis="a"):
    return json.dumps({"analysis": analysis, "code": code, "design": design})


def _refl(exp):
    return json.dumps({"analysis": "checked", "experience": exp})


class ScriptedModel:
    model = "scripted-v5"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt, **_kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_mechanisms_and_pure_blackbox_lifecycle(tmp_path, monkeypatch):
    assert EvoTrace.mechanisms == (
        "tree_population",
        "reflective_elimination",
        "local_fork_allocation",
        "lineage_context",
    )
    assert EvoTrace.mechanisms_for(False) == (
        "tree_population",
        "tree_simplification",
        "local_fork_allocation",
        "lineage_context",
    )

    # Pin parent selection to root #1 so we deterministically exercise duplicate -> weak1 -> weak2 -> prune+reflect.
    monkeypatch.setattr(search_mod, "choose_parent", lambda tree, ledger, rng: (1, []))
    model = ScriptedModel(
        [
            _cand("def solve(x):\n    return 7.2\n", "root 7.2"),
            _cand("def solve(x):\n    # duplicate comment\n    return 7.2\n", "dup 7.2"),
            _cand("def solve(x):\n    return 8.0\n", "weak 8.0"),
            _cand("def solve(x):\n    return 8.2\n", "weak 8.2"),
            _refl("Both higher constant offsets regressed relative to 7.2."),
        ]
    )
    search = EvoTrace(
        _task(),
        model,
        config=Config(evaluation_budget=4, initial_size=1, seed=0),
        run_dir=tmp_path / "run",
    )
    res = search.run()
    assert res.status == "finished"
    assert res.evaluations == 4
    assert res.best.id == 1 and res.best.fitness == 7.2
    assert search.archive[2].duplicate_of == 1
    assert search.tree.parents == {1: 0}
    assert search.tree.node_notes[1] == "Both higher constant offsets regressed relative to 7.2."
    assert search.ledger.direct[1].count == 3


def test_path_contraction_and_reflection_in_live_search(tmp_path, monkeypatch):
    # Chain 1 (8.0) -> 2 (8.5) -> 3 (7.4): 3 beats 2, so 1 -> 2 -> 3 contracts to 1 -> 3 with edge note.
    parents = iter([1, 2])
    monkeypatch.setattr(search_mod, "choose_parent", lambda tree, ledger, rng: (next(parents), []))
    model = ScriptedModel(
        [
            _cand("def solve(x):\n    return 8.0\n", "init 8.0"),
            _cand("def solve(x):\n    return 8.5\n", "dip 8.5"),
            _cand("def solve(x):\n    return 7.4\n", "recover 7.4"),
            _refl("Lowering constant after the 8.5 dip improved score from 8.0 to 7.4."),
        ]
    )
    search = EvoTrace(
        _task(),
        model,
        config=Config(evaluation_budget=3, initial_size=1, seed=0),
        run_dir=tmp_path / "contract_run",
    )
    res = search.run()
    assert res.evaluations == 3
    assert search.tree.parents == {1: 0, 3: 1}
    assert search.tree.edge_notes[3] == "Lowering constant after the 8.5 dip improved score from 8.0 to 7.4."
    assert search.ledger.branch[3].count == 1
    assert search.ledger.branch[3].step_gains == [8.5 - 7.4]


def test_repair_is_charged_and_crash_reservation_not_double_spent(tmp_path):
    # Attempt 1 (Init): broken code (eval #1) -> repaired code (eval #2, score 5.0).
    model = ScriptedModel(
        [
            _cand("def solve(x):\n    return 1 / 0\n", "broken"),
            _cand("def solve(x):\n    return 5.0\n", "fixed"),
        ]
    )
    run_dir = tmp_path / "repair_run"
    search = EvoTrace(
        _task(),
        model,
        config=Config(evaluation_budget=2, initial_size=1, seed=0),
        run_dir=run_dir,
    )
    res = search.run()
    assert res.evaluations == 2
    assert res.best.fitness == 5.0
    assert search.attempts[0].repair_used is True

    # Simulate an interrupted evaluation persisted in pending.json and resumed with budget=3.
    pending = {
        "attempt": {
            "id": 2,
            "parent_id": 2,
            "operator": "Refine",
            "reference_id": None,
            "program_id": 3,
            "calls": [{"stage": "generate", "status": "success"}],
            "evaluations": [
                {
                    "id": 3,
                    "kind": "generation",
                    "program_id": 3,
                    "repair": False,
                    "status": "error",
                    "interrupted": True,
                    "evaluation": {
                        "status": "error",
                        "fitness": None,
                        "failure": "Interrupted evaluation; outcome unknown",
                    },
                }
            ],
            "repair_used": False,
            "finished_at": None,
            "record": {},
        },
        "failed_programs": [],
        "program": {
            "id": 3,
            "code": "def solve(x):\n    return 4.0\n",
            "design": "interrupted",
            "operator": "Refine",
            "parent_id": 2,
            "reference_id": None,
            "evaluation": {
                "status": "error",
                "fitness": None,
                "failure": "Interrupted evaluation; outcome unknown",
            },
            "analysis": "a",
            "duplicate_of": None,
        },
    }
    (run_dir / "pending.json").write_text(json.dumps(pending))
    resumed_model = ScriptedModel([])
    resumed = EvoTrace(
        _task(),
        resumed_model,
        config=Config(evaluation_budget=3, initial_size=1, seed=0),
        run_dir=run_dir,
        resume=True,
    )
    res2 = resumed.run()
    assert res2.evaluations == 3
    assert resumed_model.calls == 0
    assert res2.model_calls == 3


def test_resume_matches_uninterrupted_run_and_reflection_off_works(tmp_path):
    def make_script():
        return [
            _cand("def solve(x):\n    return 8.0\n", "d1"),
            _cand("def solve(x):\n    return 7.5\n", "d2"),
            _cand("def solve(x):\n    return 7.0\n", "d3"),
            _cand("def solve(x):\n    return 6.5\n", "d4"),
            _cand("def solve(x):\n    return 6.8\n", "d5"),
        ]

    full = EvoTrace(
        _task(),
        ScriptedModel(make_script()),
        config=Config(evaluation_budget=5, initial_size=1, seed=7),
        run_dir=tmp_path / "full",
        reflection=False,
    )
    res_full = full.run()

    script = make_script()
    part1 = EvoTrace(
        _task(),
        ScriptedModel(script[:3]),
        config=Config(evaluation_budget=3, initial_size=1, seed=7),
        run_dir=tmp_path / "split",
        reflection=False,
    )
    part1.run()
    part2 = EvoTrace(
        _task(),
        ScriptedModel(script[3:]),
        config=Config(evaluation_budget=5, initial_size=1, seed=7),
        run_dir=tmp_path / "split",
        resume=True,
        reflection=False,
    )
    res_split = part2.run()

    assert res_full.best.fitness == res_split.best.fitness
    assert full.tree.snapshot() == part2.tree.snapshot()
    assert full.ledger.direct.keys() == part2.ledger.direct.keys()
    assert full.ledger.branch.keys() == part2.ledger.branch.keys()


def test_init_exact_duplicate_does_not_count_toward_initial_size(tmp_path):
    model = ScriptedModel(
        [
            _cand("def solve(x):\n    return 5.0\n", "init1"),
            _cand("def solve(x):\n    # same code\n    return 5.0\n", "dup_init"),
            _cand("def solve(x):\n    return 4.0\n", "init2"),
        ]
    )
    search = EvoTrace(
        _task(),
        model,
        config=Config(evaluation_budget=3, initial_size=2, seed=0),
        run_dir=tmp_path / "init_dup",
    )
    res = search.run()
    assert res.evaluations == 3
    assert [a.operator for a in search.attempts] == ["Init", "Init", "Init"]
    assert sorted(search.tree.parents) == [1, 3]


def test_llm_post_retries_transient_http_429(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from evotrace_v5 import llm

    calls = {"n": 0}

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeOpener:
        def open(self, req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(
                    req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"rate limited")
                )
            return FakeResp(b'{"choices": [{"message": {"content": "ok"}}]}')

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: FakeOpener())
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    data = llm._post("http://example.test/v1/chat/completions", "k", {}, timeout=10.0)
    assert calls["n"] == 2 and data["choices"][0]["message"]["content"] == "ok"
