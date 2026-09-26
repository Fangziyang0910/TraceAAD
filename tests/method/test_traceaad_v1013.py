import json
import random
from types import SimpleNamespace

import pytest

from experiments.infra.artifacts import load_scored_samples
from traceaad.v10_13 import TraceAADV1013
from traceaad.v10_13.parsing import parse_candidate, template_target
from traceaad.v10_13.prompts import PromptBuilder
from traceaad.v10_13.selection import (
    PARENT_UNIFORM_PROBABILITY,
    ess,
    mix_uniform,
    quality_distribution,
    sample_parent,
    sample_reference,
)
from traceaad.v10_13.tree import Node
from traceaad.v10_13.storage import RunStorage, read_journal
from tests.support import FakeLLM, TinyEvaluation, response


def make_method(path, llm, *, budget=1, n_roots=1, seed=0, init_mode=None):
    return TraceAADV1013(
        evaluation=TinyEvaluation(),
        llm=llm,
        run_dir=path,
        budget=budget,
        n_roots=n_roots,
        seed=seed,
        **({"init_mode": init_mode} if init_mode is not None else {}),
    )


@pytest.mark.parametrize("mode,n_roots,expected", [
    (None, 8, [0, 0, 0, 0, 4, 5, 6, 7]),
    ("independent", 8, [0] * 8),
    ("sequential", 8, list(range(8))),
    ("hybrid", 5, [0, 0, 0, 3, 4]),
])
def test_initialization_mode_controls_root_context(tmp_path, mode, n_roots, expected):
    llm = FakeLLM(*(response(i) for i in range(n_roots)))
    method = make_method(tmp_path, llm, budget=n_roots, n_roots=n_roots,
                         init_mode=mode)
    method.run()

    assert [prompt.count("# Previous Initial Algorithm") for prompt, _ in llm.calls] == expected
    assert len(method.tree.roots) == n_roots
    assert all(method.tree.nodes[root_id].parent_id is None
               for root_id in method.tree.roots)


def test_failed_candidate_does_not_advance_hybrid_boundary(tmp_path):
    llm = FakeLLM("invalid output", *(response(i) for i in range(8)))
    method = make_method(tmp_path, llm, budget=8, n_roots=8)
    method.run()

    assert [prompt.count("# Previous Initial Algorithm") for prompt, _ in llm.calls] == [
        0, 0, 0, 0, 0, 4, 5, 6, 7,
    ]


def test_runner_checks_resume_settings_and_skips_finished_run(tmp_path, monkeypatch):
    from experiments.traceaad_v10_13 import run as runner

    args = runner.build_parser().parse_args(["--task", "tsp_construct"])
    assert (args.init_mode, args.n_roots) == ("hybrid", 8)

    (tmp_path / "run_config.json").write_text(json.dumps({
        "seed": 0, "method_params": {"n_roots": 8, "init_mode": "sequential"},
    }))
    llm = SimpleNamespace(close=lambda: None)
    context = SimpleNamespace(resumed=True, run_dir=tmp_path, llm=llm,
                              evaluation=TinyEvaluation())
    monkeypatch.setattr(runner, "setup_experiment_run", lambda *args, **kwargs: context)
    with pytest.raises(ValueError, match="resume with --init-mode sequential"):
        runner.main(["--task", "tsp_construct"])
    with pytest.raises(ValueError, match="resume with --n-roots 8"):
        runner.main(["--task", "tsp_construct", "--init-mode", "sequential",
                     "--n-roots", "4"])

    with pytest.raises(ValueError, match="seed=0"):
        runner.main(["--task", "tsp_construct", "--init-mode", "sequential", "--seed", "7"])

    RunStorage(tmp_path).save_summary({"status": "finished", "budget": 1000})
    runner.main(["--task", "tsp_construct"])


def test_parent_distribution_keeps_ess_and_uniform_exploration():
    nodes = [Node(i, f"def score(x):\n    return {i}", "", i) for i in range(12)]
    quality = quality_distribution(nodes)
    mixed = mix_uniform(quality, PARENT_UNIFORM_PROBABILITY)

    assert abs(ess(quality) - 8.0) < 1e-6
    assert ess(mixed) > ess(quality)
    assert sample_parent(nodes, random.Random(0)) in nodes


def test_reference_is_a_different_program():
    nodes = [
        Node(0, "def score(x):\n    return x", "", 1),
        Node(1, "def score(x):\n    return x", "", 9),
        Node(2, "def score(x):\n    return x + 1", "", 2),
    ]
    assert sample_reference(nodes, nodes[0], random.Random(2)).id == 2


def test_parser_accepts_python_block_and_preserves_template_helpers():
    template = "import math\n\ndef score(x):\n    pass"
    interface, _ = template_target(template)
    parsed, error = parse_candidate(response("math.sqrt(x)"), "stop", interface, template)

    assert error is None
    assert parsed.idea == "return math.sqrt(x)"
    assert "import math" in parsed.program_code
    assert "return math.sqrt(x)" in parsed.program_code


def test_parser_applies_one_simple_edit():
    template = "def score(x):\n    pass"
    interface, _ = template_target(template)
    parent = "def score(x):\n    return x"
    edit = json.dumps({
        "mode": "edit",
        "idea": "add one",
        "edits": [{"search": "return x", "replacement": "return x + 1"}],
    })

    parsed, error = parse_candidate(edit, "stop", interface, template, base_code=parent)

    assert error is None
    assert parsed.idea == "add one"
    assert "return x + 1" in parsed.program_code


def test_parser_rejects_wrong_target_signature():
    template = "def score(x):\n    pass"
    interface, _ = template_target(template)
    parsed, error = parse_candidate(
        "def score(x, y):\n    return x + y", "stop", interface, template,
    )

    assert parsed is None
    assert error.startswith("signature_error")


def test_operator_prompts_only_include_relevant_context(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)))
    root = method.tree.add(
        code="def score(x):\n    return 1",
        idea="root",
        fitness=1,
        parent_id=None,
        operator="Init",
    )
    child = method.tree.add(
        code="def score(x):\n    return 2",
        idea="child",
        fitness=2,
        parent_id=root.id,
        operator="Refine",
    )
    reference = method.tree.add(
        code="def score(x):\n    return 3",
        idea="reference",
        fitness=3,
        parent_id=None,
        operator="Init",
    )
    builder = PromptBuilder(
        method.llm,
        method.task_prompt,
        max_tokens=20000,
        history_depth=3,
        lookup=method.tree.nodes.get,
        all_nodes=method.tree.all_nodes,
    )

    refine = builder.build_development(child, "Refine")
    fuse = builder.build_development(child, "Fuse", reference)
    pivot = builder.build_development(child, "Pivot", reference)

    assert "Recent Design History" in refine.prompt
    assert "Reference Algorithm" in fuse.prompt
    assert reference.code not in pivot.prompt


def test_initial_prompt_does_not_offer_parent_edit(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)))
    prompt = method.prompts.build_initial().prompt

    assert '"mode":"edit"' not in prompt


def test_run_saves_results_with_a_small_state(tmp_path):
    method = make_method(
        tmp_path,
        FakeLLM(response(1), response(2), response(3)),
        budget=3,
    )
    method.run()

    state = method.storage.load_state()
    summary = json.loads(method.storage.summary_path.read_text())
    events = method.storage.records("events")

    assert state["budget_used"] == len(method.storage.records("nodes")) == 3
    assert len(events) == 3
    calls = method.storage.records("calls")
    assert len(calls) == 3
    assert calls[0]["prompt"] == method.llm.calls[0][0]
    assert calls[0]["response"] == response(1)
    assert all("state" in record and "node" in record
               for record in read_journal(method.storage.path) if record["kind"] == "candidate")
    assert summary["status"] == "finished"
    assert summary["best"]["fitness"] == 3
    assert summary["best"]["evaluation_id"] == 3
    assert load_scored_samples(tmp_path)[0]["sample_order"] == 3


def test_invalid_output_does_not_spend_evaluation_budget(tmp_path):
    method = make_method(
        tmp_path,
        FakeLLM("This is not Python.", response(1)),
        budget=1,
    )
    method.run()

    events = method.storage.records("events")
    assert [event["status"] for event in events] == ["invalid_output", "ok"]
    assert [event["budget_used"] for event in events] == [0, 1]


def test_failed_evaluation_spends_budget_and_search_continues(tmp_path):
    method = make_method(
        tmp_path,
        FakeLLM(response("1 / 0"), response(2)),
        budget=2,
    )
    method.run()

    events = method.storage.records("events")
    assert [event["status"] for event in events] == ["eval_failed", "ok"]
    assert "def score(x):" in events[0]["code"]
    assert method.evaluations_used == 2


def test_resume_reads_saved_state_and_finishes_remaining_budget(tmp_path):
    make_method(tmp_path, FakeLLM(response(1)), budget=1).run()
    resumed = make_method(tmp_path, FakeLLM(response(2)), budget=2)
    resumed.run()

    assert resumed.evaluations_used == 2
    assert len(resumed.tree.nodes) == 2
    assert len(resumed.storage.records("events")) == 2


def test_missing_summary_is_rebuilt_from_completed_search(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    method.run()
    method.storage.summary_path.unlink()

    resumed = make_method(tmp_path, FakeLLM(), budget=1)
    resumed.run()

    assert resumed.storage.load_summary()["status"] == "finished"
    assert len(resumed.storage.records("calls")) == 1


def test_failed_model_call_is_preserved_and_candidate_retries(tmp_path):
    with pytest.raises(StopIteration):
        make_method(tmp_path, FakeLLM(), budget=1).run()

    resumed = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    resumed.run()

    assert [call["candidate_id"] for call in resumed.storage.records("calls")] == [1, 1]
    assert [event["candidate_id"] for event in resumed.storage.records("events")] == [1]
    assert resumed.storage.load_state()["budget_used"] == 1


def test_uncertain_evaluation_blocks_automatic_resume(tmp_path, monkeypatch):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)

    def interrupt_before_commit(*_args):
        raise OSError("interrupted after evaluator call")

    monkeypatch.setattr(method.storage, "commit_candidate", interrupt_before_commit)
    with pytest.raises(OSError, match="interrupted after evaluator call"):
        method.run()

    pending = method.storage.load_state()["pending_evaluation"]
    assert pending["candidate_id"] == pending["evaluation_id"] == 1
    assert pending["operator"] == "Init"
    assert "def score(x):" in pending["code"]
    assert len(method.storage.records("calls")) == 1
    assert method.storage.records("events") == []
    with pytest.raises(RuntimeError, match="evaluation result is uncertain"):
        make_method(tmp_path, FakeLLM(response(2)), budget=1).run()


def test_damaged_journal_blocks_resume(tmp_path):
    storage = RunStorage(tmp_path)
    storage.save_state({"candidate_count": 0, "budget_used": 0})
    with storage.path.open("ab") as handle:
        handle.write(b'{"kind":"candidate","candidate_id":')

    with pytest.raises(RuntimeError, match="incomplete search record"):
        storage.load_state()

    storage.path.write_text('{"kind":"state","state":{}}\nnot json\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid search record"):
        storage.load_state()
