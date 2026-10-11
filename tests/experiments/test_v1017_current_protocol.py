"""V10.17 replay: old delivery/search, current evaluator, no V10.25 prompts."""

import random

import pytest

from benchmarks.tasks import CO_TASKS, FUNCTION_LIMITS, INSTANCE_LIMITS, training_task
from experiments.traceaad_v10_17.current_protocol import Config, CurrentProtocolV1017, PromptBuilder
from experiments.traceaad_v10_17.historical_delivery import DeliveryError, SourceError, parse_response
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v10_17.prompts import PromptBuilder as V1017Prompts
from traceaad.common.prompts import ContextTooLong


@pytest.mark.parametrize("task", CO_TASKS)
def test_current_protocol_preserves_operator_instructions(task):
    evaluation, _ = training_task(task, condition="traceaad")
    prompts = PromptBuilder(TokenLLM(), task, evaluation, {}, {}, Config(eval_timeout_seconds=INSTANCE_LIMITS[task]))
    for name in ("REFINE", "EXPLORE", "CROSSOVER", "REPAIR", "ANALYSIS", "FORMATION_INTRO"):
        assert getattr(prompts, name) == getattr(V1017Prompts, name)
    text = prompts.initial([])["prompt"]
    assert f"{FUNCTION_LIMITS[task]} seconds of CPU time" in text
    assert f"{INSTANCE_LIMITS[task]} seconds of wall time" in text
    assert "Lower is better" in text and "The whole evaluation must finish" not in text
    assert "the function may perform any computation on its inputs" in text
    assert "Analysis:" not in text


def test_original_delivery_completes_template_import_and_chooses_complete_block():
    template = "import math\ndef score(x):\n    pass"
    text = "Design: root\n```python\ndef score(x):\n    return math.sqrt(x)\n```\n```python\nx = 1\n```"
    code, idea, meta = parse_response(text, "stop", template)
    assert code.startswith("import math\n") and idea == "root"
    assert meta["template_additions"] == ["math"]
    assert "import math" not in meta["submitted_code"]
    namespace = {}
    exec(code, namespace)
    assert namespace["score"](4) == 2
    with pytest.raises(DeliveryError):
        parse_response(text, "length", template)
    with pytest.raises(SourceError):
        parse_response("```python\ndef score(y):\n    return y\n```", "stop", template)


def test_search_rng_repair_freeze_and_resume(tmp_path):
    bad = "```python\ndef score(x):\n    raise ValueError('bad')\n```"
    search = CurrentProtocolV1017(evaluation=TinyEvaluation(), llm=TokenLLM(response(1), bad, response(2)),
                                  run_dir=tmp_path, config=Config(roots=1, budget=3))
    assert search.parent_rng.getstate() == random.Random("v10.17:0:parent").getstate()
    assert search.action_rng.getstate() == random.Random("v10.17:0:action").getstate()
    assert search.reference_rng.getstate() == random.Random("v10.17:0:reference").getstate()
    for i in range(100):
        assert search._developed(i) == (random.Random(f"v10.17:0:develop:{i}").random() < .125)
    search._roots()
    search._roots()
    parent = search.archive[1]
    search._attempt(search.prompts.build("Refine", parent), parent=parent)
    child = search.archive[3]
    history = search.prompts.build("Refine", child)["prompt"]
    assert "then Repair" in history and "previous → current" in history
    assert "+    raise ValueError" not in history
    assert search.attempts == 3
    search._freeze()
    assert search.progress.selected_id == 3
    restored = CurrentProtocolV1017(evaluation=TinyEvaluation(), llm=TokenLLM(), run_dir=tmp_path,
                                    config=Config(roots=1, budget=3))
    assert restored.attempts == 3
    with pytest.raises(ValueError, match="same mechanism"):
        CurrentProtocolV1017(evaluation=TinyEvaluation(), llm=TokenLLM(), run_dir=tmp_path,
                            config=Config(roots=1, budget=4))


def test_timeout_reports_actual_instance_failure(tmp_path):
    search = CurrentProtocolV1017(evaluation=TinyEvaluation(), llm=TokenLLM(), run_dir=tmp_path)
    failed = {"failure": {"kind": "timeout", "error": "instance 4 used more than 2s of CPU inside the function"}}
    assert search.prompts.failure(failed) == failed["failure"]["error"]


def test_context_overflow_does_not_introduce_an_explore_fallback(tmp_path, monkeypatch):
    search = CurrentProtocolV1017(evaluation=TinyEvaluation(), llm=TokenLLM(response(1)),
                                  run_dir=tmp_path, config=Config(roots=1, budget=5))
    search._roots()
    search._roots()
    requested = []

    def overflow(action, *args, **kwargs):
        requested.append(action)
        raise ContextTooLong("required history")

    monkeypatch.setattr(search.prompts, "build", overflow)
    search._ordinary_search("Refine")
    assert requested == ["Refine"]
    assert search.progress.too_long == [1] and search.attempts == 1
