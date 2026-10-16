"""Budget and token counts used in research comparisons."""

import json
from types import SimpleNamespace

from baselines.shared.profiler import ProfilerBase
from core import Function
from tests.support import TinyEvaluation


def test_saved_candidates_keep_the_actual_budget_positions(tmp_path):
    profiler = ProfilerBase(tmp_path)
    for position, score in ((0, 5.0), (7, 4.0), (31, 3.0)):
        function = Function(name="score", args="x", body=f"    return {score}")
        function.score, function.operator = score, "Refine"
        profiler.register_function(function, program=str(function), budget_used=position)
    profiler.write_run_summary(budget_used=40)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["budget_used"] for event in events] == [0, 7, 31]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["budget_used"] == 40 and summary["candidate_count"] == 3


def test_token_cost_includes_retries_and_reflection_without_double_counting(tmp_path):
    profiler = ProfilerBase(tmp_path)
    calls = [
        {"transport_id": "retry", "usage": {}, "error": "timeout"},
        {"transport_id": "generated", "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        {"transport_id": "reflection", "usage": {"prompt_tokens": 40, "completion_tokens": 10}},
    ]
    for call in calls:
        call.update(prompt="task", response="reply")
    profiler.log_llm_call(transport_calls=calls[:2], stage="generate")
    profiler.log_llm_call(transport_calls=calls[:2], stage="generate")
    profiler.log_llm_call(transport_calls=calls[2:], stage="reflection")
    totals = json.loads((tmp_path / "token_usage.json").read_text())
    assert totals["calls"] == 3 and totals["calls_with_usage"] == 2
    assert totals["prompt_tokens"] == 140 and totals["completion_tokens"] == 60
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 3


def test_run_logs_keep_model_settings_without_credentials(tmp_path):
    profiler = ProfilerBase(tmp_path)
    profiler.record_parameters(
        SimpleNamespace(model="test-model", api_key="private-key"),
        TinyEvaluation(),
        SimpleNamespace(),
    )
    text = (tmp_path / "logs/run_log.txt").read_text()
    assert "test-model" in text and "private-key" not in text
