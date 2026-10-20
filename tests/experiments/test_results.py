"""A saved winner and its test scores must refer to the same program individual."""

import json
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import urlopen

import pytest

from experiments.infra import evaluate
from experiments.infra.artifacts import pick_best_sample
from experiments.visualization import ResultsMonitor
from experiments.visualization.server import make_request_handler
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad import Config, TraceAAD
from traceaad.storage import write_json


@pytest.mark.parametrize("fail", [False, True])
def test_heldout_evaluates_the_frozen_training_winner(tmp_path, monkeypatch, fail):
    config = Config(budget=3, roots=3)
    method = TraceAAD(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response(2), response(5), response(1)),
        run_dir=tmp_path,
        config=config,
    )
    summary = method.run()
    write_json(
        tmp_path / "run_config.json", {"task": "tsp_construct", "method_params": asdict(config)}
    )
    frozen = (tmp_path / "best_program.py").read_text()

    class TestEvaluation(TinyEvaluation):
        def evaluate_program(self, source, function):
            if fail:
                raise ValueError("the selected program failed on heldout")
            return function(1)

    monkeypatch.setattr(evaluate, "heldout_task", lambda *args: TestEvaluation())
    result = evaluate.evaluate_run(tmp_path, ["eval_50"], condition="traceaad")[0]
    assert summary["best"]["fitness"] == -5 and result["fitness"] == (None if fail else 5)
    assert result["node_id"] == summary["best"]["id"] == 2
    assert (tmp_path / "best_program.py").read_text() == frozen
    monkeypatch.setattr(
        evaluate, "heldout_evaluator", lambda *a, **k: pytest.fail("repeat evaluation")
    )
    assert evaluate.evaluate_run(tmp_path, ["eval_50"])[0] == result


def test_result_reader_and_monitor_follow_individual_ids_and_generation_budget(tmp_path):
    directory = tmp_path / "traceaad" / "tsp_construct" / "run"
    method = TraceAAD(
        evaluation=TinyEvaluation(),
        llm=TokenLLM("{bad", response(2, analysis="first"), response(2, analysis="second")),
        run_dir=directory,
        config=Config(budget=3, roots=2),
    )
    method.run()
    write_json(directory / "run_config.json", {"task": "tsp_construct", "budget": 3})
    chosen, samples = pick_best_sample(directory)
    assert chosen["node_id"] == 1 and chosen["sample_order"] == 2
    assert len(samples) == 2 and samples[0]["node_id"] != samples[1]["node_id"]
    monitor = ResultsMonitor(tmp_path)
    detail = monitor.run_detail("traceaad", "tsp_construct", "run")
    assert detail["budget_used"] == 3
    assert detail["best"]["code"] == method.programs[1].code
    assert detail["curve"][-1]["fitness"] == -2
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_request_handler(monitor))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for path in ["/", "/app.js", "/style.css"]:
            with urlopen(base + path) as result:
                assert result.status == 200 and result.read()
        with urlopen(base + "/api/run?batch=traceaad&task=tsp_construct&name=run") as result:
            assert json.load(result)["best"]["id"] == chosen["node_id"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before


def test_comparison_never_averages_across_models(tmp_path):
    models = {"rep1": "qwen3.8-27b-awq", "rep2": "Qwen3.8-27B", "pilot": "deepseek-v4.1-flash"}
    for name, model in models.items():
        write_json(tmp_path / "eoh" / "tsp_construct" / name / "run_config.json",
                   {"task": "tsp_construct", "budget": 3, "llm": {"model": model}})
    monitor = ResultsMonitor(tmp_path)
    cohorts = {c["id"]: c["runs"] for c in monitor.cohorts()}
    assert cohorts == {"eoh@qwen3.8-27b": 2, "eoh@deepseek-v4.1-flash": 1}
    compared = monitor.compare(["eoh@qwen3.8-27b"])["tasks"][0]["cohorts"][0]["runs"]
    assert sorted(run["name"] for run in compared) == ["rep1", "rep2"]
