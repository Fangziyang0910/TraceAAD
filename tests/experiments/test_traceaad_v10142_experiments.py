"""Offline CLI and real task adapters; no model service is needed."""

from dataclasses import replace
import json

import pytest

from benchmarks.cvrp_aco import CVRPACOEvaluation
from benchmarks.online_bin_packing import OBPEvaluation
from benchmarks.op_aco import OPACOEvaluation
from benchmarks.tsp_construct import TSPEvaluation
from benchmarks.vrptw_construct import VRPTWEvaluation
from experiments.traceaad_v10_14_2 import run as runner
from tests.method.test_traceaad_v10142 import TokenLLM as FakeLLM
from traceaad.v10_14_2 import Config, TraceAADV10142
from traceaad.v10_14_2.evaluation import fingerprint


def small_task(task, seed=10):
    if task == "tsp_construct":
        return TSPEvaluation(n_instance=2, problem_size=10, seed=seed)
    if task == "vrptw_construct":
        return VRPTWEvaluation(n_instance=2, problem_size=10, seed=seed)
    if task == "online_bin_packing":
        return OBPEvaluation(dataset_specs=[{"n_instances": 1, "n_items": 64, "capacities": [100, 500]}], seed=seed)
    cls = CVRPACOEvaluation if task == "cvrp_aco" else OPACOEvaluation
    evaluation = cls(split="train" if seed == 10 else "val_50", n_ants=3, n_iterations=2, n_workers=1)
    evaluation._datasets = evaluation._datasets[:2]
    evaluation.n_instance = 2
    return evaluation


@pytest.mark.parametrize("task", ["tsp_construct", "vrptw_construct", "online_bin_packing", "cvrp_aco", "op_aco"])
def test_real_task_search_probes_and_selection(tmp_path, task):
    train, selection = small_task(task), small_task(task, seed=11)
    responses = ["```python\n" + train.template_program.strip() + "\n```"] * 3
    method = TraceAADV10142(evaluation=train, selection_evaluation=selection,
        llm=FakeLLM(*responses), run_dir=tmp_path, task=task,
        config=Config(budget=3, max_evaluations=3, init_proposals=1))
    probes_before = fingerprint(method.probes)
    result = method.run()
    assert result["status"] == "finished", result
    assert result["descriptor"] == "diagnostic_training_probes"
    assert result["best"]["profile"]
    assert len(method.frontier.regions) == 1  # identical source doesn't manufacture diversity
    assert method.ledger.candidates == 3 and method.ledger.evaluations == 1
    assert fingerprint(method.probes) == probes_before
    evaluation = method.facts.tables["evaluation"][1]
    assert evaluation["result"]["probe_error"] is None
    assert evaluation["result"]["probe_calls"] == len(method.probes)
    assert result["selection_evaluations"] == 1


def test_cli_dry_run_has_no_model_calls(capsys):
    runner.main(["--task", "tsp_construct", "--budget", "32", "--evaluation-seeds", "7", "8", "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["config"]["max_evaluations"] == 64
    assert result["selection"]["seed"] not in {2024, 2025}


@pytest.mark.parametrize("task", ["tsp_construct", "vrptw_construct", "online_bin_packing", "cvrp_aco", "op_aco"])
def test_selection_builder_does_not_use_final_test_split(task):
    train = small_task(task)
    selection = runner.selection_task(task, train)
    assert fingerprint(train._datasets) != fingerprint(selection._datasets)
    if task in {"cvrp_aco", "op_aco"}:
        assert selection.split == "val_50"
        assert selection.timeout_seconds == train.timeout_seconds * max(1, selection.n_instance/train.n_instance)
    else:
        assert selection.seed == runner.SELECTION_SEED


def test_protocol_rejects_invalid_caps():
    with pytest.raises(ValueError):
        Config(trial_fraction=.3)
    with pytest.raises(ValueError):
        Config(evaluation_seeds=(1, 1))
    with pytest.raises(ValueError):
        replace(Config(), max_seconds=float("nan"))


def test_monitor_understands_native_candidate_budget(tmp_path):
    from experiments.monitor import ResultsMonitor
    from tests.support import TinyEvaluation, response
    directory = tmp_path / "traceaad_v10_14_2" / "tsp_construct" / "test"
    method = TraceAADV10142(evaluation=TinyEvaluation(), llm=FakeLLM(response(1)),
        run_dir=directory, config=Config(budget=2, max_evaluations=2, init_proposals=1))
    method._initialize()
    (directory / "run_config.json").write_text(json.dumps({"task": "tsp_construct", "method": "v1014_2"}))
    monitor = ResultsMonitor(tmp_path)
    view = monitor.overview("traceaad_v10_14_2")
    assert view["summary"]["running"] == 1
    assert view["summary"]["budget_used"] == 1
    assert monitor._recorded_progress(directory) == (1, 1)


def test_cli_experimental_switches_are_explicit():
    args = runner.build_parser().parse_args(['--task', 'tsp_construct', '--online-revalidation',
        '--recheck-fraction', '.1', '--fixed-three-step-commitment', '--trial-fraction', '.2'])
    config = runner.config_from_args(args)
    assert config.online_revalidation and config.fixed_three_step_commitment
    assert config.challenger_gap == .1  # legacy option kept, not silently rescaled


def test_final_selected_program_is_available_to_shared_heldout(tmp_path):
    from experiments.infra.artifacts import pick_best_sample
    train, selection = small_task('tsp_construct'), small_task('tsp_construct', seed=11)
    responses = ['```python\n' + train.template_program.strip() + '\n```'] * 3
    method = TraceAADV10142(evaluation=train, selection_evaluation=selection, llm=FakeLLM(*responses),
        run_dir=tmp_path, task='tsp_construct', config=Config(budget=3, max_evaluations=3, init_proposals=1))
    result = method.run()
    sample, _ = pick_best_sample(tmp_path)
    assert sample['program'] == result['best']['code']
