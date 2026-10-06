import json
import pytest
from traceaad.common.storage import save_heldout

from experiments.infra.monitor_results import load_batch_heldout, load_selection
from benchmarks.tasks import scale_of_split
from experiments.monitor import ResultsMonitor


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_split_names_map_to_canonical_scales():
    assert scale_of_split("cvrp_aco", "test_100") == 100
    assert scale_of_split("tsp_construct", "eval") == 50
    assert scale_of_split("vrptw_construct", "eval_200") == 200
    assert scale_of_split("online_bin_packing", "eval_10000_500") == "10k_500"
    with pytest.raises(ValueError):
        scale_of_split("cvrp_aco", "paper_test_50")


def test_heldout_variants_formats_and_skipped_sources(tmp_path):
    batch = tmp_path / "traceaad_vx"
    def result(task, name, scale, score, variant=""):
        save_heldout(batch / task / name, {"task": task, "scale": str(scale), "fitness": score,
                     "verification": "legacy", "key": None, "node_id": None}, variant)
    result("cvrp_aco", "a_rep1", 50, -9, "alpha")
    result("cvrp_aco", "a_rep1", 50, -8, "alpha")
    result("tsp_construct", "b_rep2", 100, -8.5, "beta")
    result("online_bin_packing", "c_rep1", "1k_100", -412)
    # Unnormalized old exports must not enter the current reader.
    write_json(batch / "heldout_old_beta/cvrp_aco/results.json", {
        "results_by_split": {"test_50": {"results": [{"run_name": "b_rep1", "eval_score": -1}]}}})
    heldout = load_batch_heldout(batch)
    assert heldout["alpha"]["cvrp_aco"]["runs"] == {"a_rep1": {50: -8}}
    assert heldout["beta"]["tsp_construct"]["runs"] == {"b_rep2": {100: -8.5}}
    assert "cvrp_aco" not in heldout["beta"]
    assert heldout[""]["online_bin_packing"]["runs"] == {"c_rep1": {"1k_100": -412}}


def test_native_heldout_and_selection_ties(tmp_path):
    run = tmp_path / "b" / "op_aco" / "r_rep1"
    save_heldout(run, {"task": "op_aco", "scale": "100", "fitness": 30.5, "verification": "legacy"})
    assert load_batch_heldout(tmp_path / "b")[""]["op_aco"]["runs"] == {"r_rep1": {100: 30.5}}
    write_json(run / "selection.json", {"selected_node": 2, "results": [
        {"node_id": 1, "fitness": 15.0}, {"node_id": 2, "fitness": 15.0}, {"node_id": 3, "fitness": None}]})
    assert load_selection(run) == {"fitness": 15.0, "finalists": 3, "failed": 1, "ties": 2, "distinct": 1}


def test_state_signature_tracks_journal_changes(tmp_path):
    run = tmp_path / "traceaad_vx" / "op_aco" / "rep1"
    write_json(run / "run_config.json", {"task": "op_aco", "repeat": 1})
    (run / "events.jsonl").write_text("", encoding="utf-8")
    monitor = ResultsMonitor(tmp_path)
    before = monitor.state_signature("traceaad_vx")
    (run / "events.jsonl").write_text('{"kind":"candidate"}\n', encoding="utf-8")
    assert monitor.state_signature("traceaad_vx") != before
