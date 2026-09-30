import json

from experiments.infra.monitor_results import load_batch_heldout, load_selection, scale_of_split
from experiments.monitor import ResultsMonitor


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_split_names_map_to_canonical_scales():
    assert scale_of_split("cvrp_aco", "test_100") == 100
    assert scale_of_split("tsp_construct", "eval") == 50
    assert scale_of_split("vrptw_construct", "eval_200") == 200
    assert scale_of_split("online_bin_packing", "eval_10000_500") == "10k_500"
    assert scale_of_split("cvrp_aco", "paper_test_50") is None


def test_heldout_variants_formats_and_skipped_sources(tmp_path):
    batch = tmp_path / "traceaad_vx"
    result = lambda name, score: {"run_name": name, "eval_score": score}
    # Two variants in one batch; the redo of a variant (newer date) wins.
    write_json(batch / "heldout_20260901_alpha/cvrp_aco/results.json", {
        "created_at": "2026-09-01", "results_by_split": {
            "test_50": {"results": [result("a_rep1", -9.0)]},
            "paper_test_50": {"results": [result("a_rep1", -1.0)]}}})
    write_json(batch / "heldout_20260902_alpha/cvrp_aco/results.json", {
        "created_at": "2026-09-02", "results_by_split": {"test_50": {"results": [result("a_rep1", -8.0)]}}})
    write_json(batch / "heldout_20260901_beta/tsp_construct/results.json", {
        "eval_results_by_size": {"tsp100": {"problem_size": 100, "results": [result("b_rep2", -8.5)]}}})
    # Partial single-rep reruns and budget-truncated evaluations are not formal results.
    write_json(batch / "heldout_20260901_beta/cvrp_aco_rep1/results.json", {
        "results_by_split": {"test_50": {"results": [result("b_rep1", -1.0)]}}})
    write_json(batch / "online_bin_packing/eval_best_budget500/results.json", {
        "eval_results_by_scale": {"x": {"n_items": 1000, "capacity": 100, "results": [result("c_rep1", -1.0)]}}})
    write_json(batch / "online_bin_packing/eval_best_formal/results.json", {
        "eval_results_by_scale": {"x": {"n_items": 1000, "capacity": 100, "results": [result("c_rep1", -412.0)]}}})
    heldout = load_batch_heldout(batch)
    assert heldout["alpha"]["cvrp_aco"]["runs"] == {"a_rep1": {50: -8.0}}
    assert heldout["beta"]["tsp_construct"]["runs"] == {"b_rep2": {100: -8.5}}
    assert "cvrp_aco" not in heldout["beta"]
    assert heldout[""]["online_bin_packing"]["runs"] == {"c_rep1": {"1k_100": -412.0}}


def test_native_heldout_and_selection_ties(tmp_path):
    run = tmp_path / "b" / "op_aco" / "r_rep1"
    write_json(run / "heldout_test_100.json", {"task": "op_aco", "split": "test_100", "fitness": 30.5})
    assert load_batch_heldout(tmp_path / "b")[""]["op_aco"]["runs"] == {"r_rep1": {100: 30.5}}
    write_json(run / "selection.json", {"selected_node": 2, "results": [
        {"node_id": 1, "fitness": 15.0}, {"node_id": 2, "fitness": 15.0}, {"node_id": 3, "fitness": None}]})
    assert load_selection(run) == {"fitness": 15.0, "finalists": 3, "failed": 1, "ties": 2, "distinct": 1}


def test_state_signature_tracks_journal_changes(tmp_path):
    run = tmp_path / "traceaad_vx" / "op_aco" / "rep1"
    write_json(run / "run_config.json", {"task": "op_aco", "repeat": 1})
    (run / "search.jsonl").write_text("", encoding="utf-8")
    monitor = ResultsMonitor(tmp_path)
    before = monitor.state_signature("traceaad_vx")
    (run / "search.jsonl").write_text('{"kind":"candidate"}\n', encoding="utf-8")
    assert monitor.state_signature("traceaad_vx") != before
