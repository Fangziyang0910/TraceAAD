import json

from experiments.traceaad_v10_11 import monitor


def _manifest(batch, created_at, *, status=None):
    payload = {
        "batch": batch,
        "created_at": created_at,
        "session_prefix": f"p_{batch}",
        "plan": [{"run_name": f"{batch}_tsp_v1011_rep1"}],
    }
    if status:
        payload["status"] = status
    return payload


def _write(root, name, payload):
    (root / f"batch_{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_v1011_monitor_derives_versions_from_batch_manifests(tmp_path):
    old = _manifest("20260913_v1011_q36", "2026-09-13T09:00:00")
    baseline = _manifest("20260914_v1011_q38_restart2", "2026-09-14T10:20:00")
    history = _manifest("20260914_v1011_q38_history_code", "2026-09-14T10:50:00")
    superseded = _manifest("20260914_v1011_q38", "2026-09-14T09:00:00", status="superseded")
    for name, payload in (("old", old), ("baseline", baseline), ("history", history), ("sup", superseded)):
        _write(tmp_path, name, payload)

    engine = monitor.MonitorDataEngine(results_root=tmp_path)
    versions = engine.get_available_versions()
    assert [v["id"] for v in versions] == [
        "20260914_v1011_q38_history_code",
        "20260914_v1011_q38_restart2",
        "20260913_v1011_q36",
    ]
    assert versions[0]["is_latest"] is True
    assert all(v["is_latest"] is False for v in versions[1:])


def test_v1011_monitor_resolves_version_meta_and_manifest_per_batch(tmp_path):
    baseline = _manifest("20260914_v1011_q38_restart2", "2026-09-14T10:20:00")
    history = _manifest("20260914_v1011_q38_history_code", "2026-09-14T10:50:00")
    _write(tmp_path, "baseline", baseline)
    _write(tmp_path, "history", history)

    engine = monitor.MonitorDataEngine(results_root=tmp_path)

    vid, root, prefix, badge = engine._resolve_version_meta(baseline["batch"])
    assert (vid, badge, prefix) == (baseline["batch"], baseline["batch"], "p_20260914_v1011_q38_restart2")
    assert engine._load_latest_batch_manifest(tmp_path, baseline["batch"])["batch"] == baseline["batch"]
    assert engine._load_latest_batch_manifest(tmp_path, history["batch"])["batch"] == history["batch"]

    # No version given: the newest manifest wins.
    vid, _, prefix, _ = engine._resolve_version_meta(None)
    assert (vid, prefix) == (history["batch"], "p_20260914_v1011_q38_history_code")
    assert engine.get_available_versions()[0]["id"] == history["batch"]


def test_v1011_monitor_scans_persisted_runs_without_run_config(tmp_path):
    batch = "20260918_bc_B"
    _write(tmp_path, batch, {
        "batch": batch,
        "created_at": "2026-09-18T23:00:00",
        "method": "bc_b",
        "session_prefix": "bcB",
        "plan": [{"task": "tsp_construct", "repeat": 1,
                   "run_name": f"{batch}_tsp_bcb_rep1"}],
    })
    run_dir = tmp_path / "tsp_construct" / f"{batch}_tsp_bcb_rep1"
    run_dir.mkdir(parents=True)
    (run_dir / "tree_state.json").write_text(json.dumps({
        "started_at": "2026-09-18T23:01:00",
        "budget_used": 17,
    }))
    (run_dir / "events.jsonl").write_text("")

    overview = monitor.MonitorDataEngine(results_root=tmp_path).get_overview(batch)

    assert overview["global_summary"]["total_runs"] == 1
    assert overview["global_summary"]["total_evals"] == 17
    assert overview["tasks"][0]["runs"][0]["status"] == "stalled"
