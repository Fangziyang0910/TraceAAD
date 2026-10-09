import json

import pytest

from experiments.decision_model.train import check_encoding, load_splits


def write_splits(directory, runs=("a", "b", "c")):
    for name, run in zip(("train", "calibration", "test"), runs):
        row = {"run_id": run, "state": "a program and its past attempts",
               "questions": {"operator": {"type": "choice", "criteria": {"Refine": "improve"}}},
               "gold": {"operator": "Refine"}}
        (directory / f"{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_loads_disjoint_runs(tmp_path):
    write_splits(tmp_path)
    assert set(load_splits(tmp_path)) == {"train", "calibration", "test"}


def test_rejects_a_search_run_shared_by_train_and_test(tmp_path):
    write_splits(tmp_path, ("same-run", "calibration", "same-run"))
    with pytest.raises(ValueError, match="multiple splits"):
        load_splits(tmp_path)


def test_rejects_missing_run_id(tmp_path):
    write_splits(tmp_path, ("", "b", "c"))
    with pytest.raises(ValueError, match="nonempty run_id"):
        load_splits(tmp_path)


def test_rejects_an_unanswered_question(tmp_path):
    write_splits(tmp_path)
    path = tmp_path / "train.jsonl"
    row = json.loads(path.read_text())
    row["gold"] = {}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="gold must answer every question"):
        load_splits(tmp_path)


@pytest.mark.parametrize("field", ["skipped", "truncated"])
def test_rejects_lost_training_information(field):
    report = {"skipped": 0, "truncated": 0}
    report[field] = 1
    with pytest.raises(ValueError, match="refusing"):
        check_encoding("train", [object()], report)
