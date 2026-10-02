"""Archive paths, contents and independent writes survive storage cleanup."""
import json

import pytest

from experiments.infra.deduplicate_archives import apply, plan, restore


def archive(tmp_path):
    directory = tmp_path / "archive"
    directory.mkdir()
    for name, value in [("pop_1.json", [1, 2]), ("pop_2.json", [1, 2]), ("pop_3.json", [3])]:
        (directory / name).write_text(json.dumps(value))
    return directory


def test_iteration_history_and_independent_writes_after_restore(tmp_path):
    directory = archive(tmp_path)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    manifest = tmp_path / "manifest.json"
    result = apply(plan([directory]), manifest)
    assert result["n_linked"] == 1
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert [json.loads(p.read_text()) for p in sorted(directory.iterdir())] == [[1, 2], [1, 2], [3]]
    assert (directory / "pop_1.json").stat().st_ino == (directory / "pop_2.json").stat().st_ino
    restore(manifest)
    assert (directory / "pop_1.json").stat().st_ino != (directory / "pop_2.json").stat().st_ino
    (directory / "pop_2.json").write_text("[]")
    assert json.loads((directory / "pop_1.json").read_text()) == [1, 2]


def test_changed_archive_is_not_overwritten(tmp_path):
    directory = archive(tmp_path)
    pending = plan([directory])
    (directory / "pop_2.json").write_text("[9]")
    manifest = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="changed"):
        apply(pending, manifest)
    assert json.loads((directory / "pop_2.json").read_text()) == [9]
    assert manifest.exists()  # Recovery metadata survives a partial operation.
    with pytest.raises(ValueError, match="changed"):
        restore(manifest)


def test_existing_recovery_manifest_and_symlinks_are_preserved(tmp_path):
    directory = archive(tmp_path)
    (directory / "link.json").symlink_to(directory / "pop_1.json")
    (directory / "notes.txt").write_text("[1, 2]")
    pending = plan([directory])
    assert pending["n_duplicates"] == 1
    manifest = tmp_path / "manifest.json"
    manifest.write_text("original recovery record")
    with pytest.raises(FileExistsError):
        apply(pending, manifest)
    assert manifest.read_text() == "original recovery record"
    assert (directory / "pop_1.json").stat().st_ino != (directory / "pop_2.json").stat().st_ino
