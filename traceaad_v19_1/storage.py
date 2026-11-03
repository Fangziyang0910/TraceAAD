"""Local run records, source snapshot and atomic checkpoints."""

import json
import os
from pathlib import Path
import shutil
import tempfile


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Store:
    def __init__(self, directory: str | Path | None, *, resume=False):
        self.path = Path(directory) if directory is not None else None
        self._lock = None
        if self.path is None:
            if resume:
                raise ValueError("Resume requires a run directory")
            return
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = (self.path / ".lock").open("a+")
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            existing = [p for p in self.path.iterdir() if p.name != ".lock"]
            if resume and not (self.path / "state.json").is_file():
                raise FileNotFoundError("No checkpoint exists in this run directory")
            if not resume and existing:
                raise FileExistsError("Run directory is not empty; use resume=True to continue it")
        except BaseException:
            self.close()
            raise

    def write(self, relative: str, data):
        if self.path is not None:
            write_json(self.path / relative, data)

    def read(self, relative: str):
        if self.path is None:
            raise ValueError("This search has no persistent run directory")
        return json.loads((self.path / relative).read_text())

    def update(self, relative: str, changes: dict):
        if self.path is not None:
            value = self.read(relative)
            value.update(changes)
            self.write(relative, value)

    def snapshot(self):
        if self.path is not None:
            target = self.path / "source"
            target.mkdir()
            for source in Path(__file__).parent.glob("*.py"):
                shutil.copy2(source, target / source.name)

    def program(self, program):
        self.write(f"programs/{program.id:06d}.json", program.to_dict())
        if self.path is not None:
            (self.path / "programs" / f"{program.id:06d}.py").write_text(program.code)

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

