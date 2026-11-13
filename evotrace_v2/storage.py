"""Run directory: JSON files and append-only JSON lines."""

import json
import os
from pathlib import Path


def write_json(path: Path, value):
    """Write through a temporary file, so a killed process never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class Store:
    def __init__(self, directory):
        self.path = Path(directory) if directory is not None else None
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)

    def write(self, relative: str, data):
        if self.path is not None:
            write_json(self.path / relative, data)

    def read(self, relative: str, default=None):
        if self.path is None or not (self.path / relative).exists():
            return default
        return json.loads((self.path / relative).read_text())

    def append(self, relative: str, row: dict):
        if self.path is not None:
            with (self.path / relative).open("a") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rows(self, relative: str) -> list[dict]:
        """Read complete lines. Only a final line without a newline can be unfinished."""
        if self.path is None or not (self.path / relative).exists():
            return []
        return [json.loads(line) for line in (self.path / relative).read_bytes().split(b"\n")[:-1]]

    def repair_tail(self, relative: str):
        """Remove a line cut off by a killed process, so the next row starts on its own line."""
        if self.path is None or not (self.path / relative).exists():
            return
        path = self.path / relative
        end = path.read_bytes().rfind(b"\n") + 1
        with path.open("r+b") as output:
            output.truncate(end)
