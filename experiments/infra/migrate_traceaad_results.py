"""Convert historical TraceAAD search runs to one journal per run.

Run from the repository root with ``python3 -m experiments.infra.migrate_traceaad_results``.
The source files are removed only after the generated journal has been reread
and checked against every parsed source record. Held-out evaluations remain
separate results and are not modified here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


RESULTS_ROOT = Path("experiments_result")
SKIP = {"run_config.json", "logs/run_summary.json", "logs/summary.json", "search.jsonl"}
KINDS = {
    "llm_calls": "call",
    "tokenizer_calls": "tokenizer_call",
    "candidates": "candidate",
    "nodes": "node",
    "events": "event",
    "evaluations": "evaluation",
    "decisions": "decision",
    "best_history": "best",
    "best_program": "program",
    "tree_state": "state",
    "latest": "state",
}


def source_records(run_dir: Path, path: Path):
    relative = path.relative_to(run_dir).as_posix()
    kind = KINDS.get(path.stem, path.stem)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield {"kind": kind, "source": relative, "line": line_number,
                           "data": json.loads(line)}
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 1):
                yield {"kind": kind, "source": relative, "row": row_number, "data": row}
    elif path.suffix == ".json":
        yield {"kind": kind, "source": relative,
               "data": json.loads(path.read_text(encoding="utf-8"))}
    elif path.suffix == ".py":
        yield {"kind": kind, "source": relative,
               "data": path.read_text(encoding="utf-8")}
    else:
        raise ValueError(f"unhandled source file: {path}")


def encoded(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=False, allow_nan=True,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(records) -> tuple[int, str]:
    hasher = hashlib.sha256()
    count = 0
    for record in records:
        hasher.update(encoded(record))
        hasher.update(b"\n")
        count += 1
    return count, hasher.hexdigest()


def migrate_run(run_dir: Path) -> tuple[int, int]:
    journal = run_dir / "search.jsonl"
    if journal.exists():
        return 0, 0
    sources = sorted(p for p in run_dir.rglob("*") if p.is_file()
                     and p.relative_to(run_dir).as_posix() not in SKIP)
    if not sources:
        raise ValueError(f"no search records under {run_dir}")
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in sources}
    temporary = journal.with_suffix(".jsonl.tmp")
    expected = hashlib.sha256()
    count = 0
    try:
        with temporary.open("wb") as output:
            for path in sources:
                for record in source_records(run_dir, path):
                    line = encoded(record) + b"\n"
                    output.write(line)
                    expected.update(line)
                    count += 1
            output.flush()
            os.fsync(output.fileno())
        with temporary.open("rb") as generated:
            actual_count, actual_hash = digest(json.loads(line) for line in generated)
        if (actual_count, actual_hash) != (count, expected.hexdigest()):
            raise RuntimeError(f"journal verification failed: {run_dir}")
        for path, metadata in before.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != metadata:
                raise RuntimeError(f"source changed during migration: {path}")
        os.replace(temporary, journal)
        # A journal is a complete copy at this point. Move the old summary only
        # after the journal is durable so an interrupted run keeps its source.
        old_summary = run_dir / "logs" / "summary.json"
        new_summary = run_dir / "logs" / "run_summary.json"
        if old_summary.exists():
            os.replace(old_summary, new_summary)
        elif not new_summary.exists():
            new_summary.parent.mkdir(parents=True, exist_ok=True)
            new_summary.write_text(json.dumps({"status": "unknown",
                "note": "The historical run has no final summary."}) + "\n", encoding="utf-8")
        for path in sources:
            path.unlink()
        for directory in sorted((p for p in run_dir.rglob("*") if p.is_dir()),
                                key=lambda p: len(p.parts), reverse=True):
            if directory != run_dir and not any(directory.iterdir()):
                directory.rmdir()
    finally:
        temporary.unlink(missing_ok=True)
    return len(sources), count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--version", action="append", help="process only this version directory")
    args = parser.parse_args()
    versions = (sorted(args.root.glob("traceaad_v[0-9]*")) if not args.version
                else [args.root / name for name in args.version])
    total = 0
    for version in versions:
        if version.name in {"traceaad_v10_13", "traceaad_v10_prompt_probe"}:
            continue
        runs = sorted(path.parent for path in version.rglob("run_config.json"))
        converted = 0
        for run_dir in runs:
            files, records = migrate_run(run_dir)
            converted += bool(files)
            total += bool(files)
            if files:
                print(f"{run_dir}: {files} files -> {records} records", flush=True)
        print(f"{version.name}: {converted}/{len(runs)} runs converted", flush=True)
    print(f"Total converted: {total}")


if __name__ == "__main__":
    main()
