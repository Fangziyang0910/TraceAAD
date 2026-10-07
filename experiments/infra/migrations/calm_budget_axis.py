"""Place recorded CALM candidates at the evaluation that produced them.

CALM spends its 1000 evaluations on every candidate but recorded only the
programs it accepted, at budget positions 1, 2, ... in record order. Its
original ``logs/calm/output.log`` (in the storage archive) logs each accepted
program with ``Evals: n/1000``, the evaluation count at the end of its batch.
This rewrites each candidate's ``budget_used`` to that count (the seed to 0)
and the summary's ``budget_used`` to the final count, after checking that the
logged scores match the recorded ones in order.

    uv run python -m experiments.infra.migrations.calm_budget_axis [--apply]
"""

import argparse
import json
import math
import os
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "experiments_result"
ARCHIVE = ROOT / ".archive" / "storage_20261006" / "calm"
ACCEPTED = re.compile(r"\| Perf: (-?[0-9.]+) \| New Best: .*\| Evals: (\d+)/(\d+)")


def logged_acceptances(archive: Path):
    with tarfile.open(archive) as tar:
        log = tar.extractfile("logs/calm/output.log").read().decode()
    rows = [ACCEPTED.search(line) for line in log.splitlines()
            if "Evals:" in line and "Result:" not in line and "Rejected" not in line]
    final = max(int(m.group(1)) for m in re.finditer(r"Evals: (\d+)/", log))
    return [(abs(float(m.group(1))), int(m.group(2))) for m in rows if m], final


def rewrite(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)  # new inode: never edits a hard-linked archive copy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    report = {}
    for archive in sorted(ARCHIVE.glob("*/*.tar.gz")):
        run = ROOT / "calm" / archive.parent.name / archive.name.removesuffix(".tar.gz")
        accepted, final = logged_acceptances(archive)
        lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        candidates = [row for row in rows if row.get("kind") == "candidate"]
        if len(candidates) != len(accepted) + 1:
            raise ValueError(f"{run}: {len(candidates)} records, {len(accepted)} logged acceptances")
        positions = [0] + [evals for _, evals in accepted]
        for row, (score, _) in zip(candidates[1:], accepted):
            if not math.isclose(abs(row["fitness"]), score, rel_tol=0, abs_tol=5e-6):
                raise ValueError(f"{run}: record {row['candidate_id']} {row['fitness']} != logged {score}")
        for row in rows:
            if row.get("kind") == "progress" and "progress" in row:
                row["progress"]["attempts"] = final
        for row, position in zip(candidates, positions):
            row["budget_used"] = position
            if "progress" in row:
                row["progress"]["attempts"] = position
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        summary["budget_used"] = final
        report[run.name] = {"records": len(candidates), "last_record": positions[-1], "final": final}
        if args.apply:
            rewrite(run / "events.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            rewrite(run / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
            # resume.json bounds the visible journal by its committed size.
            resume = json.loads((run / "resume.json").read_text(encoding="utf-8"))
            resume["files"]["events.jsonl"] = (run / "events.jsonl").stat().st_size
            rewrite(run / "resume.json", json.dumps(resume, ensure_ascii=False))
    print(json.dumps({"runs": report, "applied": args.apply}, indent=2))


if __name__ == "__main__":
    main()
