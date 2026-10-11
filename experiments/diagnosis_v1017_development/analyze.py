"""What V10.17's short development did to the new designs it was given.

Developed explorations are drawn by a fixed per-exploration draw, independent of the first version's quality.
For each: the gap of the first and of the best developed version to the search best at the close,
and whether the best developed version is back at the starting program (code similarity >= 0.9).
"""

import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from traceaad.common.canonical import similarity  # noqa: E402

RUNS = ROOT / "experiments_result" / "traceaad_v10_17"
OUT = ROOT / "experiments_result" / "diagnosis_v1017_development" / "summary.json"
BACK = 0.9
TASKS = ("tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing", "vrptw_construct")


def gap(score, best):
    return (score - best) / abs(best) * 100


def run_explorations(run):
    code = {}
    programs = {json.loads(l)["key"]: json.loads(l)["code"] for l in open(run / "programs.jsonl")}
    events = [json.loads(l) for l in open(run / "events.jsonl")]
    for e in events:
        if e.get("kind") == "candidate" and e.get("program"):
            code[e["program"]["id"]] = programs[e["program"]["key"]]
    for e in events:
        if e.get("kind") == "exploration" and e["exploration"].get("developed") and e["exploration"]["first_score"] is not None:
            yield e["exploration"], code


def main():
    summary = {}
    for task in TASKS:
        first, best, rows = [], [], dict(developed=0, improved=0, back=0, kept=0, first_within_1=0, within_1=0, back_within_1=0, kept_within_1=0)
        for run in sorted(glob.glob(str(RUNS / task / "*rep*"))):
            for x, code in run_explorations(Path(run)):
                a, b = gap(x["first_score"], x["search_best_score"]), gap(x["best_score"], x["search_best_score"])
                first.append(a), best.append(b)
                rows["developed"] += 1
                rows["first_within_1"] += a <= 1
                rows["within_1"] += b <= 1
                if x["best_score"] < x["first_score"]:
                    rows["improved"] += 1
                    back = similarity(code[x["best_id"]], code[x["start_id"]]) >= BACK
                    rows["back" if back else "kept"] += 1
                    rows["back_within_1" if back else "kept_within_1"] += b <= 1
        summary[task] = dict(rows, median_first_gap=statistics.median(first), median_best_gap=statistics.median(best))
    total = {k: sum(v[k] for v in summary.values()) for k in summary[TASKS[0]] if not k.startswith("median")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"by_task": summary, "total": total}, indent=1, ensure_ascii=False))
    print(json.dumps(total))


if __name__ == "__main__":
    main()
