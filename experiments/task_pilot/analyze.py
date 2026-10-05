"""Summarize the task pilot: how far short searches get, how they vary, and the CO-Bench references."""

import json
from pathlib import Path
import statistics

from experiments.infra.base import RESULTS_ROOT

# CO-Bench (Sun et al., 2025), test-set normalized scores: tuned classical solver, and the best of
# seven LLM agent frameworks (FunSearch, ReEvo, EoH, AIDE, MCTS, refine, best-of-N) on o3-mini, 10 s per instance.
REFERENCE = {
    "cob_period_vrp": (0.124, 0.536), "cob_crew": (0.455, 0.669), "cob_container": (0.097, 0.816),
    "cob_jssp": (0.820, 0.819), "cob_steiner": (0.978, 0.697), "cob_set_cover": (0.888, 0.936),
    "cob_graph_colour": (0.868, 0.924), "cob_cwl": (0.698, 0.752), "cob_set_partition": (1.000, 0.932),
    "cob_tsp": (0.986, 0.940), "cob_flow_shop": (0.922, 0.940), "cob_mdmkp": (0.896, 0.908),
}


def run_summary(run_dir):
    attempts, nodes = [], []
    for line in (run_dir / "search.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["kind"] == "attempt":
            attempts.append(row["data"])
        elif row["kind"] == "node":
            nodes.append(row["data"])
    roots = [n["score"] for n in nodes if n["action"] in ("Init",) or (n["action"] == "Repair" and n["depth"] <= 1)]
    best = max((n["score"] for n in nodes), default=None)
    curve, running = [], None
    for a in sorted(attempts, key=lambda a: a["id"]):
        if a["status"] == "valid":
            running = a["score"] if running is None else max(running, a["score"])
        curve.append(running)
    out = {"attempts": len(attempts), "valid_rate": sum(a["status"] == "valid" for a in attempts) / max(1, len(attempts)),
           "timeout_or_error": sum(a["status"] in ("timeout", "runtime_error", "invalid_output") for a in attempts),
           "root_best": max(roots) if roots else None, "train_best": best,
           "train_at_50": curve[49] if len(curve) >= 50 else None,
           "frontier_refreshes": sum(1 for i in range(1, len(curve)) if curve[i] is not None and
                                     (curve[i - 1] is None or curve[i] > curve[i - 1] + 1e-12))}
    heldout = run_dir / "heldout_test.json"
    if heldout.exists():
        out["test"] = json.loads(heldout.read_text())["score"]
    return out


def main():
    root = RESULTS_ROOT / "task_pilot"
    report = {}
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("cob_")):
        runs = {r.name: run_summary(r) for r in sorted(task_dir.iterdir()) if (r / "search.jsonl").exists()}
        tests = [r["test"] for r in runs.values() if r.get("test") is not None]
        trains = [r["train_best"] for r in runs.values() if r.get("train_best") is not None]
        classical, best_llm = REFERENCE.get(task_dir.name, (None, None))
        report[task_dir.name] = {"runs": runs, "classical": classical, "cobench_best_llm": best_llm,
                                 "test_mean": statistics.fmean(tests) if tests else None,
                                 "test_spread": max(tests) - min(tests) if len(tests) > 1 else None,
                                 "train_spread": max(trains) - min(trains) if len(trains) > 1 else None}
    (root / "pilot_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for task, r in report.items():
        line = " | ".join(f"{name}: att {v['attempts']} valid {v['valid_rate']:.2f} root {v['root_best'] or 0:.3f} "
                          f"best {v['train_best'] or 0:.3f} test {v.get('test', float('nan')):.3f}"
                          for name, v in r["runs"].items())
        print(f"{task:18s} classical {r['classical']} best-LLM {r['cobench_best_llm']} || {line}")


if __name__ == "__main__":
    main()
