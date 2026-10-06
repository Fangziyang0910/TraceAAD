"""Scored programs and frozen winners from canonical result files."""

from pathlib import Path
from traceaad.common.storage import Programs, read_json, rows


def load_run_summary(run_dir, *, require_finished=True):
    summary = read_json(Path(run_dir) / "summary.json")
    if summary is None:
        raise RuntimeError(f"missing summary: {run_dir}")
    if require_finished and summary["status"] != "finished":
        raise RuntimeError(f"run is not a completed search: {run_dir}")
    return summary


def load_scored_samples(run_dir, *, max_sample_order=None):
    sources = Programs(run_dir)
    programs, samples = {}, []
    for row in rows(Path(run_dir) / "events.jsonl"):
        if row.get("program"):
            programs[row["program"]["id"]] = row["program"]
        if row["kind"] != "candidate" or not row["valid"]:
            continue
        order = row["candidate_id"]
        if max_sample_order is not None and order > max_sample_order:
            continue
        program = programs.get(row["node_id"])
        if program is not None:
            code = sources.get(program["key"])
            if code:
                samples.append({"sample_order": order, "score": row["fitness"], "program": code,
                                "operator": row["operator"], "node_id": program["id"], "key": program["key"]})
    return samples


def pick_best_sample(run_dir, *, max_sample_order=None, sample_order=None, allow_incomplete=False):
    summary = load_run_summary(run_dir, require_finished=not allow_incomplete)
    samples = load_scored_samples(run_dir, max_sample_order=max_sample_order)
    if sample_order is not None:
        selected = next((r for r in samples if r["sample_order"] == sample_order), None)
    elif max_sample_order is None and summary["status"] == "finished" and summary.get("best"):
        best = summary["best"]
        selected = next((r for r in samples if r["key"] == best["key"]), None)
        if selected is None:
            selected = {"sample_order": best["id"], "score": best["fitness"],
                        "program": Programs(run_dir).get(best["key"]), "operator": best["action"]}
    else:
        selected = max(samples, key=lambda r: r["score"], default=None)
    if selected is None:
        raise RuntimeError(f"no matching scored program: {run_dir}")
    return selected, samples
