"""Scored programs and frozen winners from canonical result files."""

from pathlib import Path
from traceaad.common.storage import Programs, committed_rows, read_json, selected_program


def load_run_summary(run_dir, *, require_finished=True):
    summary = read_json(Path(run_dir) / "summary.json")
    if summary is None and not require_finished:
        return {"status": "searching"}
    if summary is None:
        raise RuntimeError(f"missing summary: {run_dir}")
    if require_finished and summary["status"] != "finished":
        raise RuntimeError(f"run is not a completed search: {run_dir}")
    return summary


def load_scored_samples(run_dir, *, max_sample_order=None, include_source=True):
    sources = Programs(run_dir)
    programs, samples = {}, []
    for row in committed_rows(run_dir):
        if row.get("program"):
            programs[row["program"]["id"]] = row["program"]
        if row["kind"] != "candidate" or not row["valid"]:
            continue
        order = row["candidate_id"]
        if max_sample_order is not None and order > max_sample_order:
            continue
        program = programs.get(row["node_id"])
        if program is not None:
            sample = {"sample_order": order, "score": row["fitness"],
                      "operator": row["operator"], "node_id": program["id"], "key": program["key"]}
            if include_source:
                sample["program"] = sources.get(program["key"])
            samples.append(sample)
    return samples


def pick_best_sample(run_dir, *, max_sample_order=None, sample_order=None, allow_incomplete=False):
    summary = load_run_summary(run_dir, require_finished=not allow_incomplete)
    samples = load_scored_samples(run_dir, max_sample_order=max_sample_order, include_source=False)
    if sample_order is not None:
        selected = next((r for r in samples if r["sample_order"] == sample_order), None)
    elif max_sample_order is None and summary["status"] == "finished" and summary.get("best"):
        best = selected_program(run_dir)
        if not best["verified"]:
            raise ValueError(f"selected program changed after selection: {run_dir}")
        selected = next((dict(r) for r in samples if r["key"] == best["key"]), None)
        if selected is None:
            selected = {"sample_order": best["id"], "score": best["fitness"],
                        "program": best["code"], "operator": best["action"], "key": best["key"], "node_id": best["id"]}
    else:
        selected = max(samples, key=lambda r: r["score"], default=None)
    if selected is None:
        raise RuntimeError(f"no matching scored program: {run_dir}")
    selected = dict(selected)
    if "program" not in selected:
        selected["program"] = Programs(run_dir).get(selected["key"])
    return selected, samples
