"""Offline behavioral audit of the eight roots in each initialization run.

Profiles use training-only A/B probes and do not affect candidate selection.
Raw trajectories are cached beneath the ignored results directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr

from experiments.infra import behavior_profile as core

from .analyze import read_jsonl, report_job

from .launch import RESULTS, SCHEDULE, run_dir

PROFILE_DIR = RESULTS / "behavior_profiles"
REPORT = RESULTS / "diversity_summary.json"
PANELS = ("A", "B")


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def roots_for(job: dict) -> list[dict]:
    roots = [node for node in read_jsonl(run_dir(job) / "nodes.jsonl")
             if node["operator"] == "Init" and node["parent_id"] is None]
    if len(roots) != 8:
        raise ValueError(f"expected eight roots in {job['run_name']}, got {len(roots)}")
    return roots


def profile_path(job: dict, node: dict) -> Path:
    return PROFILE_DIR / job["run_name"] / f"{node['id']}.json"


def profile_one(task: str, node: dict) -> dict:
    candidate = {"key": str(node["id"]), "id": node["id"], "code": node["code"]}
    panels = {}
    for panel in PANELS:
        core._init_worker(task, panel, core.DEFAULT_TRAJECTORY_POINTS[task],
                          core.DEFAULT_TIMEOUT_SECONDS[task])
        panels[panel] = core._profile_candidate(candidate)
    return {"code_sha256": hashlib.sha256(node["code"].encode()).hexdigest(),
            "panels": panels}


def load_profiles(jobs: list[dict], *, workers: int, refresh: bool) -> dict:
    profiles = {}
    pending = {}
    for job in jobs:
        for node in roots_for(job):
            key = (job["run_name"], node["id"])
            path = profile_path(job, node)
            code_hash = hashlib.sha256(node["code"].encode()).hexdigest()
            if path.exists() and not refresh:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("code_sha256") == code_hash and set(cached["panels"]) == set(PANELS):
                    profiles[key] = cached
                    continue
            pending[key] = (job, node)

    print(f"behavior profiles: cached={len(profiles)} pending={len(pending)}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(profile_one, job["task"], node): (key, job, node)
                       for key, (job, node) in pending.items()}
            for index, future in enumerate(as_completed(futures), 1):
                key, job, node = futures[future]
                value = future.result()
                dump(profile_path(job, node), value)
                profiles[key] = value
                if index % 40 == 0 or index == len(pending):
                    print(f"behavior profiles: {index}/{len(pending)} new", flush=True)
    return profiles


def pair_mean(matrix: np.ndarray, indices: list[int]) -> float | None:
    if len(indices) < 2:
        return None
    sub = matrix[np.ix_(indices, indices)]
    return float(np.mean(sub[np.triu_indices(len(indices), 1)]))


def summarize_run(job: dict, profiles: dict) -> dict:
    roots = roots_for(job)
    nodes = read_jsonl(run_dir(job) / "nodes.jsonl")
    events = read_jsonl(run_dir(job) / "events.jsonl")
    report = report_job(job)
    valid = []
    failures = []
    for node in roots:
        result = profiles[(job["run_name"], node["id"])]
        if all(result["panels"][panel]["ok"] for panel in PANELS):
            valid.append(node)
        else:
            failures.append({"node_id": node["id"], "panels": {
                panel: {"error_type": result["panels"][panel].get("error_type"),
                        "error": result["panels"][panel].get("error")}
                for panel in PANELS if not result["panels"][panel]["ok"]}})

    matrices = {}
    for panel in PANELS:
        panel_profiles = [profiles[(job["run_name"], node["id"])]["panels"][panel]
                          for node in valid]
        matrices[panel] = core.compute_distance_matrix(
            panel_profiles, prefix_mode=job["task"] in core.PREFIX_TASKS)
    matrix = (matrices["A"] + matrices["B"]) / 2
    indices = list(range(len(valid)))
    ranked = sorted(roots, key=lambda node: node["fitness"], reverse=True)
    cutoff = ranked[3]["fitness"]
    top_ids = {node["id"] for node in roots if node["fitness"] >= cutoff}
    top_indices = [index for index, node in enumerate(valid) if node["id"] in top_ids]
    upper = np.triu_indices(len(valid), 1)
    pair_values = matrix[upper]
    panel_corr = None
    if (len(pair_values) > 2 and np.ptp(matrices["A"][upper]) > 0
            and np.ptp(matrices["B"][upper]) > 0):
        rho = spearmanr(matrices["A"][upper], matrices["B"][upper]).statistic
        if np.isfinite(rho):
            panel_corr = float(rho)

    root_ids = {node["id"] for node in roots}
    used_as_parent = Counter(event["parent_id"] for event in events
                             if event["operator"] != "Init" and event["parent_id"] in root_ids)
    used_as_donor = Counter(event["reference_id"] for event in events
                            if event["operator"] != "Init" and event["reference_id"] in root_ids)
    by_id = {node["id"]: node for node in nodes}
    best_id = max(nodes, key=lambda node: node["fitness"])["id"]
    ancestor = by_id[best_id]
    while ancestor["parent_id"] is not None:
        ancestor = by_id[ancestor["parent_id"]]
    init_context = read_jsonl(run_dir(job) / "init_context.jsonl")
    prompt_tokens = sum(context["prompt_tokens"] for context in init_context
                        if context["candidate_id"] <= report["root_completion_attempt"])

    return {"task": job["task"], "mode": job["mode"], "repeat": job["repeat"],
            "run_name": job["run_name"], "profile_valid": len(valid),
            "profile_failures": failures, "root_best_train": max(n["fitness"] for n in roots),
            "root_mean_train": mean(n["fitness"] for n in roots),
            "mean_pair_distance": pair_mean(matrix, indices),
            "high_quality_root_count": len(top_ids),
            "high_quality_mean_pair_distance": pair_mean(matrix, top_indices)
            if len(top_indices) == len(top_ids) else None,
            "panel_a_mean_pair_distance": pair_mean(matrices["A"], indices),
            "panel_b_mean_pair_distance": pair_mean(matrices["B"], indices),
            "zero_pairs_both_panels": int(np.sum((matrices["A"][upper] == 0)
                                             & (matrices["B"][upper] == 0))),
            "pair_count": len(pair_values), "panel_distance_spearman": panel_corr,
            "unique_root_codes": len({node["code"] for node in roots}),
            "root_parent_attempts": sum(used_as_parent.values()),
            "roots_used_as_parent": len(used_as_parent),
            "root_donor_attempts": sum(used_as_donor.values()),
            "roots_used_as_donor": len(used_as_donor),
            "best_lineage_root_id": ancestor["id"],
            "best_lineage_root_train": ancestor["fitness"],
            "best_lineage_root_rank": 1 + sum(root["fitness"] > ancestor["fitness"]
                                               for root in roots),
            "best_is_root": best_id in root_ids,
            "root_completion_evaluation": report["root_completion_evaluation"],
            "init_prompt_tokens": prompt_tokens,
            "development_gain_train": report["best_train_fitness"] - report["root_best_train_fitness"],
            "heldout_fitness": report["heldout_fitness"]}


def summarize(jobs: list[dict], profiles: dict) -> dict:
    rows = [summarize_run(job, profiles) for job in jobs]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["mode"])].append(row)
    group_summary = []
    for (task, mode), group in sorted(groups.items()):
        fields = ("mean_pair_distance", "high_quality_mean_pair_distance",
                  "high_quality_root_count", "panel_a_mean_pair_distance",
                  "panel_b_mean_pair_distance", "root_best_train",
                  "root_mean_train", "root_completion_evaluation", "development_gain_train",
                  "heldout_fitness", "root_parent_attempts", "roots_used_as_parent",
                  "root_donor_attempts", "init_prompt_tokens", "best_lineage_root_rank")
        group_summary.append({"task": task, "mode": mode, "runs": len(group),
                              "valid_profiles": sum(r["profile_valid"] for r in group),
                              "unique_root_codes": sum(r["unique_root_codes"] for r in group),
                              "zero_pairs_both_panels": sum(r["zero_pairs_both_panels"] for r in group),
                              "pair_count": sum(r["pair_count"] for r in group),
                              "best_is_root_runs": sum(r["best_is_root"] for r in group),
                              **{field: mean(r[field] for r in group if r[field] is not None)
                                 for field in fields if any(r[field] is not None for r in group)}})
    return {"protocol": "training-only profile_core v3; A/B; four instances per panel; "
                         "OBP compact probe; ACO default iterations and one seed per instance",
            "quality_subset": "best four roots plus all roots tied at the fourth-best training score",
            "runs": rows, "groups": group_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    jobs = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    profiles = load_profiles(jobs, workers=args.workers, refresh=args.refresh)
    core.numba.set_num_threads(min(4, core.numba.get_num_threads()))
    result = summarize(jobs, profiles)
    dump(REPORT, result)
    for group in result["groups"]:
        print(f"{group['task']} {group['mode']}: "
              f"profiles={group['valid_profiles']}/32 "
              f"distance={group['mean_pair_distance']:.3f} "
              f"high_quality={group['high_quality_mean_pair_distance']:.3f} "
              f"zero_pairs={group['zero_pairs_both_panels']}/112 "
              f"heldout={group['heldout_fitness']:.3f}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
