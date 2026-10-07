"""Prepare independent generated primary data and portable standard supplementary data."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import platform

import numpy as np
import scipy
import numba
from scipy.optimize import Bounds, LinearConstraint, milp

from .dataset import DATA_ROOT, PROBLEMS, PROTOCOL, TASKS
from .generated import COUNTS, RECIPES, SEED, instances as generated_instances


def reference_table(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "optimal_scores" for t in node.targets):
            return ast.literal_eval(node.value)
    return {}


def digest_arrays(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.asarray(value)
        h.update(key.encode())
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def feasible_start(instance):
    a = np.vstack((instance["A_leq"], instance["A_geq"]))
    lower = np.r_[np.full(len(instance["b_leq"]), -np.inf), instance["b_geq"]]
    upper = np.r_[instance["b_leq"], np.full(len(instance["b_geq"]), np.inf)]
    result = milp(np.zeros(a.shape[1]), integrality=np.ones(a.shape[1]), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(a, lower, upper), options={"time_limit": 60.0})
    if result.x is None:
        raise ValueError(f"could not prepare standard MDMKP feasible start: {result.message}")
    x = np.rint(result.x).astype(np.int8)
    if np.any(a @ x < lower) or np.any(a @ x > upper):
        raise ValueError("MDMKP feasible start violates the original integer constraints")
    return x


def standard_instances(task, source_root, source_hashes):
    """Same dimensions, different provenance: never merged into primary held-out."""
    folder = Path(source_root) / PROBLEMS[task]
    config_path = folder / "config.py"
    source = config_path.read_text()
    spec = importlib.util.spec_from_file_location("prepare_cobench", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    refs = reference_table(source)
    names = {"fssp_gls": ["tai50_20.txt"], "mdmkp_search": ["mdmkp_ct4.txt"],
             "graph_colouring": [f"gcol{i}.txt" for i in range(21, 31)],
             "set_cover_construct": [f"scp5{i}.txt" for i in range(1, 11)]}[task]
    source_hashes[f"{PROBLEMS[task]}/config.py"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    starts = {}
    for name in names:
        path = folder / name
        source_hashes[f"{PROBLEMS[task]}/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases = module.load_data(str(path))
        for index, case in enumerate(cases):
            meta = {"id": f"standard_{path.stem}_{index}", "group": f"standard_{path.stem}_{index}",
                    "source_kind": "CO-Bench/OR-Library", "source_case": name, "source_index": index,
                    "scale": int(case["n"])}
            if task == "mdmkp_search":
                if case["q"] != 5:
                    continue
                base = index // 6
                meta.update(group=f"standard_{path.stem}_base{base}", cost_type=case["cost_type"],
                            dimensions={"items": 100, "upper_constraints": 10, "lower_constraints": 5})
                if base not in starts:
                    starts[base] = feasible_start(case)
                arrays = {k: np.asarray(case[k], dtype=np.int64) for k in ("A_leq", "b_leq", "A_geq", "b_geq", "cost_vector")}
                arrays["initial_solution"] = starts[base]
                meta["tightness"] = round(float(np.mean(arrays["b_leq"] / arrays["A_leq"].sum(axis=1))), 2)
            elif task == "fssp_gls":
                arrays = {"processing_times": np.asarray(case["matrix"], dtype=float)}
                meta["dimensions"] = {"jobs": 50, "machines": 20}
            elif task == "graph_colouring":
                adjacency = np.zeros((300, 300), dtype=bool)
                for u, v in case["edges"]:
                    adjacency[u-1, v-1] = adjacency[v-1, u-1] = True
                if adjacency.diagonal().any():
                    raise ValueError("standard graph has a self-loop")
                arrays = {"adjacency": adjacency}
                meta.update(dimensions={"vertices": 300}, density=float(adjacency.sum() / (300*299)))
            else:
                coverage = np.zeros((200, 2000), dtype=bool)
                for row, columns in enumerate(case["row_cover"]):
                    coverage[row, np.asarray(columns)-1] = True
                arrays = {"coverage": coverage, "costs": np.asarray(case["costs"], dtype=float)}
                meta.update(dimensions={"elements": 200, "sets": 2000}, density=float(coverage.mean()))
            reference = float(case["upper_bound"] if task == "fssp_gls" else refs[name][index])
            kind = "published Taillard upper bound" if task == "fssp_gls" else "CO-Bench published reference"
            yield arrays, meta, reference, kind


def prepare(source_root, destination=DATA_ROOT):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    old_manifest = destination / "manifest.json"
    old = json.loads(old_manifest.read_text()) if old_manifest.exists() else {}
    report = {"protocol": PROTOCOL, "seed": SEED, "counts": COUNTS,
              "primary_source": "TraceAAD explicit generated distributions",
              "standard_source_url": "https://huggingface.co/datasets/CO-Bench/CO-Bench",
              "split_policy": "Independent seed streams per task, phase and base group. MDMKP profit variants stay together. Standard inputs are a separate supplementary split and never used in search/selection.",
              "distributions": RECIPES, "tasks": {}, "source_files": {},
              "preparation_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                          "scipy": scipy.__version__, "numba": numba.__version__}}
    for path in (Path(__file__), *(Path(__file__).with_name(name) for name in
                                  ("generated.py", "fssp.py", "graph.py", "set_cover.py", "templates.py"))):
        report["source_files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ahd_prepare_", dir=destination.parent) as temp:
        staged = Path(temp)
        for task in TASKS:
            collected = []
            for split in ("train", "val", "test", "standard"):
                print(f"preparing {task}/{split}", flush=True)
                cases = (standard_instances(task, source_root, report["source_files"]) if split == "standard"
                         else generated_instances(task, split))
                for arrays, meta, reference, kind in cases:
                    if not np.isfinite(reference) or reference <= 0:
                        raise ValueError(f"non-positive reference: {task}/{meta['id']}")
                    relative = Path(task) / (meta["id"] + ".npz")
                    target = staged / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target, **arrays)
                    collected.append({**meta, "split": split, "reference": reference, "reference_kind": kind,
                                      "content": digest_arrays(arrays), "file": str(relative),
                                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            for field in ("group", "content"):
                used = {}
                for row in collected:
                    if used.setdefault(row[field], row["split"]) != row["split"]:
                        raise ValueError(f"split leakage: {task}/{row['id']}")
            report["tasks"][task] = collected
            print(task, {split: sum(r["split"] == split for r in collected)
                         for split in ("train", "val", "test", "standard")}, flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        for rows in report["tasks"].values():
            for row in rows:
                target = destination / row["file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                (staged / row["file"]).replace(target)
        staged_manifest = staged / "manifest.json"
        staged_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        staged_manifest.replace(old_manifest)
        retained = {r["file"] for rows in report["tasks"].values() for r in rows}
        for rows in old.get("tasks", {}).values():
            for row in rows:
                if row["file"] not in retained:
                    (destination / row["file"]).unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/fang/code/LLM4AD/data/CO-Bench"))
    parser.add_argument("--output", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
