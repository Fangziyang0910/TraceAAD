"""Load the explicitly prepared, content-identified development and test data."""

import hashlib
import json
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).with_name("data")
PROTOCOL = "ahd-six-tasks-v2"
TASKS = ("fssp_gls", "mdmkp_search", "graph_colouring", "set_cover_construct")
SCALES = {"fssp_gls": (50,), "mdmkp_search": (100,),
          "graph_colouring": (300,), "set_cover_construct": (2000,)}
PROBLEMS = {"fssp_gls": "Flow shop scheduling",
            "mdmkp_search": "Multi-Demand Multidimensional Knapsack problem",
            "graph_colouring": "Graph colouring", "set_cover_construct": "Set covering"}


def manifest(root=DATA_ROOT):
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing AHD data: {path}. Run python -m benchmarks.ahd_suite.prepare_data.")
    result = json.loads(path.read_text())
    if result.get("protocol") != PROTOCOL:
        raise ValueError("AHD data use an obsolete provisional protocol; run benchmarks.ahd_suite.prepare_data again")
    return result


def records(task, split, root=DATA_ROOT):
    if task not in TASKS:
        raise ValueError(f"unknown AHD task: {task}")
    if split not in {"train", "val", "test", "standard", "test_standard"} and not split.startswith("test_"):
        raise ValueError(f"unknown AHD split: {split}")
    phase = "standard" if split == "test_standard" else "test" if split.startswith("test_") else split
    scale = int(split[5:]) if split.startswith("test_") and split != "test_standard" else None
    result = [r for r in manifest(root)["tasks"][task] if r["split"] == phase
              and (scale is None or r["scale"] == scale)]
    if not result:
        raise ValueError(f"empty AHD split: {task}/{split}")
    return result


def load(record, root=DATA_ROOT):
    path = Path(root) / record["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"AHD instance changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def describe(task, split, root=DATA_ROOT):
    rows = records(task, split, root)
    dimensions = {tuple(sorted(r["dimensions"].items())) for r in rows}
    if len(dimensions) != 1:
        raise ValueError(f"mixed dimensions in the fixed-scale task {task}/{split}")
    dim = dict(dimensions.pop())
    size = {"fssp_gls": lambda: f"{dim['jobs']} jobs and {dim['machines']} machines",
            "mdmkp_search": lambda: f"{dim['items']} items, {dim['upper_constraints']} upper and {dim['lower_constraints']} lower constraints",
            "graph_colouring": lambda: f"{dim['vertices']} vertices (edge density about 0.5)",
            "set_cover_construct": lambda: f"{dim['elements']} elements and {dim['sets']} sets (coverage density about 0.02)"}[task]()
    detail = ""
    if task == "mdmkp_search":
        detail = "; positive/mixed profits and constraint-tightness fractions 0.25/0.50/0.75 are balanced by base group"
    provenance = "standard supplementary" if split in {"standard", "test_standard"} else "generated"
    return f"{len(rows)} fixed {provenance} {split} instances with {size}{detail}"
