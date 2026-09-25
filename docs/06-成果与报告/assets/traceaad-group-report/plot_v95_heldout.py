#!/usr/bin/env python3
"""Plot TraceAAD V9.5 held-out results against the recorded V9 reference.

V9.5 values are read from the formal held-out ``results.json`` files. V9 values
are parsed from the authoritative task-level result pages. Error bars are sample
standard deviations across independent search repeats, not confidence intervals.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent

V95_RESULTS = {
    "TSP": REPO_ROOT
    / "experiments/tsp_construct/traceaad_v9_5/eval_best_20260812_v95/results.json",
    "CVRP": REPO_ROOT
    / "experiments/cvrp_aco/traceaad_v9_5/eval_best_20260812_v95/results.json",
    "OP": REPO_ROOT
    / "experiments/op_aco/traceaad_v9_5/eval_best_20260812_v95/results.json",
    "OBP": REPO_ROOT
    / "experiments/online_bin_packing/traceaad_v9_5/eval_best_20260812_v95/results.json",
}

V9_RESULT_PAGES = {
    "TSP": REPO_ROOT / "docs/01-主线版本/tsp_construct/主实验汇总.md",
    "CVRP": REPO_ROOT / "docs/01-主线版本/cvrp_aco/主实验汇总.md",
    "OP": REPO_ROOT / "docs/01-主线版本/op_aco/主实验汇总.md",
    "OBP": REPO_ROOT / "docs/01-主线版本/online_bin_packing/主实验汇总.md",
}

TASK_SCALES = {
    "TSP": ["50", "100", "200"],
    "CVRP": ["50", "100", "200"],
    "OP": ["50", "100", "200"],
    "OBP": ["1k_100", "5k_100", "10k_100", "1k_500", "5k_500", "10k_500"],
}


def parse_mean_sd(cell: str) -> tuple[float, float]:
    clean = cell.replace("**", "").strip()
    match = re.fullmatch(r"([-+0-9.]+)\s*±\s*([-+0-9.]+)", clean)
    if match is None:
        raise ValueError(f"Cannot parse mean ± SD cell: {cell!r}")
    return float(match.group(1)), float(match.group(2))


def load_v9(task: str) -> tuple[np.ndarray, np.ndarray]:
    text = V9_RESULT_PAGES[task].read_text(encoding="utf-8")
    row = next(
        line for line in text.splitlines() if line.startswith("| TraceAAD V9 |")
    )
    cells = [part.strip() for part in row.strip().strip("|").split("|")][1:]
    parsed = [parse_mean_sd(cell) for cell in cells]
    return np.array([x[0] for x in parsed]), np.array([x[1] for x in parsed])


def load_v95(task: str) -> tuple[np.ndarray, np.ndarray, int]:
    payload = json.loads(V95_RESULTS[task].read_text(encoding="utf-8"))
    if task == "TSP":
        source = payload["eval_results_by_size"]
        keys = [f"tsp{scale}" for scale in TASK_SCALES[task]]
    elif task in {"CVRP", "OP"}:
        source = payload["results_by_split"]
        keys = [f"test_{scale}" for scale in TASK_SCALES[task]]
    else:
        source = payload["eval_results_by_scale"]
        keys = TASK_SCALES[task]

    summaries = [source[key]["summary"] for key in keys]
    means = np.array([x["mean_eval_objective"] for x in summaries])
    sds = np.array([x["sample_std_eval_objective"] for x in summaries])
    ns = {int(x["num_runs"]) for x in summaries}
    if len(ns) != 1:
        raise ValueError(f"Inconsistent repeat counts for {task}: {sorted(ns)}")
    return means, sds, ns.pop()


def draw_panel(
    ax: plt.Axes,
    *,
    title: str,
    labels: list[str],
    v9: tuple[np.ndarray, np.ndarray],
    v95: tuple[np.ndarray, np.ndarray, int],
    ylabel: str,
) -> None:
    x = np.arange(len(labels), dtype=float)
    v9_mean, v9_sd = v9
    v95_mean, v95_sd, _v95_n = v95
    cap = 3.0

    ax.errorbar(
        x - 0.06,
        v9_mean,
        yerr=v9_sd,
        color="#0072B2",
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        capsize=cap,
        label="V9 (n=3)",
        zorder=3,
    )
    ax.errorbar(
        x + 0.06,
        v95_mean,
        yerr=v95_sd,
        color="#D55E00",
        marker="s",
        markersize=4.5,
        linewidth=1.8,
        capsize=cap,
        label="V9.5",
        zorder=4,
    )
    ax.set_title(title, loc="left")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.margins(x=0.12)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    v9 = {task: load_v9(task) for task in V95_RESULTS}
    v95 = {task: load_v95(task) for task in V95_RESULTS}

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0))
    draw_panel(
        axes[0, 0],
        title="(a) TSP ↓",
        labels=["50", "100", "200"],
        v9=v9["TSP"],
        v95=v95["TSP"],
        ylabel="平均路径长度",
    )
    draw_panel(
        axes[0, 1],
        title="(b) CVRP ↓",
        labels=["50", "100", "200"],
        v9=v9["CVRP"],
        v95=v95["CVRP"],
        ylabel="平均最优路径长度",
    )
    draw_panel(
        axes[0, 2],
        title="(c) OP ↑",
        labels=["50", "100", "200"],
        v9=v9["OP"],
        v95=v95["OP"],
        ylabel="平均收集奖励",
    )

    obp_v9_mean, obp_v9_sd = v9["OBP"]
    obp_v95_mean, obp_v95_sd, obp_n = v95["OBP"]
    draw_panel(
        axes[1, 0],
        title="(d) OBP, capacity=100 ↓",
        labels=["1k", "5k", "10k"],
        v9=(obp_v9_mean[:3], obp_v9_sd[:3]),
        v95=(obp_v95_mean[:3], obp_v95_sd[:3], obp_n),
        ylabel="平均箱数",
    )
    draw_panel(
        axes[1, 1],
        title="(e) OBP, capacity=500 ↓",
        labels=["1k", "5k", "10k"],
        v9=(obp_v9_mean[3:], obp_v9_sd[3:]),
        v95=(obp_v95_mean[3:], obp_v95_sd[3:], obp_n),
        ylabel="平均箱数",
    )

    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="center", fontsize=10, handlelength=2.8)
    axes[1, 2].text(
        0.5,
        0.32,
        "误差棒：独立搜索重复间样本标准差\n"
        "V9 均为 3 次；V9.5 的 TSP/CVRP 为 2 次，OP/OBP 为 3 次",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
        transform=axes[1, 2].transAxes,
        linespacing=1.5,
    )

    fig.suptitle("TraceAAD V9.5 阶段性 held-out 结果：与强版本 V9 对照", fontsize=13, weight="bold")
    fig.supxlabel("测试规模", y=0.025, fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.1, wspace=0.34, hspace=0.38)

    fig.savefig(OUTPUT_DIR / "traceaad-v95-heldout.pdf")
    fig.savefig(OUTPUT_DIR / "traceaad-v95-heldout.png")


if __name__ == "__main__":
    main()
