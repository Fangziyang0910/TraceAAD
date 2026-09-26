"""Diagnose whether V9.7 optimism changed route and anchor selections.

The analysis replays each recorded allocation snapshot with the optimism
term removed while preserving the selector's deterministic tie-breaks.  It
therefore answers whether the recorded bonus changed a decision at that
snapshot.  It does not construct the unobserved full search trajectory that
would have followed from a different earlier decision, and it does not
estimate final policy value.

Usage:

    uv run python experiments/traceaad_v9_7/analyze.py \
        --batch 20260813_184519 \
        --json-out experiments/traceaad_v9_7/analysis/allocation_summary.json \
        --markdown-out experiments/traceaad_v9_7/analysis/allocation_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "traceaad-v9.7-route-refine-explore"
TASK_LABELS = {
    "tsp_construct": "TSP",
    "cvrp_aco": "CVRP",
    "op_aco": "OP",
    "online_bin_packing": "OBP",
}
PHASES = ("early", "middle", "late")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load append-only JSONL, tolerating only an incomplete final line."""
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def phase_for(index: int, total: int) -> str:
    if total <= 0 or not 0 <= index < total:
        raise ValueError("phase index must identify an observed decision")
    return PHASES[min(2, 3 * index // total)]


def _route_score_key(route: dict[str, Any]) -> tuple[float, int, int]:
    return (
        float(route["score"]),
        -int(route["n"]),
        -int(route["root_state_id"]),
    )


def _route_quality_key(route: dict[str, Any]) -> tuple[float, int, int]:
    # Root state ids are allocated in root creation order.  The decisions log
    # does not repeat creation_order, so root_state_id preserves both final
    # deterministic tie-breaks used by the live selector.
    return (
        float(route["best_q"]),
        -int(route["n"]),
        -int(route["root_state_id"]),
    )


def _state_score_key(state: dict[str, Any]) -> tuple[float, int, int, int]:
    return (
        float(state["score"]),
        -int(state["n"]),
        -int(state["creation_order"]),
        -int(state["state_id"]),
    )


def _state_quality_key(state: dict[str, Any]) -> tuple[float, int, int, int]:
    return (
        float(state["q"]),
        -int(state["n"]),
        -int(state["creation_order"]),
        -int(state["state_id"]),
    )


def analyze_route_event(event: dict[str, Any]) -> dict[str, Any]:
    routes = list(event["routes"])
    if len(routes) < 2:
        raise ValueError("route allocation diagnostic requires at least two routes")
    selected = max(routes, key=_route_score_key)
    selected_id = int(event["selected_root_state_id"])
    if int(selected["root_state_id"]) != selected_id:
        raise ValueError("recorded route selection does not match the score snapshot")

    quality_winner = max(routes, key=_route_quality_key)
    quality_winner_id = int(quality_winner["root_state_id"])
    competitors: list[dict[str, Any]] = []
    critical_multipliers: list[float] = []
    for route in routes:
        if int(route["root_state_id"]) == quality_winner_id:
            continue
        quality_gap = float(quality_winner["best_q"]) - float(route["best_q"])
        optimism_advantage = float(route["optimism"]) - float(
            quality_winner["optimism"]
        )
        margin = optimism_advantage - quality_gap
        critical = None
        if quality_gap > 0.0 and optimism_advantage > 0.0:
            critical = quality_gap / optimism_advantage
            critical_multipliers.append(critical)
        competitors.append(
            {
                "root_state_id": int(route["root_state_id"]),
                "quality_gap": quality_gap,
                "optimism_advantage": optimism_advantage,
                "margin": margin,
                "critical_multiplier": critical,
            }
        )
    strongest = max(
        competitors,
        key=lambda item: (item["margin"], -item["root_state_id"]),
    )
    return {
        "iteration": int(event["iteration"]),
        "selected_id": selected_id,
        "quality_winner_id": quality_winner_id,
        "intervened": selected_id != quality_winner_id,
        "strongest_challenger_id": strongest["root_state_id"],
        "quality_gap": strongest["quality_gap"],
        "optimism_advantage": strongest["optimism_advantage"],
        "margin": strongest["margin"],
        "critical_multiplier": (
            min(critical_multipliers) if critical_multipliers else None
        ),
    }


def analyze_anchor_event(event: dict[str, Any]) -> dict[str, Any]:
    states = list(event["states"])
    if not states:
        raise ValueError("anchor selection snapshot is empty")
    selected = max(states, key=_state_score_key)
    selected_id = int(event["selected_state_id"])
    if int(selected["state_id"]) != selected_id:
        raise ValueError("recorded anchor selection does not match the score snapshot")
    quality_winner = max(states, key=_state_quality_key)
    return {
        "iteration": int(event["iteration"]),
        "selected_id": selected_id,
        "quality_winner_id": int(quality_winner["state_id"]),
        "intervened": selected_id != int(quality_winner["state_id"]),
    }


def summarize_interventions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    changed = sum(bool(row["intervened"]) for row in rows)
    by_phase: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase]
        phase_changed = sum(bool(row["intervened"]) for row in phase_rows)
        by_phase[phase] = {
            "decisions": len(phase_rows),
            "interventions": phase_changed,
            "intervention_rate": (
                phase_changed / len(phase_rows) if phase_rows else None
            ),
        }
    return {
        "decisions": total,
        "interventions": changed,
        "intervention_rate": changed / total if total else None,
        "by_phase": by_phase,
    }


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "q25": quantile(finite, 0.25),
        "median": quantile(finite, 0.5),
        "q75": quantile(finite, 0.75),
        "q90": quantile(finite, 0.9),
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "run_config.json")
    summary = load_json(run_dir / "logs" / "summary.json")
    params = config.get("method_params", {})
    if params.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected protocol in {run_dir}")
    if summary.get("status") != "finished":
        raise ValueError(f"run is not finished: {run_dir}")
    if int(summary.get("evaluator_call_count", -1)) != int(
        params.get("evaluator_call_budget", params.get("budget"))
    ):
        raise ValueError(f"run did not reach its evaluator budget: {run_dir}")

    decisions = load_jsonl(run_dir / "artifacts" / "decisions.jsonl")
    route_events = sorted(
        (row for row in decisions if row.get("event") == "route_selected"),
        key=lambda row: int(row["iteration"]),
    )
    anchor_events = sorted(
        (row for row in decisions if row.get("event") == "anchor_selected"),
        key=lambda row: int(row["iteration"]),
    )
    if len(route_events) != len(anchor_events):
        raise ValueError(f"unpaired route/anchor decisions in {run_dir}")
    if [row["iteration"] for row in route_events] != [
        row["iteration"] for row in anchor_events
    ]:
        raise ValueError(f"route/anchor iteration mismatch in {run_dir}")

    route_rows = [analyze_route_event(row) for row in route_events]
    anchor_rows = [analyze_anchor_event(row) for row in anchor_events]
    for index, row in enumerate(route_rows):
        row["phase"] = phase_for(index, len(route_rows))
    for index, row in enumerate(anchor_rows):
        row["phase"] = phase_for(index, len(anchor_rows))

    route_counts = Counter(row["selected_id"] for row in route_rows)
    decisions_n = len(route_rows)
    shares = {
        str(route_id): count / decisions_n
        for route_id, count in sorted(route_counts.items())
    }
    switches = sum(
        left["selected_id"] != right["selected_id"]
        for left, right in zip(route_rows, route_rows[1:])
    )
    critical = [
        float(row["critical_multiplier"])
        for row in route_rows
        if row["critical_multiplier"] is not None
    ]
    bootstrap_deltas = [float(value) for value in summary["bootstrap_deltas"]]
    # Completed batches record the scale as ``optimism_scale``; newer runs as ``s``.
    scale = summary.get("optimism_scale")
    if scale is None:
        scale = summary["s"]
    task = str(config["task"])
    return {
        "run_name": str(config.get("timestamp", run_dir.name)),
        "run_dir": str(run_dir.relative_to(REPO)),
        "task": task,
        "task_label": TASK_LABELS.get(task, task),
        "repeat": int(config.get("repeat") or 0),
        "history_selector_id": str(
            params.get("history_selector_id") or params.get("protocol_id")
        ),
        "evaluator_calls": int(summary["evaluator_call_count"]),
        "optimism_scale": float(scale),
        "bootstrap_delta_count": len(bootstrap_deltas),
        "bootstrap_zero_delta_count": sum(value == 0.0 for value in bootstrap_deltas),
        "route": {
            **summarize_interventions(route_rows),
            "routes_selected": len(route_counts),
            "route_switches": switches,
            "route_shares": shares,
            "max_route_share": max(shares.values()),
            "hhi": sum(share * share for share in shares.values()),
            "quality_gap": _distribution(row["quality_gap"] for row in route_rows),
            "optimism_advantage": _distribution(
                row["optimism_advantage"] for row in route_rows
            ),
            "margin": _distribution(row["margin"] for row in route_rows),
            "critical_multiplier": {
                **_distribution(critical),
                "fraction_gt_10": (
                    sum(value > 10.0 for value in critical) / len(critical)
                    if critical
                    else None
                ),
                "fraction_gt_50": (
                    sum(value > 50.0 for value in critical) / len(critical)
                    if critical
                    else None
                ),
                "fraction_gt_100": (
                    sum(value > 100.0 for value in critical) / len(critical)
                    if critical
                    else None
                ),
            },
        },
        "anchor": summarize_interventions(anchor_rows),
    }


def _aggregate_interventions(
    runs: Sequence[dict[str, Any]], level: str
) -> dict[str, Any]:
    decisions = sum(int(run[level]["decisions"]) for run in runs)
    interventions = sum(int(run[level]["interventions"]) for run in runs)
    by_phase: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        phase_decisions = sum(
            int(run[level]["by_phase"][phase]["decisions"]) for run in runs
        )
        phase_interventions = sum(
            int(run[level]["by_phase"][phase]["interventions"]) for run in runs
        )
        by_phase[phase] = {
            "decisions": phase_decisions,
            "interventions": phase_interventions,
            "intervention_rate": (
                phase_interventions / phase_decisions if phase_decisions else None
            ),
        }
    return {
        "decisions": decisions,
        "interventions": interventions,
        "intervention_rate": interventions / decisions if decisions else None,
        "by_phase": by_phase,
    }


def aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["task_label"])].append(run)
    tasks: dict[str, Any] = {}
    for task, task_runs in grouped.items():
        tasks[task] = {
            "runs": len(task_runs),
            "route": _aggregate_interventions(task_runs, "route"),
            "anchor": _aggregate_interventions(task_runs, "anchor"),
            "max_route_share_min": min(
                float(run["route"]["max_route_share"]) for run in task_runs
            ),
            "max_route_share_max": max(
                float(run["route"]["max_route_share"]) for run in task_runs
            ),
            "mean_hhi": mean(float(run["route"]["hhi"]) for run in task_runs),
        }
    return {
        "runs": len(runs),
        "tasks": tasks,
        "overall": {
            "route": _aggregate_interventions(runs, "route"),
            "anchor": _aggregate_interventions(runs, "anchor"),
            "runs_with_zero_bootstrap_delta": sum(
                int(run["bootstrap_zero_delta_count"]) > 0 for run in runs
            ),
        },
    }


def discover_runs(batch: str) -> list[Path]:
    results_dir = Path(__file__).resolve().parent / "results"
    configs = sorted(results_dir.glob(f"*/*{batch}*/run_config.json"))
    return [path.parent for path in configs]


def analyze_batch(batch: str) -> dict[str, Any]:
    run_dirs = discover_runs(batch)
    if not run_dirs:
        raise FileNotFoundError(f"no V9.7 runs found for batch {batch}")
    runs = [analyze_run(run_dir) for run_dir in run_dirs]
    runs.sort(key=lambda run: (str(run["task_label"]), int(run["repeat"])))
    return {
        "analysis": "traceaad_v97_allocation_activation",
        "batch": batch,
        "evidence_boundary": (
            "same-snapshot replay measures direct selection activation; it is not "
            "an alternate full-search trajectory or final-quality evidence"
        ),
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def render_markdown(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    overall_route = aggregate["overall"]["route"]
    overall_anchor = aggregate["overall"]["anchor"]
    lines = [
        "# TraceAAD V9.7 路线级分配诊断",
        "",
        f"> 分析批次：`{result['batch']}`；{aggregate['runs']} 个完成 run。",
        ">",
        "> 本分析在每次选择当时记录的 `(q, n, optimism, score)` snapshot 上将 optimism 置零后重放平局规则。它只回答 bonus 当时是否改变选择，不构造另一条完整搜索轨迹，也不是最终质量证据。",
        ">",
        f"> 协议边界：该批次 `run_config` 记录的历史选择器是 `{result['runs'][0]['history_selector_id']}`；当前工作树已另行修改历史组成。本报告只依赖批次中保持不变的路线/锚点分配公式和决策日志，不声称该批次验证了当前 parent-path 历史协议。",
        "",
        "## 核心判断",
        "",
        f"V9.7 的 route score 确实被计算，但 route optimism 在 {overall_route['decisions']:,} 次决策中只改变 {overall_route['interventions']:,} 次（{_pct(overall_route['intervention_rate'])}）。同一尺度在锚点层改变 {overall_anchor['interventions']:,}/{overall_anchor['decisions']:,} 次（{_pct(overall_anchor['intervention_rate'], 1)}）。因此共享 bootstrap 一步尺度在两层产生了明显不同的干预强度：锚点层活跃，路线层接近未激活。",
        "",
        "## 任务汇总",
        "",
        "| Task | Route intervention | Anchor intervention | Top-1 route share |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task in ("TSP", "CVRP", "OP", "OBP"):
        if task not in aggregate["tasks"]:
            continue
        item = aggregate["tasks"][task]
        route = item["route"]
        anchor = item["anchor"]
        lines.append(
            f"| {task} | {route['interventions']}/{route['decisions']} ({_pct(route['intervention_rate'])}) | "
            f"{anchor['interventions']}/{anchor['decisions']} ({_pct(anchor['intervention_rate'], 1)}) | "
            f"{_pct(item['max_route_share_min'], 1)}–{_pct(item['max_route_share_max'], 1)} |"
        )
    lines.extend(
        [
            "",
            "OBP 虽然访问了多个初始来源，但其 route intervention 仍很少；这些切换主要由 best-q 更新、离散同分和纯质量平局规则产生。",
            "",
            "## 分阶段激活",
            "",
            "| Phase | Route intervention | Anchor intervention |",
            "| --- | ---: | ---: |",
        ]
    )
    for phase in PHASES:
        route = overall_route["by_phase"][phase]
        anchor = overall_anchor["by_phase"][phase]
        lines.append(
            f"| {phase} | {route['interventions']}/{route['decisions']} ({_pct(route['intervention_rate'])}) | "
            f"{anchor['interventions']}/{anchor['decisions']} ({_pct(anchor['intervention_rate'], 1)}) |"
        )
    lines.extend(
        [
            "",
            "## 路线竞争尺度",
            "",
            "下表的 $\\Delta q$、$\\Delta o$ 和 $M=\\Delta o-\\Delta q$ 均取每次决策中对纯质量赢家威胁最大的竞争路线；$\\lambda_{\\mathrm{crit}}$ 是任一严格低质量、但欠投入优势为正的路线反超所需的最小尺度倍率。表中均报告 run 内中位数，不在任务间平均原始质量单位。",
            "",
            "| Run | $s$ | median $\\Delta q$ | median $\\Delta o$ | median $M$ | median $\\lambda_{\\mathrm{crit}}$ | zero $\\Delta q$ bootstrap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in result["runs"]:
        route = run["route"]
        lines.append(
            f"| {run['task_label']} r{run['repeat']} | {_number(run['optimism_scale'])} | "
            f"{_number(route['quality_gap']['median'])} | "
            f"{_number(route['optimism_advantage']['median'])} | "
            f"{_number(route['margin']['median'])} | "
            f"{_number(route['critical_multiplier']['median'])} | "
            f"{run['bootstrap_zero_delta_count']}/{run['bootstrap_delta_count']} |"
        )
    lines.extend(
        [
            "",
            "$M$ 在典型决策中为负，而且中位 $\\lambda_{\\mathrm{crit}}$ 约为当前尺度的 3–11 倍。这说明路线层的问题不是少数浮点边界未跨过，而是锚点一步尺度在路线竞争中通常不足以抵消路线 best-q 差。这些倍率只用于诊断尺度错位，不构成下一版的参数建议。",
            "",
            "## Bootstrap 0-delta",
            "",
            f"{aggregate['overall']['runs_with_zero_bootstrap_delta']}/{aggregate['runs']} 个 run 的 bootstrap 尺度集合包含至少一个零变化。正式实现将所有成功形成新 child state 的有效转换纳入中位数，包括 `delta_q = 0`；文档已与这一实际协议统一。",
            "",
            "## 机制诊断说明",
            "",
            "- 这是“机制运行但几乎没有改变路线选择”的过程负结果。",
            "- 它不代表两级结构本身失效，也不代表路线 coverage 对最终质量无价值。",
            "- 不能根据 intervention rate 反向盲目调大 $s$ 或把 $\lambda_{\mathrm{crit}}$ 的中位数直接写成下一版乘数。",
            "- V9.7 同时改变了其他机制；search 或 held-out 差异不应直接归因为 route optimism 提供了跨路线探索。",
            "",
            "## 可复现命令",
            "",
            "```bash",
            "uv run python experiments/traceaad_v9_7/analyze.py \\",
            f"  --batch {result['batch']} \\",
            "  --json-out experiments/traceaad_v9_7/analysis/allocation_summary.json \\",
            "  --markdown-out experiments/traceaad_v9_7/analysis/allocation_report.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="20260813_184519")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    result = analyze_batch(args.batch)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    if args.json_out is None and args.markdown_out is None:
        print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
