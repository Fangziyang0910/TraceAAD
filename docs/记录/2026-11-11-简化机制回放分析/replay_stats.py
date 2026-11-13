"""在 Base 日志上流式回放 evotrace_v1 的重构规则，统计保留角色、移除区域与各条规则的作用。

运行：python3 docs/记录/2026-11-11-简化机制回放分析/replay_stats.py
只读日志并调用 plan_consolidation/apply_consolidation；不调用模型，不评价程序。
后代的日志父代若已被移除，改挂到最近的保留祖先。
"""
import glob
import itertools
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evotrace_v1.models import Edge, Node, TreePopulation  # noqa: E402
from evotrace_v1.topology import TopologyConfig, apply_consolidation, code_distance, plan_consolidation  # noqa: E402

BATCH = ROOT / "experiments_result/traceaad_base_boltzmann_20261109"
TASKS = ["tsp_construct", "cvrp_aco", "fssp_gls", "op_aco"]
A = TopologyConfig()
VARIANTS = {
    "去掉方向配额": replace(A, max_directions=10**6, max_per_direction=10**6),
    "去掉克隆过滤": replace(A, clone_distance=0.0),
    "去掉里程碑": replace(A, milestone_distance=2.0),
    "去掉深度上限": replace(A, max_direction_depth=10**6),
    "去掉方向归并": replace(A, direction_merge_distance=0.0),
    "去掉质量下限": replace(A, frontier_quality_floor=0.0),
    "去掉新方向保护": replace(A, protected_new_directions=0),
}


def load(path):
    rows = []
    for line in open(path):
        p = json.loads(line).get("program")
        if not p:
            continue
        ev = p.get("evaluation") or {}
        rows.append(dict(id=p["id"], parent=p.get("parent_id") or 0, op=p.get("operator"), ref=p.get("reference_id"),
                         code=p["code"], design=p.get("design") or "", fit=ev.get("fitness"), fail=ev.get("failure")))
    return rows


def replay(rows):
    tree, alias, removed_ever = TreePopulation(root_id=0), {}, set()
    s, absorbed, jac = defaultdict(float), Counter(), defaultdict(list)
    best, rounds = float("inf"), 0
    for r in rows:
        if rounds:
            s["later"] += 1
            s["later_parent_removed"] += r["parent"] in removed_ever
            if r["fit"] is not None and r["fit"] < best:
                s["new_best"] += 1
                s["new_best_parent_removed"] += r["parent"] in removed_ever
        if r["fit"] is not None:
            best = min(best, r["fit"])
        p = r["parent"]
        while p in alias:
            p = alias[p]
        if p and p not in tree.nodes:
            p = 0
        tree.nodes[r["id"]] = Node(id=r["id"], code=r["code"], design=r["design"], fitness=r["fit"], failure=r["fail"])
        tree.edges[r["id"]] = Edge(parent_id=p, child_id=r["id"], operator=None if r["op"] in (None, "Init") else r["op"],
                                   reference_id=r["ref"])
        if len(tree.nodes) < A.trigger_capacity:
            continue
        rounds += 1
        plan = plan_consolidation(tree, A)
        kept = set(plan.kept_ids)
        for name, cfg in VARIANTS.items():
            other = set(plan_consolidation(tree, cfg).kept_ids)
            jac[name].append(len(kept & other) / len(kept | other))
        s["kept"] += len(kept)
        s["frontier"] += len(plan.frontier_ids)
        s["lca"] += len(plan.lca_anchor_ids)
        s["milestone"] += len(plan.milestone_ids)
        s["contracted_nodes"] += sum(len(c.removed_ids) for c in plan.path_contractions)
        cons = sorted(plan.branch_consolidations, key=lambda b: -len(b.removed_ids))
        for b in cons:
            n = len(b.removed_ids)
            s["side_nodes"] += n
            s["not_shown"] += max(0, n - 6)                      # MAX_PRUNED_SAMPLES = 6
            s["other_parent"] += sum(tree.edges[x].parent_id != b.parent_id for x in b.removed_ids)
            s["max_per_parent"] = max(s["max_per_parent"], n)
            absorbed[b.parent_id] += n
        s["beyond_calls"] += sum(len(b.removed_ids) for b in [b for b in cons if b.parent_id][8:])  # max_selection_calls = 8
        for x in tree.nodes:
            if x not in kept:
                cur = tree.edges[x].parent_id
                while cur and cur not in kept:
                    cur = tree.edges[cur].parent_id
                alias[x] = cur
                removed_ever.add(x)
        tree = apply_consolidation(tree, plan)
    s["rounds"] = rounds
    top = sorted(absorbed.values(), reverse=True)
    s["top1_absorbed"], s["top5_absorbed"], s["all_absorbed"] = top[0], sum(top[:5]), sum(top)
    return s, jac


def direction_distance(rows, rng):
    prog = {r["id"]: r for r in rows if r["fit"] is not None}
    step, direction = defaultdict(list), {}
    for i in sorted(prog):
        r = prog[i]
        if r["parent"] in prog and r["op"] in ("Refine", "Crossover", "Explore"):
            step[r["op"]].append(code_distance(prog[r["parent"]]["code"], r["code"]))
        new = r["parent"] not in prog or r["op"] == "Explore"
        direction[i] = i if new else direction[r["parent"]]
    ids = sorted(prog)[:200]
    within, between = [], []
    for a, b in rng.sample(list(itertools.combinations(ids, 2)), 1500):
        (within if direction[a] == direction[b] else between).append(code_distance(prog[a]["code"], prog[b]["code"]))
    return step, within, between, len({direction[i] for i in sorted(prog)[:64]})


def main():
    rng = random.Random(0)
    total, jac_all = defaultdict(float), defaultdict(list)
    for task in TASKS:
        step, within, between, ndirs = defaultdict(list), [], [], []
        for path in sorted(glob.glob(str(BATCH / task / "*/native/attempts.jsonl"))):
            rows = load(path)
            s, jac = replay(rows)
            for k, v in s.items():
                total[k] = max(total[k], v) if k == "max_per_parent" else total[k] + v
            for k, v in jac.items():
                jac_all[k] += v
            st_, w, b, nd = direction_distance(rows, rng)
            for k, v in st_.items():
                step[k] += v
            within += w
            between += b
            ndirs.append(nd)
        print(f"{task}: 前64个可行节点的方向数 {ndirs}; 父子距离中位数 "
              + ", ".join(f"{k} {st.median(v):.3f}" for k, v in sorted(step.items()))
              + f"; 同方向节点对 {st.mean(within):.3f}, 跨方向节点对 {st.mean(between):.3f}")
    r = total["rounds"]
    print(f"\n重构 {int(r)} 轮（12路）。每轮均值：保留 {total['kept']/r:.1f}，前沿 {total['frontier']/r:.1f}，"
          f"分叉锚点 {total['lca']/r:.1f}，里程碑 {total['milestone']/r:.1f}")
    print(f"每轮移除：路径收缩 {total['contracted_nodes']/r:.2f} 个节点，旁支整理 {total['side_nodes']/r:.2f} 个节点")
    side = total["side_nodes"]
    print(f"旁支节点中：超出每父6个样本而未展示 {total['not_shown']/side:.1%}，超出8次调用走确定性摘要 "
          f"{total['beyond_calls']/side:.1%}，真实父代不是承接节点 {total['other_parent']/side:.1%}，单父单轮最多 {int(total['max_per_parent'])}")
    print(f"单路内吸收最多的1个父节点占全部旁支 {total['top1_absorbed']/total['all_absorbed']:.1%}，前5个占 "
          f"{total['top5_absorbed']/total['all_absorbed']:.1%}")
    print(f"首轮重构后 Base 的尝试：父代已被移除的占 {total['later_parent_removed']/total['later']:.1%}；"
          f"刷新全局最优的 {int(total['new_best'])} 步中，父代已被移除的占 {total['new_best_parent_removed']/total['new_best']:.1%}")
    print("\n去掉单条规则后，保留集合与默认规则的 Jaccard（同一棵重构前的树）：")
    for name, v in jac_all.items():
        print(f"  {name}: 均值 {st.mean(v):.3f}，完全相同 {sum(x == 1.0 for x in v)}/{len(v)} 轮")


if __name__ == "__main__":
    main()
