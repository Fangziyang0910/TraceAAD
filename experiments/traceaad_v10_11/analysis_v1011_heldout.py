#!/usr/bin/env python3
"""Summarize V10.11 generic / no_traj held-out results and compute main-table ranks.

Reads results.json from the two held-out output dirs and baseline rows from
docs/02-实验结果/主实验结果汇总.md; prints per-task tables and the 15-column
average rank (VRPTW counted separately) for both batches.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/fang/code/LLM4AD/LLM4AD')
GEN = ROOT / 'experiments/traceaad_v10_11/results_heldout_20260915_v1011_generic'
NTJ = ROOT / 'experiments/traceaad_v10_11/results_heldout_20260915_v1011_no_traj_idea'
SUMMARY = ROOT / 'docs/02-实验结果/主实验结果汇总.md'

# column spec: (doc table anchor regex, held-out column keys, lower_is_better)
TABLES = [
    ('TSP', [r'\| TSP50 \(Held-out\)'], ['tsp50', 'tsp100', 'tsp200'], True),
    ('CVRP', [r'\| CVRP50 \(Held-out\)'], ['test_50', 'test_100', 'test_200'], True),
    ('OP', [r'\| OP50 \(Held-out\)'], ['test_50', 'test_100', 'test_200'], False),
    ('OBP', [r'\| 1k_100 ↓'], ['1k_100', '5k_100', '10k_100', '1k_500', '5k_500', '10k_500'], True),
    ('VRPTW', [r'\| VRPTW50 \(Held-out\)'], ['vrptw50', 'vrptw100', 'vrptw200'], True),
]


def parse_baseline_rows():
    text = SUMMARY.read_text(encoding='utf-8')
    rows = {}
    for line in text.splitlines():
        if not line.startswith('| ') or 'Held-out' in line or line.startswith('| ---'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) not in (5, 8):
            continue
        name = cells[0]
        if not (name.startswith('EoH') or name.startswith('ReEvo') or name.startswith('MCTS')
                or name.startswith('PathWise') or name.startswith('CALM')):
            continue
        values = []
        for cell in cells[2:]:
            m = re.match(r'^([0-9.]+)', cell.strip('*').strip())
            values.append(float(m.group(1)) if m else None)
        if len(values) in (3, 6) and all(v is not None for v in values):
            rows.setdefault(name, []).extend(values)
    return rows


def load_batch_means(outdir):
    means = {}
    for task, key in [('tsp_construct', 'eval_results_by_size'), ('vrptw_construct', 'results_by_size')]:
        d = json.loads((outdir / task / 'results.json').read_text())
        for k, c in d[key].items():
            means[(task, k)] = [r['eval_score'] for r in c['results'] if 'eval_score' in r]
    for task in ['op_aco', 'cvrp_aco']:
        d = json.loads((outdir / task / 'results.json').read_text())
        for k, c in d['results_by_split'].items():
            means[(task, k)] = [r['eval_score'] for r in c['results'] if 'eval_score' in r]
    d = json.loads((outdir / 'online_bin_packing' / 'results.json').read_text())
    for k, c in d['eval_results_by_scale'].items():
        means[('online_bin_packing', k)] = [r['bins_used_mean'] for r in c['results'] if 'bins_used_mean' in r]
    return means


def avg_rank(column: dict[str, float], lower_is_better: bool) -> dict[str, float]:
    items = sorted(column.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
    ranks, i = {}, 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][1] == items[i][1]:
            j += 1
        rank = (i + 1 + j + 1) / 2
        for t in range(i, j + 1):
            ranks[items[t][0]] = rank
        i = j + 1
    return ranks


def main():
    base = parse_baseline_rows()
    baselines = {'EoH': [], 'ReEvo': [], 'MCTS-AHD': [], 'PathWise': [], 'CALM (w/o GRPO)': []}
    keymap = {
        'EoH': 'EoH', 'ReEvo': 'ReEvo', 'MCTS-AHD': 'MCTS-AHD',
        'PathWise': 'PathWise', 'CALM (w/o GRPO)': 'CALM (w/o GRPO)',
    }
    order = []
    for name, vals in base.items():
        canon = keymap.get(name)
        if canon is None:
            continue
        baselines[canon].extend(vals)
    # build ordered column list per task from the doc
    col_defs = []
    doc_order = {
        'TSP': ['tsp50', 'tsp100', 'tsp200'],
        'CVRP': ['test_50', 'test_100', 'test_200'],
        'OP': ['test_50', 'test_100', 'test_200'],
        'OBP': ['1k_100', '5k_100', '10k_100', '1k_500', '5k_500', '10k_500'],
        'VRPTW': ['vrptw50', 'vrptw100', 'vrptw200'],
    }
    # baseline rows appear in doc order TSP(3) CVRP(3) OP(3) OBP(6) VRPTW(3)
    for task in ['TSP', 'CVRP', 'OP', 'OBP', 'VRPTW']:
        for k in doc_order[task]:
            col_defs.append((task, k))

    for canon, vals in baselines.items():
        assert len(vals) == 18, (canon, len(vals))
        canon_cols = dict(zip(col_defs, vals))
        baselines[canon] = canon_cols

    for label, outdir in [('generic', GEN), ('no_traj', NTJ)]:
        means = load_batch_means(outdir)
        task_of = {'tsp_construct': 'TSP', 'cvrp_aco': 'CVRP', 'op_aco': 'OP',
                   'online_bin_packing': 'OBP', 'vrptw_construct': 'VRPTW'}
        print(f'===== {label} =====')
        mine = {}
        mine_obj = {}
        for (task, k), vals in means.items():
            if len(vals) != 3:
                print(f'  WARN {task} {k}: {len(vals)} values')
            mine[(task_of[task], k)] = vals
            # doc tables store positive objectives (distances / bins / prizes)
            sign = -1.0 if task in ('tsp_construct', 'cvrp_aco', 'vrptw_construct') else 1.0
            mine_obj[(task_of[task], k)] = [sign * v for v in vals]
        lower = {'TSP': True, 'CVRP': True, 'OP': False, 'OBP': True, 'VRPTW': True}
        task_cols = {'TSP': ['tsp50', 'tsp100', 'tsp200'],
                     'CVRP': ['test_50', 'test_100', 'test_200'],
                     'OP': ['test_50', 'test_100', 'test_200'],
                     'OBP': ['1k_100', '5k_100', '10k_100', '1k_500', '5k_500', '10k_500'],
                     'VRPTW': ['vrptw50', 'vrptw100', 'vrptw200']}
        for task, cols in task_cols.items():
            print(f'-- {task} --')
            for k in cols:
                vals = mine[(task, k)]
                m = statistics.fmean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                if task in ('TSP', 'CVRP', 'VRPTW'):
                    disp = -m
                else:
                    disp = m
                per = ' / '.join(f'{v:.4f}' for v in vals)
                print(f'  {k}: {per}  mean={disp:.4f} ±{sd:.4f}')
        # ranks over 15 columns (TSP/CVRP/OP/OBP), VRPTW separate
        rank_sum, rank_n = 0.0, 0
        all_ranks = {}
        for task in ['TSP', 'CVRP', 'OP', 'OBP']:
            for k in task_cols[task]:
                column = {name: cols_map[(task, k)] for name, cols_map in baselines.items()}
                column[label] = statistics.fmean(mine_obj[(task, k)])
                ranks = avg_rank(column, lower[task])
                all_ranks[(task, k)] = ranks
                rank_sum += ranks[label]
                rank_n += 1
        print('per-column ranks (15 cols):')
        for (task, k), ranks in all_ranks.items():
            print(f'  {task} {k}: {label}={ranks[label]:g}')
        print(f'15-column avg rank: {rank_sum / rank_n:.3f}')
        per_method = {name: statistics.fmean([r[name] for r in all_ranks.values()]) for name in baselines}
        per_method[label] = rank_sum / rank_n
        for name, avg in sorted(per_method.items(), key=lambda kv: kv[1]):
            print(f'  {name}: {avg:.3f}')
        vr_col = {}
        for k in task_cols['VRPTW']:
            column = {name: cols_map[('VRPTW', k)] for name, cols_map in baselines.items()}
            column[label] = statistics.fmean(mine_obj[('VRPTW', k)])
            ranks = avg_rank(column, True)
            vr_col[k] = ranks[label]
            print(f'  VRPTW {k}: rank {ranks[label]}')
        print(f'VRPTW avg rank: {statistics.fmean(list(vr_col.values())):.3f}')


if __name__ == '__main__':
    main()
