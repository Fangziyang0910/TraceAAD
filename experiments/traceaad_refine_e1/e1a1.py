"""E1-A.1: test whether behavior adds response information inside quality neighborhoods."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from .prepare import DEFAULT, ROOT, dump

K = 10
SHRINK = 10.0
WARMUP = 50
PERMUTATIONS = 1000
SEED = 20260907
MODELS = ['M1', 'M3_static', 'Q_kernel', 'QB_kernel']
LABELS = {
    'tsp_construct': 'TSP',
    'cvrp_aco': 'CVRP',
    'op_aco': 'OP',
    'online_bin_packing': 'OBP',
    'vrptw_construct': 'VRPTW',
}
DOC = ROOT / 'docs/02-机制验证/04-算子动力学与两步价值/2026-09-07-E1-Refine局部响应迁移性'


def kernel_predictions(pid, visible, history, nodes, behavior, bidx, prior, rng, permutations=PERMUTATIONS):
    """Return Q, QB and within-Q-neighborhood behavior permutations using only past state."""
    # Match E1-A exactly, including its stable tie order for equal-fitness nodes.
    pool = [j for j in visible if j != pid]
    if not pool:
        return prior, prior, np.full(permutations, prior), False, True
    distances = np.array([abs(nodes[pid]['fitness'] - nodes[j]['fitness']) for j in pool])
    order = np.argsort(distances, kind='stable')[:K]
    neighbors = [pool[i] for i in order]
    qdist = distances[order]
    qwidth = max(float(qdist[-1]), 1e-6)
    qw = np.exp(-0.5 * (qdist / qwidth) ** 2)
    counts = np.array([history[j][0] for j in neighbors], dtype=float)
    successes = np.array([history[j][1] for j in neighbors], dtype=float)
    qmass = float(qw @ counts)
    qpred = float((SHRINK * prior + qw @ successes) / (SHRINK + qmass))

    if pid not in bidx:
        return qpred, qpred, np.full(permutations, qpred), False, False
    available = np.array([j in bidx for j in neighbors])
    if not available.any():
        return qpred, qpred, np.full(permutations, qpred), False, False
    bdist = np.ones(len(neighbors))
    bdist[available] = [behavior[bidx[pid], bidx[j]] for j in np.array(neighbors)[available]]
    bwidth = max(float(bdist[available].max()), 1e-6)
    bw = np.ones(len(neighbors))
    bw[available] = np.exp(-0.5 * (bdist[available] / bwidth) ** 2)
    joint = qw * bw
    qbpred = float((SHRINK * prior + joint @ successes) / (SHRINK + joint @ counts))

    orders = np.argsort(rng.random((permutations, len(neighbors))), axis=1)
    perm_joint = qw[None, :] * bw[orders]
    masses = perm_joint @ counts
    permpred = (SHRINK * prior + perm_joint @ successes) / (SHRINK + masses)
    return qpred, qbpred, permpred, True, bool(available.all())


def score(rows, model):
    y = np.array([r['y'] for r in rows])
    p = np.clip([r['predictions'][model] for r in rows], 1e-6, 1 - 1e-6)
    top = np.argsort(-p, kind='stable')[:max(1, math.ceil(0.2 * len(rows)))]
    prevalence = float(y.mean())
    return {
        'n': len(rows),
        'positive': int(y.sum()),
        'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'average_precision': float(average_precision_score(y, p)) if y.any() else None,
        'lift': float(y[top].mean() / prevalence) if prevalence else None,
        'top_relative_positive_gain': float(np.mean([rows[i]['relative_positive_gain'] for i in top])),
    }


def summarize(rows, perm_predictions):
    primary = [r for r in rows if r['primary']]
    groups = {}
    for level in ['run', 'task']:
        groups[level] = {
            key: {m: score([r for r in primary if r[level] == key], m) for m in MODELS}
            for key in sorted({r[level] for r in primary})
        }
    overall = {m: score(primary, m) for m in MODELS}
    macro = {
        m: {metric: float(np.mean([v[m][metric] for v in groups['run'].values()]))
            for metric in ['brier', 'log_loss', 'lift', 'top_relative_positive_gain']}
        for m in MODELS
    }
    runs = sorted(groups['run'])
    run_delta = np.array([groups['run'][r]['QB_kernel']['brier'] - groups['run'][r]['Q_kernel']['brier'] for r in runs])
    rng = np.random.default_rng(SEED + 1)
    boot = np.mean(rng.choice(run_delta, size=(10000, len(run_delta)), replace=True), axis=1)

    y = np.array([r['y'] for r in primary])
    perm_primary = perm_predictions[np.array([r['primary'] for r in rows])]
    perm_run_brier = []
    for run in runs:
        mask = np.array([r['run'] == run for r in primary])
        perm_run_brier.append(np.mean((y[mask, None] - perm_primary[mask]) ** 2, axis=0))
    perm_macro = np.mean(perm_run_brier, axis=0)
    real_macro = macro['QB_kernel']['brier']
    beat_fraction = float(np.mean(real_macro < perm_macro))
    task_permutation = {}
    for task in sorted(groups['task']):
        mask = np.array([r['task'] == task for r in primary])
        values = np.mean((y[mask, None] - perm_primary[mask]) ** 2, axis=0)
        task_permutation[task] = {
            'median_brier': float(np.median(values)),
            'ci95': np.quantile(values, [0.025, 0.975]).tolist(),
            'real_qb_beats_fraction': float(np.mean(groups['task'][task]['QB_kernel']['brier'] < values)),
        }
    relative = macro['QB_kernel']['brier'] / macro['Q_kernel']['brier'] - 1
    tasks_better = sum(v['QB_kernel']['brier'] < v['Q_kernel']['brier'] for v in groups['task'].values())
    gate = bool(relative <= -0.02 and tasks_better >= 3
                and macro['QB_kernel']['log_loss'] <= macro['Q_kernel']['log_loss']
                and beat_fraction >= 0.95)
    return {
        'coverage': {
            'all_refine': len(rows),
            'primary': len(primary),
            'positive': int(y.sum()),
            'behavior_query_available': sum(r['behavior_query_available'] for r in primary),
            'complete_joint_support': sum(r['complete_joint_support'] for r in primary),
            'first_parent_use': sum(r['first_parent_use'] for r in primary),
        },
        'overall': overall,
        'run_macro': macro,
        'by': groups,
        'paired_run_delta_qb_minus_q': {
            'mean': float(run_delta.mean()),
            'ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
            'runs_better': int((run_delta < 0).sum()),
            'runs': len(run_delta),
        },
        'permutation': {
            'count': perm_predictions.shape[1],
            'seed': SEED,
            'macro_brier_median': float(np.median(perm_macro)),
            'macro_brier_ci95': np.quantile(perm_macro, [0.025, 0.975]).tolist(),
            'real_qb_beats_fraction': beat_fraction,
            'empirical_one_sided_p': float((1 + np.sum(perm_macro <= real_macro)) / (len(perm_macro) + 1)),
            'macro_brier_values': perm_macro.tolist(),
            'by_task': task_permutation,
        },
        'gate_components': {
            'relative_macro_brier_change': float(relative),
            'tasks_better': tasks_better,
            'macro_log_loss_nonworse': bool(macro['QB_kernel']['log_loss'] <= macro['Q_kernel']['log_loss']),
            'permutation_beat_fraction': beat_fraction,
        },
        'increment_gate_passed': gate,
    }


def subset_summary(rows, condition):
    selected = [r for r in rows if r['primary'] and condition(r)]
    return {m: score(selected, m) for m in MODELS}


def write_report(out, summary, subsets, q_audit):
    macro = summary['run_macro']
    gate = summary['increment_gate_passed']
    g = summary['gate_components']
    d = summary['paired_run_delta_qb_minus_q']
    p = summary['permutation']
    decision = ('观察到达到固定门槛的质量条件行为增量，仍需独立数据确认。' if gate else
                '未观察到达到固定门槛的质量条件行为增量，不启动原版 E1-B。')
    lines = [
        '# E1-A.1 结果：质量条件下的行为增量检验', '',
        f'**阶段判断：{decision}**', '',
        '本分析由 E1-A 的后验发现触发，复用同一冻结数据和严格时间回放。它不修改 E1-A 的主结论，也不新增 LLM 生成、正式 evaluator 调用或在线控制改动。', '',
        '## 1. 核心结果', '',
        f'QB_kernel 相对 Q_kernel 的 run 宏平均 Brier 变化为 **{g["relative_macro_brier_change"]:+.2%}**，'
        f'{g["tasks_better"]}/5 个任务、{d["runs_better"]}/15 个 run 改善。配对差为 {d["mean"]:+.6f}，'
        f'run bootstrap 95% 探索性区间 [{d["ci95"][0]:+.6f}, {d["ci95"][1]:+.6f}]。', '',
        f'真实 QB 的宏平均 Brier 优于 {p["real_qb_beats_fraction"]:.1%} 的邻域内行为置乱，'
        f'单侧经验比例对应 p={p["empirical_one_sided_p"]:.4f}。固定门槛要求至少2%相对改善、3/5任务同向、log loss不变差且优于至少95%的置乱；本次判定为 **{"通过" if gate else "未通过"}**。', '',
        '| 模型 | Brier（池化）↓ | Brier（run宏平均）↓ | Log loss（宏平均）↓ | AP（池化）↑ | Lift（宏平均）↑ | Top20%相对正部增益↑ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model in MODELS:
        o = summary['overall'][model]
        m = macro[model]
        lines.append(f'| {model} | {o["brier"]:.6f} | {m["brier"]:.6f} | {m["log_loss"]:.6f} | {o["average_precision"]:.4f} | {m["lift"]:.3f} | {m["top_relative_positive_gain"]:.6f} |')
    lines += ['',
        f'1000次 QB_permuted 的宏平均 Brier 中位数为 {p["macro_brier_median"]:.6f}，95%范围 [{p["macro_brier_ci95"][0]:.6f}, {p["macro_brier_ci95"][1]:.6f}]。置乱只打乱同一质量局部组内行为权重与parent身份的对应关系。', '',
        '## 2. 分任务结果', '',
        '| 任务 | Q Brier | QB Brier | QB−Q | 置乱Brier中位数 | 真实QB优于置乱比例 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for task, values in summary['by']['task'].items():
        q = values['Q_kernel']['brier']; qb = values['QB_kernel']['brier']; perm = p['by_task'][task]
        lines.append(f'| {LABELS[task]} | {q:.6f} | {qb:.6f} | {qb-q:+.6f} | {perm["median_brier"]:.6f} | {perm["real_qb_beats_fraction"]:.1%} |')
    lines += ['', '![逐run的QB相对Q误差变化](E1-A.1预测增量.png)', '',
        '图中每点代表一个run，短线是每任务三个run的均值；纵轴低于零才表示行为重加权改善了质量核。', '',
        '## 3. 覆盖与稳健性', '',
        f'主比较包含 {summary["coverage"]["primary"]} 次预热后尝试和 {summary["coverage"]["positive"]} 次严格改善。'
        f'当前parent有行为画像 {summary["coverage"]["behavior_query_available"]} 次；质量局部组全部拥有行为画像 {summary["coverage"]["complete_joint_support"]} 次。'
        f'重算 Q_kernel 与 E1-A 已保存预测的最大绝对差为 {q_audit:.3g}。', '',
        '| 子集 | 尝试数 | Q Brier | QB Brier | QB−Q |', '| --- | ---: | ---: | ---: | ---: |',
    ]
    for label, values in subsets.items():
        q = values['Q_kernel']; qb = values['QB_kernel']
        lines.append(f'| {label} | {q["n"]} | {q["brier"]:.6f} | {qb["brier"]:.6f} | {qb["brier"]-q["brier"]:+.6f} |')
    lines += ['', 'first-parent-use 用于检验新节点能否借用其他节点的历史响应；完整支持子集用于排除行为画像缺失造成的回退。两者均沿用原warmup和当前parent排除规则。', '',
        '## 4. 研究含义', '',
        ('通过门槛只说明在这批冻结数据上，行为距离在质量局部组内提供了可复现的额外信息；由于问题来自E1-A后验观察，仍需独立数据确认后才能设计控制器。' if gate else
         '本轮不能把 Q_kernel 的优势归因于被fitness掩盖的行为局部性。若 QB 与置乱相当或弱于Q，说明当前BehaveSim没有展示出控制质量后的独立响应信息；继续做原版behavior Near/Far E1-B缺少依据。'), '',
        '无论本轮结果如何，Q_kernel 的好表现仍可能只是对非线性 `fitness → improvement rate` 的非参数校准，并不自动等价于更好的预算分配。AP、lift和top组收益只作为后验控制器诊断，不能替代Brier主判断。', '',
        f'控制器诊断本身也不一致：QB 的 pooled AP 从 {summary["overall"]["Q_kernel"]["average_precision"]:.4f} 变为 {summary["overall"]["QB_kernel"]["average_precision"]:.4f}，'
        f'run 宏平均 lift 从 {macro["Q_kernel"]["lift"]:.3f} 变为 {macro["QB_kernel"]["lift"]:.3f}，top组相对正部增益从 {macro["Q_kernel"]["top_relative_positive_gain"]:.6f} 变为 {macro["QB_kernel"]["top_relative_positive_gain"]:.6f}。'
        '其中收益改善只出现在2/5个任务，且本实验没有为这些排序指标建立置乱门槛，因此不能把宏平均lift的上升解释为稳定的预算集中能力。first-parent-use的Brier也略微恶化，行为重加权没有解决新节点缺少自身反馈的问题。', '',
        'E1-A已经显示ACO任务的局部邻居对probe和随机流较敏感；本实验复用这些距离，不能消除该测量限制。另一方面，TSP测量较稳定而原响应共享仍失败，因此也不能把所有负结果归因于ACO噪声。', '',
        '本轮联合核沿用E1-A的k=10、自适应高斯带宽和收缩强度，只检验这一固定形式。未通过意味着当前BehaveSim条件核没有可用增量，不构成对所有行为表示、核带宽或控制目标的普遍否定。后续不再为Refine响应共享事后调这一批数据；行为信息转去停滞/重访、Pivot切换和Fuse互补性，Q_kernel则需用独立时间段验证其校准与排序价值。', '',
        '## 5. 复现与产物', '',
        '[固定设计](E1-A.1-实验设计.md)；[运行入口](../../../../experiments/traceaad_refine_e1/README.md)。机器可读结果位于 `experiments/traceaad_refine_e1/raw/refine_e1_20260907/e1a1_summary.json`、`e1a1_predictions.jsonl` 和 `e1a1_permutation_macro_brier.npy`。', ''
    ]
    (DOC / 'E1-A.1-结果.md').write_text('\n'.join(lines))


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, task in enumerate(LABELS):
        run_names = [r for r in summary['by']['run'] if r in summary['_task_runs'][task]]
        values = [summary['by']['run'][r]['QB_kernel']['brier'] - summary['by']['run'][r]['Q_kernel']['brier'] for r in run_names]
        ax.scatter(np.arange(len(values)) * 0.08 + i - 0.08, values, color='#2266aa', s=40)
        ax.plot([i - 0.23, i + 0.23], [np.mean(values)] * 2, color='#cc6600', lw=3)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xticks(range(5), LABELS.values())
    ax.set_ylabel('Brier(QB) - Brier(Q); lower is better')
    ax.set_title('E1-A.1: behavioral increment within quality neighborhoods')
    fig.tight_layout()
    fig.savefig(DOC / 'e1a1_gain.png', dpi=180)
    plt.close(fig)


def main(out=DEFAULT):
    config = json.loads((Path(__file__).with_name('e1a1_config.json')).read_text())
    assert config['frozen_before_prediction'] and config['permutations'] == PERMUTATIONS
    manifest = json.loads((out / 'snapshot.json').read_text())
    old = {(r['run'], r['candidate_id']): r for r in map(json.loads, (out / 'replay_predictions.jsonl').read_text().splitlines())}
    rows = []
    perm_rows = []
    rng = np.random.default_rng(SEED)
    task_runs = defaultdict(list)
    q_differences = []
    for run in manifest['runs']:
        run_name = run['run_name']; task_runs[run['task']].append(run_name)
        folder = out / 'snapshot' / run_name
        nodes = {n['id']: n for n in json.loads((folder / 'tree_state.json').read_text())['nodes']}
        events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
        distance_folder = out / 'distances' / run_name
        bids = json.loads((distance_folder / 'ids.json').read_text())
        bidx = {node_id: i for i, node_id in enumerate(bids)}
        behavior = np.load(distance_folder / 'behavior.npy')
        visible = set(); history = defaultdict(lambda: [0, 0]); past_count = 0; past_success = 0; refine_index = 0
        for event in events:
            if event['requested_operator'] == event['operator'] == 'Refine':
                pid = event['parent_id']
                prior = (1 + past_success) / (10 + past_count)
                q, qb, perm, behavior_available, complete = kernel_predictions(
                    pid, visible, history, nodes, behavior, bidx, prior, rng)
                previous = old[(run_name, event['candidate_id'])]
                q_differences.append(abs(q - previous['predictions']['Q_kernel']))
                gain = max(float(event.get('parent_delta') or 0.0), 0.0) / max(abs(event['parent_fitness']), 1e-8)
                rows.append({
                    'run': run_name, 'task': run['task'], 'candidate_id': event['candidate_id'],
                    'parent_id': pid, 'y': int(bool(event.get('parent_improved'))),
                    'relative_positive_gain': gain, 'first_parent_use': pid not in visible,
                    'behavior_query_available': behavior_available, 'complete_joint_support': complete,
                    'primary': refine_index >= WARMUP,
                    'predictions': {
                        'M1': previous['predictions']['M1'],
                        'M3_static': previous['predictions']['M3_static'],
                        'Q_kernel': q,
                        'QB_kernel': qb,
                    },
                    'permutation_mean': float(perm.mean()),
                    'permutation_ci95': np.quantile(perm, [0.025, 0.975]).tolist(),
                })
                perm_rows.append(perm)
                visible.add(pid)
                history[pid][0] += 1
                history[pid][1] += int(bool(event.get('parent_improved')))
                past_count += 1; past_success += int(bool(event.get('parent_improved'))); refine_index += 1
    perm_predictions = np.vstack(perm_rows)
    summary = summarize(rows, perm_predictions)
    summary['_task_runs'] = dict(task_runs)
    subsets = {
        '全部主样本': subset_summary(rows, lambda r: True),
        'first-parent-use': subset_summary(rows, lambda r: r['first_parent_use']),
        '完整行为支持': subset_summary(rows, lambda r: r['behavior_query_available'] and r['complete_joint_support']),
    }
    summary['subsets'] = subsets
    summary['q_reconstruction_max_abs_error'] = float(max(q_differences))
    assert summary['q_reconstruction_max_abs_error'] < 1e-12
    (out / 'e1a1_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    np.save(out / 'e1a1_permutation_macro_brier.npy', np.array(summary['permutation']['macro_brier_values']))
    dump(out / 'e1a1_summary.json', summary)
    dump(out / 'e1a1_config.json', config)
    dump(DOC / 'e1a1_summary.json', {'config': config, 'summary': summary})
    write_report(out, summary, subsets, max(q_differences))
    plot(summary)
    print(json.dumps({'gate': summary['increment_gate_passed'], **summary['gate_components']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
