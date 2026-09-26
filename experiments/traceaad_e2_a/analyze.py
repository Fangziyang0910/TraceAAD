"""Prospective E2-A trajectory-state interaction and Q-kernel audit."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from experiments.traceaad_refine_e1.prepare import ROOT, dump
from experiments.traceaad_refine_e1.replay import predict
from .prepare import DEFAULT

PRIMARY_TASKS = ['tsp_construct', 'online_bin_packing', 'vrptw_construct']
LABELS = {
    'tsp_construct': 'TSP', 'cvrp_aco': 'CVRP', 'op_aco': 'OP',
    'online_bin_packing': 'OBP', 'vrptw_construct': 'VRPTW',
}
OUTCOMES = ['frontier_gain', 'frontier', 'parent_positive_gain', 'valid', 'parent_signed_gain_valid']
DOC = ROOT / 'experiments/traceaad_e2_a/report'
SEED = 20260907


def load_run(out, run):
    folder = out / 'snapshot' / run['run_name']
    nodes = {n['id']: n for n in json.loads((folder / 'tree_state.json').read_text())['nodes']}
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    born = {e['node_id']: e['candidate_id'] for e in events if e.get('node_id') is not None}
    return nodes, events, born


def trajectory_state(event, nodes, born, ids, behavior):
    pid = event['parent_id']
    formation_parent = nodes[pid]['parent_id']
    if formation_parent is None:
        return None
    index = {node_id: i for i, node_id in enumerate(ids)}
    if pid not in index or formation_parent not in index:
        return None
    available = [node_id for node_id in ids if born[node_id] < event['candidate_id']
                 and node_id not in [pid, formation_parent]]
    if not available:
        return None
    move = float(behavior[index[pid], index[formation_parent]])
    revisit = float(min(behavior[index[pid], index[node_id]] for node_id in available))
    deltas = []
    operators = []
    current = pid
    for _ in range(3):
        parent = nodes[current]['parent_id']
        if parent is None:
            break
        denominator = max(abs(nodes[parent]['fitness']), 1e-8)
        deltas.append(float((nodes[current]['fitness'] - nodes[parent]['fitness']) / denominator))
        operators.append(nodes[current]['operator'])
        current = parent
    if not deltas:
        return None
    return {
        'move': move,
        'revisit': revisit,
        'recent_gain': float(sum(deltas)),
        'recent_deltas': deltas,
        'recent_operators': operators,
    }


def make_behavior_rows(out, manifest):
    prefix_states = defaultdict(list)
    suffix_rows = []
    coverage = defaultdict(int)
    for run in manifest['runs']:
        if run['task'] not in PRIMARY_TASKS:
            continue
        nodes, events, born = load_run(out, run)
        distance_folder = out / 'distances' / run['run_name']
        ids = json.loads((distance_folder / 'ids.json').read_text())
        behavior = np.load(distance_folder / 'behavior.npy')
        for event in events:
            if event['requested_operator'] not in ['Refine', 'Pivot'] or event['operator'] != event['requested_operator']:
                continue
            state = trajectory_state(event, nodes, born, ids, behavior)
            is_suffix = event['candidate_id'] > run['e1_cutoff_attempt']
            coverage['suffix_binary_actions' if is_suffix else 'prefix_binary_actions'] += 1
            if state is None:
                coverage['suffix_missing_state' if is_suffix else 'prefix_missing_state'] += 1
                continue
            if not is_suffix:
                prefix_states[run['task']].append(state)
                continue
            conditional = event['selection']['operator_conditional']
            pivot_propensity = conditional['Pivot'] / (conditional['Pivot'] + conditional['Refine'])
            if not 0.05 <= pivot_propensity <= 0.95:
                coverage['suffix_outside_propensity_overlap'] += 1
                continue
            valid = event.get('fitness') is not None
            parent_delta = float(event['fitness'] - event['parent_fitness']) if valid else None
            frontier_delta = float(event['fitness'] - event['best_before']) if valid else None
            suffix_rows.append({
                'run': run['run_name'], 'task': run['task'], 'candidate_id': event['candidate_id'],
                'parent_id': event['parent_id'], 'action': event['requested_operator'],
                'pivot_propensity': float(pivot_propensity), **state,
                'valid': float(valid),
                'frontier': float(valid and frontier_delta > 0),
                'frontier_gain': max(frontier_delta or 0.0, 0.0) / max(abs(event['best_before']), 1e-8),
                'parent_positive_gain': max(parent_delta or 0.0, 0.0) / max(abs(event['parent_fitness']), 1e-8),
                'parent_signed_gain_valid': (parent_delta / max(abs(event['parent_fitness']), 1e-8)) if valid else None,
            })
    thresholds = {
        task: {
            'move_median': float(np.median([s['move'] for s in states])),
            'revisit_median': float(np.median([s['revisit'] for s in states])),
            'prefix_states': len(states),
        }
        for task, states in prefix_states.items()
    }
    for row in suffix_rows:
        threshold = thresholds[row['task']]
        flags = [row['move'] <= threshold['move_median'],
                 row['revisit'] <= threshold['revisit_median'],
                 row['recent_gain'] <= 0]
        row['state_flags'] = [bool(v) for v in flags]
        row['stagnation_score'] = sum(flags)
        row['stagnant'] = row['stagnation_score'] >= 2
        row['strict_stagnant'] = row['stagnation_score'] == 3
    return suffix_rows, thresholds, dict(coverage)


def hajek(rows, outcome, stagnant):
    group = [r for r in rows if r['stagnant'] == stagnant and r[outcome] is not None]
    result = {}
    for action in ['Refine', 'Pivot']:
        selected = [r for r in group if r['action'] == action]
        propensity = np.array([1 - r['pivot_propensity'] if action == 'Refine' else r['pivot_propensity'] for r in selected])
        weights = 1 / propensity
        values = np.array([r[outcome] for r in selected])
        result[action] = {
            'n': len(selected),
            'estimate': float(weights @ values / weights.sum()) if len(selected) else None,
            'ess': float(weights.sum() ** 2 / (weights @ weights)) if len(selected) else 0.0,
            'propensity_min': float(propensity.min()) if len(selected) else None,
            'propensity_max': float(propensity.max()) if len(selected) else None,
            'positive': int((values > 0).sum()),
        }
    tau = None
    if result['Refine']['estimate'] is not None and result['Pivot']['estimate'] is not None:
        tau = result['Pivot']['estimate'] - result['Refine']['estimate']
    return {'Refine': result['Refine'], 'Pivot': result['Pivot'], 'tau': tau}


def interaction(rows, outcome):
    low = hajek(rows, outcome, False)
    high = hajek(rows, outcome, True)
    delta = high['tau'] - low['tau'] if high['tau'] is not None and low['tau'] is not None else None
    return {'nonstagnant': low, 'stagnant': high, 'interaction': delta}


def bootstrap_interaction(rows, outcome, samples=10000):
    rng = np.random.default_rng(SEED)
    by_task_run = {
        task: {run: [r for r in rows if r['task'] == task and r['run'] == run]
               for run in sorted({r['run'] for r in rows if r['task'] == task})}
        for task in PRIMARY_TASKS
    }
    values = []
    for _ in range(samples):
        task_values = []
        for task, run_rows in by_task_run.items():
            names = list(run_rows)
            sampled = rng.choice(names, size=len(names), replace=True)
            batch = [row for name in sampled for row in run_rows[name]]
            value = interaction(batch, outcome)['interaction']
            if value is None:
                break
            task_values.append(value)
        if len(task_values) == len(PRIMARY_TASKS):
            values.append(float(np.mean(task_values)))
    return {
        'valid_resamples': len(values),
        'samples': samples,
        'ci95': np.quantile(values, [0.025, 0.975]).tolist() if values else [None, None],
        'median': float(np.median(values)) if values else None,
    }


def behavior_summary(rows, thresholds, coverage):
    by_task = {task: {outcome: interaction([r for r in rows if r['task'] == task], outcome)
                      for outcome in OUTCOMES} for task in PRIMARY_TASKS}
    overall = {outcome: interaction(rows, outcome) for outcome in OUTCOMES}
    macro = {}
    bootstraps = {}
    for outcome in OUTCOMES:
        values = [by_task[t][outcome]['interaction'] for t in PRIMARY_TASKS]
        macro[outcome] = float(np.mean(values)) if all(v is not None for v in values) else None
        bootstraps[outcome] = bootstrap_interaction(rows, outcome)
    strict = []
    for row in rows:
        copy = dict(row)
        copy['stagnant'] = row['strict_stagnant']
        strict.append(copy)
    strict_results = {outcome: interaction(strict, outcome) for outcome in OUTCOMES}
    primary = macro['frontier_gain']
    primary_ci = bootstraps['frontier_gain']['ci95']
    directions = sum(by_task[t]['frontier_gain']['interaction'] is not None
                     and by_task[t]['frontier_gain']['interaction'] > 0 for t in PRIMARY_TASKS)
    stagnant_primary = overall['frontier_gain']['stagnant']
    ess_ok = all(stagnant_primary[action]['ess'] >= 20 for action in ['Refine', 'Pivot'])
    corroborating = [o for o in ['frontier', 'parent_positive_gain'] if macro[o] is not None and macro[o] > 0]
    support_runs = {
        outcome: sorted({r['run'] for r in rows if r['stagnant'] and r['action'] == 'Pivot'
                         and r[outcome] is not None and r[outcome] > 0})
        for outcome in ['frontier', 'parent_positive_gain']
    }
    corroboration_ok = any(len(support_runs[o]) >= 2 for o in corroborating)
    frontier_events = sum(r['frontier'] for r in rows)
    informative = bool(frontier_events >= 10 and ess_ok)
    gate = bool(primary is not None and primary > 0 and primary_ci[0] is not None and primary_ci[0] > 0
                and directions >= 2 and ess_ok and corroboration_ok)
    return {
        'thresholds_from_e1_prefix': thresholds,
        'coverage': coverage | {
            'analyzed_suffix_actions': len(rows),
            'stagnant': sum(r['stagnant'] for r in rows),
            'strict_stagnant': sum(r['strict_stagnant'] for r in rows),
        },
        'overall': overall,
        'by_task': by_task,
        'task_macro_interaction': macro,
        'bootstrap': bootstraps,
        'strict_stagnant_sensitivity': strict_results,
        'support_runs_with_positive_stagnant_pivot_outcome': support_runs,
        'gate_components': {
            'frontier_gain_interaction': primary,
            'frontier_gain_ci95': primary_ci,
            'positive_task_directions': directions,
            'stagnant_ess_ok': ess_ok,
            'corroborating_outcomes': corroborating,
            'corroboration_multiple_runs': corroboration_ok,
            'frontier_events': int(frontier_events),
            'minimum_information_for_negative': informative,
        },
        'interaction_gate_passed': gate,
        'decision_status': 'positive' if gate else 'negative' if informative else 'undetermined',
    }


def quality_kernel(pid, visible, history, nodes, prior):
    pool = [node_id for node_id in visible if node_id != pid]
    if not pool:
        return prior
    distances = np.array([abs(nodes[pid]['fitness'] - nodes[node_id]['fitness']) for node_id in pool])
    order = np.argsort(distances, kind='stable')[:10]
    neighbors = [pool[i] for i in order]
    near = distances[order]
    width = max(float(near[-1]), 1e-6)
    weights = np.exp(-0.5 * (near / width) ** 2)
    counts = np.array([history[node_id][0] for node_id in neighbors])
    successes = np.array([history[node_id][1] for node_id in neighbors])
    return float((10 * prior + weights @ successes) / (10 + weights @ counts))


def prediction_metrics(rows, model):
    y = np.array([r['y'] for r in rows])
    p = np.clip([r['predictions'][model] for r in rows], 1e-6, 1 - 1e-6)
    top = np.argsort(-p, kind='stable')[:max(1, math.ceil(0.2 * len(rows)))]
    return {
        'n': len(rows), 'positive': int(y.sum()),
        'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'average_precision': float(average_precision_score(y, p)) if y.any() else None,
        'lift': float(y[top].mean() / y.mean()) if y.any() else None,
        'top_parent_positive_gain': float(np.mean([rows[i]['parent_positive_gain'] for i in top])),
        'top_frontier_gain': float(np.mean([rows[i]['frontier_gain'] for i in top])),
        'top_frontier_rate': float(np.mean([rows[i]['frontier'] for i in top])),
    }


def q_audit(out, manifest):
    saved_e1 = {(r['run'], r['candidate_id']): r for r in map(
        json.loads, (ROOT / 'experiments/traceaad_refine_e1/raw/refine_e1_20260907/replay_predictions.jsonl').read_text().splitlines())}
    suffix = []
    q_errors = []
    m1_errors = []
    for run in manifest['runs']:
        nodes, events, _ = load_run(out, run)
        visible = set(); history = defaultdict(lambda: [0, 0]); past_count = 0; past_success = 0
        budget_before = 0; raw_rows = []; q_values = []
        for event in events:
            if event['requested_operator'] == event['operator'] == 'Refine':
                prior = (1 + past_success) / (10 + past_count)
                q = quality_kernel(event['parent_id'], visible, history, nodes, prior)
                valid = event.get('fitness') is not None
                parent_delta = float(event['fitness'] - event['parent_fitness']) if valid else None
                frontier_delta = float(event['fitness'] - event['best_before']) if valid else None
                raw_rows.append({
                    'y': int(bool(event.get('parent_improved'))), 'prior': prior,
                    'features': {'M1': [event['parent_fitness'], np.log1p(event['selection']['parent_count_before']), budget_before / 1000]},
                    'candidate_id': event['candidate_id'], 'task': run['task'], 'run': run['run_name'],
                    'is_suffix': event['candidate_id'] > run['e1_cutoff_attempt'],
                    'parent_positive_gain': max(parent_delta or 0.0, 0.0) / max(abs(event['parent_fitness']), 1e-8),
                    'frontier_gain': max(frontier_delta or 0.0, 0.0) / max(abs(event['best_before']), 1e-8),
                    'frontier': int(valid and frontier_delta > 0),
                })
                q_values.append(q)
                visible.add(event['parent_id'])
                history[event['parent_id']][0] += 1
                history[event['parent_id']][1] += int(bool(event.get('parent_improved')))
                past_count += 1; past_success += int(bool(event.get('parent_improved')))
            budget_before = event['budget_used']
        predictions = list(predict(raw_rows, models=['M0', 'M1']))
        for raw, pred, q in zip(raw_rows, predictions, q_values):
            row = {k: v for k, v in raw.items() if k != 'features'}
            row['predictions'] = pred['predictions'] | {'Q_kernel': q}
            if not raw['is_suffix']:
                old = saved_e1[(run['run_name'], raw['candidate_id'])]
                q_errors.append(abs(q - old['predictions']['Q_kernel']))
                m1_errors.append(abs(row['predictions']['M1'] - old['predictions']['M1']))
            else:
                suffix.append(row)
    assert max(q_errors) < 1e-12 and max(m1_errors) < 1e-12
    models = ['M0', 'M1', 'Q_kernel']
    overall = {m: prediction_metrics(suffix, m) for m in models}
    by_task = {task: {m: prediction_metrics([r for r in suffix if r['task'] == task], m) for m in models}
               for task in LABELS}
    by_run = {run: {m: prediction_metrics([r for r in suffix if r['run'] == run], m) for m in models}
              for run in sorted({r['run'] for r in suffix})}
    macro = {}
    for model in models:
        macro[model] = {}
        for metric in ['brier', 'log_loss', 'lift', 'top_parent_positive_gain', 'top_frontier_gain']:
            values = [v[model][metric] for v in by_run.values() if v[model][metric] is not None]
            macro[model][metric] = float(np.mean(values)) if values else None
    tasks_better = sum(v['Q_kernel']['brier'] < v['M1']['brier'] for v in by_task.values())
    gate = bool(overall['Q_kernel']['brier'] < overall['M1']['brier'] and tasks_better >= 3
                and macro['Q_kernel']['top_frontier_gain'] >= macro['M1']['top_frontier_gain']
                and macro['Q_kernel']['top_parent_positive_gain'] >= macro['M1']['top_parent_positive_gain'])
    return suffix, {
        'coverage': {'suffix_refine': len(suffix), 'improved': sum(r['y'] for r in suffix),
                     'frontier': sum(r['frontier'] for r in suffix),
                     'runs_without_refine_improvement': sum(not any(r['y'] for r in suffix if r['run'] == run)
                                                            for run in by_run)},
        'overall': overall, 'run_macro': macro, 'by_task': by_task, 'by_run': by_run,
        'tasks_brier_better': tasks_better,
        'prefix_reconstruction_max_abs_error': {'Q_kernel': max(q_errors), 'M1': max(m1_errors)},
        'allocation_candidate_gate_passed': gate,
    }


def fmt(value, digits=6):
    return '—' if value is None else f'{value:.{digits}f}'


def write_report(manifest, behavior_rows, behavior, qsummary, profile_coverage):
    gate = behavior['interaction_gate_passed']
    qgate = qsummary['allocation_candidate_gate_passed']
    g = behavior['gate_components']
    status = behavior['decision_status']
    if g['frontier_events'] < 10 and not g['stagnant_ess_ok']:
        information_reason = f'主任务suffix只有 {g["frontier_events"]} 个frontier event，且停滞层action ESS不足'
    elif g['frontier_events'] < 10:
        information_reason = f'主任务suffix只有 {g["frontier_events"]} 个frontier event；停滞层action ESS已达最低要求'
    else:
        information_reason = '停滞层action ESS不足'
    state = ('达到固定交互门槛，仍需E2-B确认。' if status == 'positive' else
             '最低信息量充足但未观察到固定的停滞交互，不修改在线机制。' if status == 'negative' else
             '未达到最低信息量，只能判未决；不修改在线机制，也不自动启动E2-B。')
    lines = ['# E2-A 结果：行为轨迹状态与 Refine/Pivot 响应', '',
        f'快照时间：`{manifest.get("created_at") or next((v for k,v in manifest.items() if k.endswith("_at")),None)}`。**阶段判断：{state}**', '',
        '本实验使用E1截点之后的未见suffix；TSP、OBP、VRPTW用于行为交互，五任务用于Q_kernel审计。没有新增LLM生成或正式evaluator调用。', '',
        '## 1. 数据与状态覆盖', '',
        f'快照15路共 {sum(r["suffix_attempts"] for r in manifest["runs"])} 个新尝试。行为主任务中有 {behavior["coverage"]["suffix_binary_actions"]} 个requested=executed Refine/Pivot动作，'
        f'{behavior["coverage"]["analyzed_suffix_actions"]} 个具有完整非根轨迹状态；其中 {behavior["coverage"]["stagnant"]} 个满足2/3停滞定义，{behavior["coverage"]["strict_stagnant"]} 个三项全部满足。', '',
        '| 任务 | Prefix状态数 | Move中位数 | Revisit中位数 | Suffix可分析动作 | 停滞动作 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for task in PRIMARY_TASKS:
        threshold = behavior['thresholds_from_e1_prefix'][task]
        task_rows = [r for r in behavior_rows if r['task'] == task]
        lines.append(f'| {LABELS[task]} | {threshold["prefix_states"]} | {threshold["move_median"]:.6f} | {threshold["revisit_median"]:.6f} | {len(task_rows)} | {sum(r["stagnant"] for r in task_rows)} |')
    lines += ['', f'D_revisit使用action前完整有效archive，排除当前节点及其直接形成父节点；阈值只由E1 prefix确定。根节点、画像失败或参考集不足不填零，排除并计入覆盖。另有 {behavior["coverage"].get("suffix_outside_propensity_overlap",0)} 个状态因二元Pivot propensity不在[0.05,0.95]而排除，结论只适用于overlap population。', '',
        '## 2. 停滞状态 × 算子响应', '',
        f'主frontier_gain交互的三任务宏平均为 {fmt(g["frontier_gain_interaction"])}，task内分层run bootstrap 95%区间 [{fmt(g["frontier_gain_ci95"][0])}, {fmt(g["frontier_gain_ci95"][1])}]；'
        f'{g["positive_task_directions"]}/3任务方向为正。固定门槛判定为 **{"通过" if gate else "未通过"}**。', '',
        '| Outcome | 非停滞 Pivot−Refine | 停滞 Pivot−Refine | 交互Δτ（三任务宏平均） | 95% bootstrap区间 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for outcome in OUTCOMES:
        overall = behavior['overall'][outcome]
        ci = behavior['bootstrap'][outcome]['ci95']
        lines.append(f'| {outcome} | {fmt(overall["nonstagnant"]["tau"])} | {fmt(overall["stagnant"]["tau"])} | {fmt(behavior["task_macro_interaction"][outcome])} | [{fmt(ci[0])}, {fmt(ci[1])}] |')
    lines += ['', 'frontier两行的全零与[0,0]区间来自主任务suffix中没有任何frontier event，是不可识别的退化结果，不是“效应被精确估计为零”。主表前两列是三任务池化Hájek估计，交互列是三个任务等权平均，因此数值不要求由前两列直接相减。`parent_signed_gain_valid`条件于处理后的valid，只用于描述，不作因果主结论。', '',
        '| 层 | Refine n / ESS | Pivot n / ESS | Refine frontier gain | Pivot frontier gain |',
        '| --- | ---: | ---: | ---: | ---: |']
    for key, label in [('nonstagnant', '非停滞'), ('stagnant', '停滞')]:
        value = behavior['overall']['frontier_gain'][key]
        lines.append(f'| {label} | {value["Refine"]["n"]} / {value["Refine"]["ess"]:.1f} | {value["Pivot"]["n"]} / {value["Pivot"]["ess"]:.1f} | {value["Refine"]["estimate"]:.6f} | {value["Pivot"]["estimate"]:.6f} |')
    parent_high = behavior['overall']['parent_positive_gain']['stagnant']
    parent_low = behavior['overall']['parent_positive_gain']['nonstagnant']
    strict_parent = behavior['strict_stagnant_sensitivity']['parent_positive_gain']['stagnant']
    lines += ['',
        f'辅助超父结果没有显示Pivot优势：非停滞层Refine/Pivot分别有 {parent_low["Refine"]["positive"]}/{parent_low["Pivot"]["positive"]} 次正增益，停滞层为 {parent_high["Refine"]["positive"]}/{parent_high["Pivot"]["positive"]}；'
        f'严格三项停滞子集也只有Refine {strict_parent["Refine"]["positive"]}/{strict_parent["Refine"]["n"]}、Pivot {strict_parent["Pivot"]["positive"]}/{strict_parent["Pivot"]["n"]}。这些事件太少，不能形成稳定负效应估计，但方向并不支持花费生成预算启动E2-B。', '',
        f'给定候选有效时，signed parent gain的三任务宏平均交互为 {behavior["task_macro_interaction"]["parent_signed_gain_valid"]:.6f}，bootstrap区间 [{behavior["bootstrap"]["parent_signed_gain_valid"]["ci95"][0]:.6f}, {behavior["bootstrap"]["parent_signed_gain_valid"]["ci95"][1]:.6f}]，且三个任务方向均为负。valid是处理后的变量，不能把这个条件分析当作无偏因果效应；它只能说明现有suffix没有出现值得据此投入E2-B的正向质量迹象。', '',
        '![E2-A逐任务交互](interaction.png)', '',
        '图中同时给出frontier gain与parent positive gain的逐任务交互。正值表示停滞状态下Pivot相对Refine的优势比非停滞状态更大。', '',
        '### 分任务主结果', '',
        '| 任务 | Frontier交互 | Parent-positive交互 | 停滞Refine/Pivot n |', '| --- | ---: | ---: | ---: |']
    for task in PRIMARY_TASKS:
        frontier = behavior['by_task'][task]['frontier_gain']
        parent = behavior['by_task'][task]['parent_positive_gain']
        high = frontier['stagnant']
        lines.append(f'| {LABELS[task]} | {fmt(frontier["interaction"])} | {fmt(parent["interaction"])} | {high["Refine"]["n"]}/{high["Pivot"]["n"]} |')
    lines += ['', '## 3. Q_kernel prospective suffix audit', '',
        f'五任务suffix包含 {qsummary["coverage"]["suffix_refine"]} 次严格Refine、{qsummary["coverage"]["improved"]} 次超父和 {qsummary["coverage"]["frontier"]} 次前缘突破。'
        f'Q与M1在E1 prefix上的重构最大误差分别为 {qsummary["prefix_reconstruction_max_abs_error"]["Q_kernel"]:.3g} 和 {qsummary["prefix_reconstruction_max_abs_error"]["M1"]:.3g}。'
        f'{qsummary["coverage"]["runs_without_refine_improvement"]}/15个run没有Refine正例，这些run的Brier/log loss仍保留，lift只对可定义run求宏平均。', '',
        '| 模型 | Brier（池化）↓ | Brier（run宏平均）↓ | AP↑ | Lift（宏平均）↑ | Top20% parent gain↑ | Top20% frontier gain↑ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model in ['M0', 'M1', 'Q_kernel']:
        overall = qsummary['overall'][model]; macro = qsummary['run_macro'][model]
        lines.append(f'| {model} | {overall["brier"]:.6f} | {macro["brier"]:.6f} | {overall["average_precision"]:.4f} | {macro["lift"]:.3f} | {macro["top_parent_positive_gain"]:.6f} | {macro["top_frontier_gain"]:.6f} |')
    lines += ['', f'Q_kernel在 {qsummary["tasks_brier_better"]}/5 个任务上Brier优于M1，allocation candidate门槛判定为 **{"通过" if qgate else "未通过"}**。该门槛同时要求parent与frontier两种top组收益不低于M1，避免把校准改善直接写成预算价值。', '',
        f'这次不是只在排序上失败：Q_kernel的池化Brier为 {qsummary["overall"]["Q_kernel"]["brier"]:.6f}，也弱于M1的 {qsummary["overall"]["M1"]["brier"]:.6f}；AP、run宏平均lift和两项top组gain均更低。E1中观察到的Q校准优势没有在未见suffix复制。', '',
        '## 4. 研究判断', '',
        ('当前suffix提供了符合固定门槛的停滞交互，但它仍是三任务、一步结果；下一步只能用既定锚点E2-B确认两步option value，不能直接修改V10.6。' if status == 'positive' else
         '当前suffix在最低信息量充分的条件下没有建立“行为停滞时应提高Pivot”的可靠一步证据，当前停滞定义不形成可迁移控制信号。' if status == 'negative' else
         f'{information_reason}，因此无法把未通过门槛解释为机制不存在。结果只能记为未决，且信息不足本身不自动授权E2-B。'), '',
        ('Q_kernel同时复制了校准与预算排序价值，可保留为独立候选，但仍需新的正式策略比较。' if qgate else
         'Q_kernel既没有复制校准优势，也没有改善预算相关排序，因此不再保留为选父候选；E1中的优势应视为特定prefix上的不稳定校准现象。'), '',
        'E1已经关闭静态行为邻域的Refine响应共享；E2-A只检验行为作为轨迹传感器的一个固定状态定义。未通过不否定行为移动、重访对Pivot/Fuse或多步continuation的其他作用，但后续问题必须用新的独立数据或固定锚点回答，不能继续在本suffix寻找切点。', '',
        '## 5. 成本与复现', '',
        f'行为画像覆盖：{sum(r["valid_profiles"] for r in profile_coverage["runs"])}/{sum(r["nodes"] for r in profile_coverage["runs"])} 个archive节点；复用E1成功画像 {profile_coverage["reused_success_profiles"]} 个，新画像任务 {profile_coverage["new_profile_jobs"]} 个。新增画像worker池墙钟 {profile_coverage["new_profile_wall_seconds"]/60:.1f} 分钟，不含随后距离矩阵汇总时间。', '',
        '[执行入口](../README.md)。机器可读产物位于 `experiments/traceaad_e2_a/raw/traceaad_e2_a_20260907/`。', '']
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / 'README.md').write_text('\n'.join(lines))


def plot(behavior):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, outcome, title in zip(axes, ['frontier_gain', 'parent_positive_gain'], ['Frontier gain', 'Parent positive gain']):
        values = [behavior['by_task'][t][outcome]['interaction'] for t in PRIMARY_TASKS]
        if outcome == 'frontier_gain' and not any(values):
            ax.text(0.5, 0.5, 'No frontier events\nin primary-task suffix', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks(range(3), [LABELS[t] for t in PRIMARY_TASKS])
            ax.set_ylim(-1, 1)
        else:
            ax.bar([LABELS[t] for t in PRIMARY_TASKS], values, color=['#4477aa', '#66a061', '#cc8844'])
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.set_title(title); ax.set_ylabel('Interaction: (Pivot-Refine) stagnant minus nonstagnant')
    fig.suptitle('E2-A prospective trajectory-state interaction')
    fig.savefig(DOC / 'interaction.png', dpi=180)
    plt.close(fig)


def main(out=DEFAULT):
    config = json.loads(Path(__file__).with_name('e2a_config.json').read_text())
    assert config['protocol_set_before_suffix_outcomes_read']
    dump(out / 'e2a_config.json', config)
    manifest = json.loads((out / 'snapshot.json').read_text())
    profile_coverage = json.loads((out / 'profile_coverage.json').read_text())
    rows, thresholds, coverage = make_behavior_rows(out, manifest)
    dump(out / 'behavior_rows.json', rows)
    behavior = behavior_summary(rows, thresholds, coverage)
    qrows, qsummary = q_audit(out, manifest)
    dump(out / 'behavior_summary.json', behavior)
    dump(out / 'q_audit_summary.json', qsummary)
    (out / 'q_audit_predictions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in qrows))
    combined = {'config': config, 'snapshot': manifest, 'profile_coverage': profile_coverage,
                'behavior': behavior, 'q_audit': qsummary}
    dump(out / 'summary.json', combined)
    dump(DOC / 'summary.json', combined)
    write_report(manifest, rows, behavior, qsummary, profile_coverage)
    plot(behavior)
    print(json.dumps({'interaction_gate': behavior['interaction_gate_passed'],
                      'q_allocation_gate': qsummary['allocation_candidate_gate_passed'],
                      'behavior_actions': len(rows), 'q_refine': len(qrows)}))


if __name__ == '__main__':
    main()
