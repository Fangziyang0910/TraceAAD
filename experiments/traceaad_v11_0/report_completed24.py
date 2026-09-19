"""Render the selected V11 completed-24 held-out batch, including missing results."""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/traceaad_v11_0/results_heldout_20260917_completed24'
BASE = ROOT / 'experiments/traceaad_v10_11/results_heldout_20260915_v1011_generic'
DOC = ROOT / 'docs/02-实验结果/各版本实验记录/V11.0-实验结果.md'
TASKS = [
    ('tsp_construct', 'TSP ↓', 'eval_results_by_size', ['tsp50', 'tsp100', 'tsp200']),
    ('cvrp_aco', 'CVRP-ACO ↓', 'results_by_split', ['test_50', 'test_100', 'test_200']),
    ('op_aco', 'OP-ACO ↑', 'results_by_split', ['test_50', 'test_100', 'test_200']),
    ('online_bin_packing', 'OBP ↓', 'eval_results_by_scale', ['1k_100', '5k_100', '10k_100', '1k_500', '5k_500', '10k_500']),
    ('vrptw_construct', 'VRPTW ↓', 'results_by_size', ['vrptw50', 'vrptw100', 'vrptw200']),
]


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def main():
    selected = json.loads((OUT / 'selection.json').read_text())['runs']
    assert len(selected) == 24
    records = {}
    for task, _, _, _ in TASKS:
        dirname = 'vrptw_construct_per_run' if task == 'vrptw_construct' else task
        path = OUT / dirname / 'results.json'
        if path.exists():
            records[task] = json.loads(path.read_text())
    count = sum(len(d['run_records']) for d in records.values())
    state = json.loads((ROOT / 'experiments/traceaad_v11_0/results/cvrp_aco/20260917_cvrp_v110_rep3/tree_state.json').read_text())
    lines = [
        '# TraceAAD V11.0 实验结果（20260917 批次）', '',
        f'> 更新：{datetime.now().astimezone().isoformat(timespec="seconds")}。本次固定测试 24 路，已有 {count}/24 路写出完整测试记录。', '',
        '## 批次与范围', '',
        '正式批次为五任务 × 五重复（seed=0–4），每路 1000 次真实评价。用户指定测试启动时已经完成的 24 路：TSP、OP、OBP、VRPTW 各 rep1–5，CVRP rep1/2/4/5。CVRP rep3 不在本次 24 路测试集合中；即使随后完赛，也不自动混入本次均值。', '',
        f'CVRP rep3 已在 929/1000、candidate 939 的 `selected` 阶段保存并切换到 `local` 续跑；本次文档刷新时检查点为 {state["budget_used"]}/1000。保留原任务、seed、搜索机制、总预算和冻结运行时。实际本地调用和后续评价已写入该路 `llm_calls.jsonl`、`events.jsonl`。', '',
        '本次 held-out 对每路只选训练 best，未按测试分数重选程序。CVRP 为四重复，其余为五重复；V10.11 generic 对照为三重复。表中均值与样本标准差描述本次观察，不代表显著性检验，也不能单独归因为 V11 的某一项机制。', '',
        '## 训练 best（本次固定 24 路）', '',
        '| 任务 | rep1 | rep2 | rep3 | rep4 | rep5 | 均值 ± 样本标准差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for task, label, _, _ in TASKS:
        by_rep = {int(r['run_name'].rsplit('rep', 1)[1]): r['best_fitness'] * (1 if task == 'op_aco' else -1)
                  for r in selected if r['task'] == task}
        vals = list(by_rep.values())
        lines.append('| ' + label + ' | ' + ' | '.join(fmt(by_rep.get(i)) for i in range(1, 6)) +
                     f' | {fmt(statistics.fmean(vals))} ± {fmt(statistics.stdev(vals))} |')
    lines += ['', '## Held-out 测试', '',
              'TSP/VRPTW：seed 2025，16 实例，规模 50/100/200；CVRP/OP：固定 test_50/100/200，各 64 实例，ACO seed 1234，蚂蚁数/迭代数分别为 30/100 与 20/50；OBP：seed 2025，六档规模各五实例。评测复用 `experiments.infra.evaluate`，未修改五任务数据与种子契约。TSP timeout=3000 秒，VRPTW held-out timeout=1000 秒，OBP timeout=30 秒；TSP/ACO workers=8。', '',
              '初始 VRPTW 批量入口在 rep4 的训练 sanity 阶段返回无分数并中止，原始日志保留。补充脚本逐路记录训练 sanity 成败，继续评价同一 best 程序的三个 held-out 规模；未替换候选，也未把失败项删除后冒充完整均值。训练 sanity 仍使用冻结的 30 秒超时。', '']
    lines += ['表中若成功数小于应测数，均值仅描述成功子集，不能作为完整批次均值参与排名。', '']
    comparison = []
    errors = []
    successes = attempts = 0
    for task, label, container, units in TASKS:
        lines += [f'### {label}', '']
        if task not in records:
            lines += ['测试仍在运行；完整 `results.json` 尚未生成。', '']
            continue
        d = records[task]
        expected = {r['run_name'] for r in selected if r['task'] == task}
        assert {r['run_name'] for r in d['run_records']} == expected
        for r in d['run_records']:
            if r.get('train_eval_failed'):
                errors.append(f'{r["run_name"]} 训练 sanity：{r["train_eval_failed"]}')
        lines += ['| 规模 | rep1 | rep2 | rep3 | rep4 | rep5 | 均值 ± 样本标准差 | 成功/应测 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        baseline_path = BASE / task / 'results.json'
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        for unit in units:
            c = d[container][unit]
            rows = c['results']
            assert {r['run_name'] for r in rows} == expected
            by_rep = {int(r['run_name'].rsplit('rep', 1)[1]): r for r in rows}
            vals = [r['eval_objective'] for r in rows if 'eval_objective' in r]
            successes += len(vals)
            attempts += len(rows)
            for r in rows:
                if r.get('eval_failed'):
                    errors.append(f'{r["run_name"]} {unit}：{r["eval_failed"]}')
            cells = []
            for rep in range(1, 6):
                r = by_rep.get(rep)
                cells.append('—' if r is None else fmt(r['eval_objective']) if 'eval_objective' in r else '失败')
            mean = statistics.fmean(vals) if vals else None
            std = statistics.stdev(vals) if len(vals) > 1 else None
            lines.append(f'| {unit} | ' + ' | '.join(cells) + f' | {fmt(mean)} ± {fmt(std)} | {len(vals)}/{len(rows)} |')
            if baseline and len(vals) == len(rows):
                bm = baseline[container][unit]['summary']['mean_eval_objective']
                improvement = ((mean / bm - 1) if task == 'op_aco' else (1 - mean / bm)) * 100
                comparison.append((label, unit, mean, bm, improvement))
        lines += ['']
    lines += ['## 与 V10.11 generic 的同协议对照', '',
              '仅列完整成功的规模；改善率为正表示 V11 更好。重复数不同，CVRP 暂缺 rep3，因此均为阶段性描述性比较。', '',
              '| 任务 | 规模 | V11 | V10.11 generic | V11 改善率 |',
              '| --- | --- | ---: | ---: | ---: |']
    for task, unit, value, base, delta in comparison:
        lines.append(f'| {task} | {unit} | {fmt(value)} | {fmt(base)} | {delta:+.2f}% |')
    lines += ['', '## 失败与完成性', '',
              f'当前已落盘的完整结果覆盖 {count}/24 路，成功的「路 × 规模」测试为 {successes}/{attempts}；全批应有 87 项（24 × 3，加 OBP 5 × 3 项）。缺失文件视为待完成，不计作成功。', '']
    lines += [f'- {error}' for error in errors] or ['完整结果文件中尚无已记录的失败；未完成任务仍待核验。']
    if count == 24:
        lines += ['', '## 当前效果判断', '',
                  'V11 本次改善集中在 CVRP：四路均值在 50/100/200 分别改善 3.21%/3.20%/1.72%。TSP 三规模退步约 3.5%–4.3%；OP 退步随规模增大（0.37%/1.70%/3.58%），其中 rep1/3 在 200 规模较弱，说明训练均值接近并不保证跨规模稳定性。OBP 五档略好、一档略差，变化均不足 0.3%。VRPTW 50/100 退步 4.92%/2.34%，200 规模还有一路无分数，不能用四个成功值的均值代表五路完整结果。因此当前证据不支持 V11 整体优于 V10.11。', '',
                  '下一步机制分析应优先核对 TSP/VRPTW 的优质路线是否得到持续开发，以及 OP rep1/3 的跨规模决策是否退化；本批同时改变调度与 Pivot/Fuse 上下文，不能把版本差异直接归因于其中一项。VRPTW rep4 的高计算成本需要单独诊断，不能用换候选或删除失败路掩盖。', '']
    audit_path = OUT / 'training_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += ['## 相同真实评价预算下的训练进展', '',
                  '仅汇总本次固定 24 路；每个预算点使用该路已评价候选的 best-so-far。', '',
                  '| 任务 | E100 | E250 | E500 | E750 | E1000 | 评价失败/真实评价 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for task, label, _, _ in TASKS:
            a = audit[task]
            nfail = a['status_counts'].get('eval_failed', 0)
            lines.append('| ' + label + ' | ' + ' | '.join(fmt(a['curves'][str(k)]['mean']) for k in (100,250,500,750,1000)) + f' | {nfail}/{a["runs"] * 1000} |')
        lines += ['', '原始统计含初始化、修复候选与正常候选；invalid_output 不消耗真实评价预算，未计入上表分母。详见测试产物目录的 `training_audit.json`。', '']
    lines += ['', '## local 续跑与显存释放', '',
              '按用户最终明确的范围，停止 server1 的一个与 server3 的两个 vLLM，共三项：server1:8080、server3:8000、server3:8001。服务停止前请求队列为空；停止后 API/engine 进程和对应监听端口均已消失。三项 engine 原占用约 32004、30038、29820 MiB，合计 91862 MiB（约 89.71 GiB）。server1 GPU0 最后核对为 5 MiB、server3 GPU0 为 15 MiB；server3 GPU1 的剩余占用来自其他进程，不在本次释放范围。server3:8001 在范围澄清过程中短暂重启，最终已再次停止。', '',
              '## 可复核产物', '',
              '- 正式清单：`experiments/traceaad_v11_0/results/batch_20260917.json`。',
              '- 冻结运行时：`experiments/traceaad_v11_0/results/runtime_20260917/`。',
              '- 服务释放与迁移证据：`experiments/traceaad_v11_0/results/service_release_20260918.json`。',
              '- 固定 24 路选择及 best 标识：`experiments/traceaad_v11_0/results_heldout_20260917_completed24/selection.json`。',
              '- 测试入口：同目录 `run_heldout.sh`，VRPTW 容错补测入口为 `run_vrptw_per_run.py`。',
              '- 结果：同目录各任务 `results.json`；VRPTW 为 `vrptw_construct_per_run/results.json`，原始失败日志为 `vrptw_construct.log`。',
              '- 本页由 `.venv/bin/python experiments/traceaad_v11_0/report_completed24.py` 从固定选择与结果文件生成。', '']
    DOC.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'document': str(DOC), 'complete_run_records': count, 'successful_cells': successes,
                      'attempted_cells': attempts, 'all_results_written': count == 24}, ensure_ascii=False))


if __name__ == '__main__':
    main()
