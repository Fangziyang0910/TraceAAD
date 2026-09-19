"""Summarize frozen E1-A measurements without turning correlations into causal claims."""
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from .prepare import DEFAULT,dump,ROOT
from .replay import MODELS

LABELS={'tsp_construct':'TSP','cvrp_aco':'CVRP','op_aco':'OP','online_bin_packing':'OBP','vrptw_construct':'VRPTW'}
DOC=ROOT/'docs/02-机制验证/04-算子动力学与两步价值/2026-09-07-E1-Refine局部响应迁移性'


def auxiliary_table(out):
    rows=json.loads((out/'auxiliary_summary.json').read_text())
    result=['| 目标与误差（run宏平均） | M0 | M1 | M2 | M3 |','| --- | ---: | ---: | ---: | ---: |']
    for target,label,metric in [('catastrophe','严重退化 Brier','brier'),('frontier','全局突破 Brier','brier'),('relative_positive_gain','相对正部增益 MSE','mse')]:
        group=[r for r in rows if r['target']==target]
        values=[float(np.mean([r['metrics'][m][metric] for r in group])) for m in ['M0','M1','M2','M3']]
        result.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    return result


def main(out=DEFAULT):
    snap=json.loads((out/'snapshot.json').read_text());s=json.loads((out/'summary.json').read_text())
    stability=json.loads((out/'stability.json').read_text());emb=json.loads((out/'embedding_metadata.json').read_text())
    seeds=json.loads((out/'seed_sensitivity.json').read_text())
    assert len(seeds)==10
    new=json.loads((out/'summary_new_parents.json').read_text());complete=json.loads((out/'summary_complete_profiles.json').read_text())
    costs=defaultdict(lambda:dict(candidates=0,panels=0,seconds=0.0,valid=0,errors=Counter()))
    for task in LABELS:
        for p in (out/'profiles'/task).glob('*.json'):
            r=json.loads(p.read_text());c=costs[task];c['candidates']+=1
            c['valid']+=int(all(x['ok'] for x in r['panels'].values()))
            for panel in r['panels'].values():
                c['panels']+=1;c['seconds']+=panel['elapsed_seconds']
                if not panel['ok']:c['errors'][panel['error_type']]+=1
    dump(out/'costs.json',dict(costs))
    macro={m:{metric:float(np.mean([v[m][metric] for v in s['by']['run'].values()])) for metric in ['brier','log_loss','lift']} for m in MODELS}
    relative=(macro['M3']['brier']/macro['M1']['brier']-1)
    tasks_better=sum(v['M3']['brier']<v['M1']['brier'] for v in s['by']['task'].values())
    gate=relative<=-.02 and tasks_better>=3 and macro['M3']['log_loss']<=macro['M1']['log_loss']
    decision='达到事先固定的进入 E1-B 筛选门槛；仍需固定锚点确认，不能直接上线。' if gate else '未达到事先固定的进入 E1-B 筛选门槛；本轮不据此自动启动固定锚点生成或修改在线选父。'
    lines=['# E1-A 结果：V10.6 Refine 局部响应历史回放','',f'数据冻结：`{snap["frozen_at"]}`。状态：E1-A 已完成；E1-B 未执行。','',
        f'**阶段判断：{decision}**','',
        f'M3 相对 M1 的 run 宏平均 Brier 变化为 **{relative:+.2%}**（负值更好），{tasks_better}/5 个任务的池化 Brier 改善。下面同时给出简单 prior、embedding 和质量邻域对照，避免将局部相关性误写成潜力已可识别。','',
        '### 研究判断','',
        '本轮不支持把当前 BehaveSim 邻域及响应特征直接加入选父控制器。M3 的 Brier、log loss 和排序指标均未稳定超过 M1；去掉唯一缺失画像样本后，以及只看父节点首次被开发时，结论方向不变。加入局部响应的 M3 也弱于仅用静态行为量的 M3_static，因此不能声称已验证轨迹反馈共享的增量。','',
        '同时，结果不能简化为“局部反馈无用”。无需拟合的 B_kernel 宏平均 Brier 为0.06108，优于 M1 的0.06242，但配对差的探索性95%区间跨0。更便宜的 Q_kernel 为0.05954，且在五个任务的池化 Brier 上都优于 B_kernel。当前证据更支持先检查质量条件化的收缩估计；尚未证明行为几何有独立于质量相近性的额外价值。拟合模型弱于直接收缩估计，提示有限正例下的拟合与校准可能是瓶颈，但本轮没有通过额外调参来验证这一解释。','',
        'OBP 是唯一 M3 任务池化改善的任务，三个重复中两个改善；但该任务 M3 仍弱于全局 prior（0.05559 对0.05196），也弱于静态行为模型（0.05409）。因此，仅凭 OBP 相对 M1 的改善，不足以支持缩小范围后立即进入 E1-B。ACO 的随机流敏感性则提示行为距离还需要更充分的可靠性核验。','',
        '建议下一步先在新的时间段或独立重复上，事先固定比较 M0、Q_kernel 与 B_kernel，并对 ACO 增加随机流重复以分离采样噪声。若行为邻域在质量条件化对照之外仍有稳定增量，再启动固定锚点近/远配对。本轮保留原门槛和全部负结果，不通过事后更换主模型将实验改判为通过。','',
        '## 1. 实际数据与执行范围','',
        f'冻结15路修订版 V10.6，共 {sum(r["refine"] for r in snap["runs"])} 次请求且执行 Refine、{sum(r["refine_parents"] for r in snap["runs"])} 个父节点、{sum(r["improved"] for r in snap["runs"])} 次严格改善。每路前50次 Refine 为预热，主比较包含 {s["coverage"]["primary"]} 次尝试。原始事件与评价收据已逐路核对；解析失败、评价失败均保留为未改善，未完成请求排除。冻结数据中没有请求算子与执行算子不一致的记录，故无 Fuse 回退样本需要单列。','',
        '| 任务 | 重复 | 已用评价截点 | Refine尝试 | 父节点 | 严格改善 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in snap['runs']:lines.append(f'| {LABELS[r["task"]]} | {r["repeat"]} | {r["cutoff_evaluation"]} | {r["refine"]} | {r["refine_parents"]} | {r["improved"]} |')
    lines+=['',f'预热后样本中，实际提交评价 {s["coverage"]["evaluated"]} 次，得到有限fitness {s["coverage"]["finite_children"]} 次。无条件改善率为 {s["coverage"]["improved"]/s["coverage"]["primary"]:.2%}；给定已提交评价为 {s["coverage"]["improved"]/s["coverage"]["evaluated"]:.2%}；给定有限fitness为 {s["coverage"]["improved"]/s["coverage"]["finite_children"]:.2%}。', '', '## 2. 测量校验与画像覆盖','',
        '每任务首路运行按node ID选两个最早入档的Refine父节点、两个面板共20个校验案例，重复PSTraj一致，正式评价器与画像过程在相同probe上的得分差均为0。这是有限候选一致性检查，不是所有可能程序的等价证明。','',
        '| 任务 | 安排画像父节点 | 双面板成功 | 面板Spearman均值 | kNN重合均值 | 累计worker墙钟秒 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for t in LABELS:
        rs=[r for r in stability if r['task']==t];c=costs[t]
        rho=np.mean([r['panel_spearman'] for r in rs if r['panel_spearman'] is not None]);over=np.mean([r['knn_overlap'] for r in rs if r['knn_overlap'] is not None])
        lines.append(f'| {LABELS[t]} | {c["candidates"]} | {c["valid"]} | {rho:.3f} | {over:.3f} | {c["seconds"]:.1f} |')
    lines+=['','### 随机流敏感性','', '| 任务 | 同算法跨随机流平均距离（2锚点） | 已画像父节点最近邻距离中位数（run均值） |', '| --- | ---: | ---: |']
    for t in LABELS:
        selfd=[r['self_distance'] for r in seeds if r['task']==t and r['self_distance'] is not None]
        near=[]
        for run in snap['runs']:
            if run['task']==t:
                mat=np.load(out/'distances'/run['run_name']/'A.npy')
                near.append(float(np.median(np.min(mat+np.eye(len(mat))*1e6,axis=1))))
        lines.append(f'| {LABELS[t]} | {np.mean(selfd):.3f} | {np.mean(near):.3f} |')
    lines+=['', '随机流检查使用每任务两个最早锚点，改变候选程序随机种子，并在ACO任务中改变ACO采样种子；只在面板A检查。若同算法跨随机流差异与跨算法距离接近，固定随机流的近邻可能含有显著采样噪声。此处是小样本敏感性检查，不能据两个锚点推断任务的完整噪声分布。', '', '画像失败的类别与每个候选详情保存在本地 `profiles/` 和 `costs.json`，失败不填成零距离。面板稳定性是观测可靠性的描述；密度等特征仅针对当时已被选择为Refine parent的历史子集，并非全archive。',
        '', 'OP面板B使用seed=20260907的独立训练probe；CVRP面板B使用训练集4–7号实例。未读取validation/test。与旧v3采用validation面板的数值不能直接等同比较。',
        '', '## 3. 严格时间预测结果','',
        '各run独立扩展窗口，每10次尝试仅用此前结果重拟合固定C=1逻辑回归；没有跨run未来训练或根据测试分数选超参。M2/M3采用相同预测器和可见邻域规则。AP为average precision，lift为本组最高20%预测的实际改善率/本组总体改善率。',
        '', '| 模型 | Brier（池化）↓ | Brier（run宏平均）↓ | Log loss（宏平均）↓ | AP（池化）↑ | Lift（宏平均）↑ |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in MODELS:
        v=s['overall'][m];a=macro[m]
        lines.append(f'| {m} | {v["brier"]:.5f} | {a["brier"]:.5f} | {a["log_loss"]:.5f} | {v["pr_auc"]:.4f} | {a["lift"]:.3f} |')
    lines+=['','M0=历史平滑prior；M1=fitness/count/stage；M2=embedding邻域；M3_static=行为静态局部量；M3=进一步加入局部历史响应；M3_samecode_excluded=排除同代码邻居；M3_self_history=允许同父反馈；M_quality=质量距离邻域；B_kernel/E_kernel/Q_kernel分别直接输出行为/embedding/质量邻域的收缩改善率，作为无需拟合的诊断对照。','',
        '| 任务 | M1 Brier | M2 Brier | M3_static Brier | M3 Brier | M3−M1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for t,v in s['by']['task'].items():lines.append(f'| {LABELS[t]} | {v["M1"]["brier"]:.5f} | {v["M2"]["brier"]:.5f} | {v["M3_static"]["brier"]:.5f} | {v["M3"]["brier"]:.5f} | {v["M3"]["brier"]-v["M1"]["brier"]:+.5f} |')
    delta=s['run_macro_brier_delta_vs_M1']['M3']
    lines+=['',f'以run为重采样单位，M3−M1的Brier宏平均差为 {delta["mean"]:+.5f}，探索性bootstrap 95%区间 [{delta["ci95"][0]:+.5f}, {delta["ci95"][1]:+.5f}]；{delta["runs_better"]}/15个run改善。每任务仅3个run，不将候选边数视为独立重复，不据此声称已识别因果效应。','',
        '![各任务逐run的Brier增量](E1-A预测增量.png)', '', '图中每点代表一个run，橙色短线为该任务三个run的均值；纵轴小于零表示M3优于M1。', '', '## 4. 缺失画像与首次开发检查','',
        '| 测试子集 | 尝试数 | M1 Brier | M2 Brier | M3 Brier |','| --- | ---: | ---: | ---: | ---: |']
    for label,r in [('全部',s),('行为画像与历史邻居均可用',complete),('父节点第一次被Refine开发',new)]:
        lines.append(f'| {label} | {r["overall"]["M1"]["n"]} | {r["overall"]["M1"]["brier"]:.5f} | {r["overall"]["M2"]["brier"]:.5f} | {r["overall"]["M3"]["brier"]:.5f} |')
    lines+=['','首次开发子集用于检查新节点泛化；同父历史消融用于区分跨节点共享与重复开发记忆。按backend、run的指标、校准分组、top组正部收益、超当前best及退化率均保留在机器可读汇总中。backend分组同时受任务构成影响，不用于推断后端优劣。','',
        '## 5. 辅助目标','', '除严格改善外，另用相同时间回放预测灾难性退化（有效候选相对退步超过10%）与超当前best，并用固定Ridge(alpha=1)预测相对正部增益。辅助目标按原设计补充实现，基础对照跑通后单列检查，不参与调参或主筛选门槛。','', *auxiliary_table(out), '', '## 6. 成本、限制与下一步','',
        f'本轮新增生成LLM调用为0，新增正式搜索评价账本调用为0；实际执行了 {sum(c["panels"] for c in costs.values())} 次行为面板画像，每个成功面板包含4个实例，累计画像耗时 {sum(c["seconds"] for c in costs.values())/3600:.2f} worker小时（并行累计，不等于墙钟时间）。另有20个一致性案例的重复画像与评价器核验，以及10个锚点的原始/替代随机流双画像。embedding计算约 {emb["seconds"]:.1f} 秒，全部在CPU；这些重放和表示成本不隐去。','',
        'embedding采用固定版本all-MiniLM-L6-v2，完整代码/摘要分块池化。它是便宜的通用表示对照，并非最强代码embedding，因此不能把M2结果外推为所有表示方法的上限。主预测器未作超参搜索；弱信号可能来自小样本、时变响应、行为测量噪声或模型欠拟合，不能把一次不通过门槛解释为所有形式的locality均不存在。','',
        '本轮没有单独验证多步停滞窗口或切换动作收益，不能将预测未改善直接称为饱和检测成功。本实验只评估旧策略实际访问状态上的预测能力；top组lift不是离策略预算收益。未建立“相近状态的反馈可无偏迁移”，也未测量新选父策略的终局效果。是否进入E1-B应同时参考跨任务方向、画像稳定性、首次开发子集和基线比较，而非只看单一总体相关系数。','',
        '## 7. 复现与产物','',
        '[运行入口与实际协议](../../../../experiments/traceaad_refine_e1/README.md)。本地原始工件位于 `experiments/traceaad_refine_e1/raw/refine_e1_20260907/`：`snapshot.json`、`experiment_config.json`、`validation.json`、`profiles/`、`stability.json`、`embedding_metadata.json`、`replay_predictions.jsonl`、`summary*.json`、`costs.json`。','']
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'E1-A-结果.md').write_text('\n'.join(lines))
    dump(DOC/'e1a_summary.json',dict(config=json.loads((out/'experiment_config.json').read_text()),environment=json.loads((out/'environment.json').read_text()),auxiliary=json.loads((out/'auxiliary_summary.json').read_text()),snapshot=snap,stability=stability,summary=s,new_parents=new,complete_profiles=complete,embedding=emb,costs=dict(costs),seed_sensitivity=[{k:v for k,v in r.items() if k not in ["original","alternate"]} for r in seeds],screening_gate=gate))
    # Standalone research figure: each dot is one independent run.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for i,t in enumerate(LABELS):
        rr=[r['run_name'] for r in snap['runs'] if r['task']==t]
        d=[s['by']['run'][r]['M3']['brier']-s['by']['run'][r]['M1']['brier'] for r in rr]
        ax.scatter(np.arange(len(d))*.08+i-.08,d,color='#2266aa',s=40)
        ax.plot([i-.23,i+.23],[np.mean(d)]*2,color='#cc6600',lw=3)
    ax.axhline(0,color='gray',ls='--',lw=1);ax.set_xticks(range(5),LABELS.values());ax.set_ylabel('Brier(M3) - Brier(M1); lower is better')
    ax.set_title('E1-A: behavior-neighborhood prediction vs fitness/count/stage');fig.tight_layout()
    fig.savefig(DOC/'e1a_gain.png',dpi=180);plt.close(fig)
    print(decision)

if __name__=='__main__':main()
