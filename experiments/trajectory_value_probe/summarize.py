"""Rebuild descriptive tables, cost checks and an inline comparison chart."""
from collections import Counter,defaultdict
import hashlib
import html
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[2]/'experiments_result/analysis/trajectory_feature_probe_20261103'
NAMES={'cvrp_aco':'CVRP','fssp_gls':'FSSP','graph_colouring':'图着色','jssp_construct':'JSSP','op_aco':'OP','tsp_construct':'TSP'}
OPS=('Refine','Explore','Crossover')


def read(p):return json.loads(p.read_text())


def main():
    folder=OUT/'prevalence';plan=read(folder/'sample.locked.json');groups=defaultdict(Counter);rows=[]
    routes={x['pair_id']:x for x in read(folder/'tsp_routes.json')}
    for row in plan['sampled_pairs']:
        pid=row['pair_id'];original=read(folder/'results'/f'{pid}.json');final=original
        retry=folder/'results-output-only'/f'{pid}.json'
        if retry.exists():final=read(retry)
        x={k:v for k,v in row.items() if not k.endswith('_code')}
        x.update(output_status=final['probe_status'],same_output=final.get('all_outputs_equal'),
                 trace_status=original['probe_status'],trace_zero=original.get('changed_statements_unexecuted'),
                 changed_statement_count=len(original.get('changed_statement_lines',[])),
                 executed_statement_count=len(original.get('executed_changed_lines',[])))
        code=row['parent_code']+'\n'+row['child_code']
        x['clock_dependent_source']=any(t in code for t in ('time.time','perf_counter','monotonic','process_time'))
        x['unseeded_rng_source']=any(t in code for t in ('default_rng()','default_rng(None)','Random()','Random(None)'))
        x['initial_state_probe_equal']=original.get('all_outputs_equal')
        if pid in routes:
            x.update(same_output=routes[pid]['all_equal'],output_status='success',comparison='two complete TSP routes')
        else:x['comparison']='fixed function outputs'
        c=groups[row['task'],row['operator']];c['sampled']+=1
        c['output_observed']+=x['output_status']=='success';c['output_same']+=x['same_output'] is True
        c['trace_observed']+=x['trace_status']=='success' and x['changed_statement_count']>0
        c['trace_zero']+=x['trace_zero'] is True
        c['same_fitness_sampled']+=x['same_fitness'];c['same_fitness_observed']+=x['same_fitness'] and x['output_status']=='success'
        c['same_fitness_same_output']+=x['same_fitness'] and x['same_output'] is True
        c['same_output_different_fitness']+=not x['same_fitness'] and x['same_output'] is True
        c['clock_dependent_source']+=x['clock_dependent_source'];c['unseeded_rng_source']+=x['unseeded_rng_source']
        rows.append(x)
    summary=[{'task':k[0],'operator':k[1],**c} for k,c in sorted(groups.items())]
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (folder/'paired_measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    lock=read(OUT/'candidates.locked.json');pilot=[];counts=Counter();models=defaultdict(Counter)
    for job in lock['jobs']:
        root=OUT/'pilot/jobs'/job['job_id'];x=read(root/'result.json');state=read(root/'native/state.json')
        calls=[read(p) for p in sorted((root/'native/calls').glob('*.json'))]
        assert len(calls)==x['requests']==job['request_budget']==6
        assert state['evaluations_used']==x['evaluations']<=6
        assert [c['requests'] for c in x['checkpoints']]==list(range(1,7))
        counts['requests']+=len(calls);counts['evaluations']+=x['evaluations'];counts['rollouts']+=1
        counts['improved_start']+=x['best_fitness']<x['starting_fitness']-1e-9
        counts['global_breakthrough']+=x['best_fitness']<x['frozen_incumbent']-1e-9
        counts['gained_after_request_2']+=x['best_fitness']<x['checkpoints'][1]['best_fitness']-1e-9
        for c in calls:models[x['candidate_key']][c.get('model','unknown')]+=1;counts['repair_requests']+=c['stage']=='repair'
        ev=[e for a in state['attempts'] for e in a['evaluations']]
        for e in ev:counts['evaluation_'+e['status']]+=1
        pilot.append({'candidate':x['candidate_key'],'role':x['role'],'task':x['task'],**job,
                      'start':x['starting_fitness'],'incumbent':x['frozen_incumbent'],
                      'after_2':x['checkpoints'][1]['best_fitness'],'after_6':x['best_fitness'],
                      'elapsed_seconds':x['elapsed_seconds']})
    counts['starting_reevaluations']=len(read(OUT/'pilot/preflight.json')['results'])
    (OUT/'pilot/summary.json').write_text(json.dumps({'costs':counts,'models':models,'rollouts':pilot},ensure_ascii=False,indent=2))
    # Export the examples for code review without opening a large JSON blob.
    examples=OUT/'examples';examples.mkdir(exist_ok=True)
    for row in plan['sampled_pairs']:
        if row['pair_id'] not in ('op_aco_s0_A10','tsp_construct_s1_A21','jssp_construct_s0_A117','fssp_gls_s1_A180','graph_colouring_s0_A19'):continue
        d=examples/row['pair_id'];d.mkdir(exist_ok=True)
        for which in ('parent','child'):(d/f'{which}.py').write_text(row[which+'_code'])
        (d/'measurement.json').write_text(json.dumps(next(x for x in rows if x['pair_id']==row['pair_id']),indent=2))
    for candidate in lock['candidates']:
        prefix=read(OUT/candidate['prefix_file']);program=next(p for p in prefix['archive'] if p['id']==candidate['node_id'])
        (examples/(candidate['candidate_key']+'.py')).write_text(program['code'])
    full=read(folder/'census.full.json');table=['|任务|算子|未改善子代|源码相同|静态条件下改动不生效|输出相同/可比较|跟踪零覆盖/可比较|','|---|---|---:|---:|---:|---:|---:|']
    for x in full:
        c=groups[x['task'],x['operator']]
        table.append(f"|{NAMES[x['task']]}|{x['operator']}|{x['nonimproving']}|{x['nonimproving_same_ast']}|{x['nonimproving_known_inactive_only']}|{c['output_same']}/{c['output_observed']}|{c['trace_zero']}/{c['trace_observed']}|")
    (folder/'table.md').write_text('\n'.join(table)+'\n')
    table=['|起点|首步|重复|初始|第2请求最好|第6请求最好|','|---|---|---:|---:|---:|---:|']
    for x in sorted(pilot,key=lambda r:(r['candidate'],r['first_operator'],r['repeat'])):
        table.append(f"|{x['candidate']}|{x['first_operator']}|{x['repeat']}|{x['start']}|{x['after_2']}|{x['after_6']}|")
    (OUT/'pilot/table.md').write_text('\n'.join(table)+'\n')
    draw(summary,pilot)
    sources={str(p.relative_to(OUT.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (OUT/'analysis_sources.json').write_text(json.dumps(sources,indent=2))
    print(json.dumps({'pilot':counts,'output_same':sum(x['output_same'] for x in summary),'output_observed':sum(x['output_observed'] for x in summary)},ensure_ascii=False))


def draw(summary,pilot):
    g={(x['task'],x['operator']):x for x in summary}
    parts=['<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:var(--font-sans);color:var(--foreground)}h3{font-size:16px;margin:8px 0}p{font-size:12px;color:var(--muted-foreground);line-height:1.5}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px 3px;border-bottom:1px solid var(--border);text-align:left}.bar{height:5px;background:var(--chart-1);border-radius:3px;margin-top:4px}svg{width:100%;height:200px}.key{font-size:11px}</style></head><body><h3>未改善子代中，固定输入上输出相同</h3><p>每格随机抽8对；每任务取两个训练实例。TSP比较完整路线。这里只测观察到的输出重复。</p><table><tr><th>任务</th><th>精炼</th><th>探索</th><th>融合</th></tr>']
    for task in NAMES:
        parts.append('<tr><td>'+NAMES[task]+'</td>')
        for op in OPS:
            x=g[task,op];num=x['output_same'];den=x['output_observed']
            parts.append(f'<td>{num}/{den}（{100*num/den:.0f}%）<div class="bar" style="width:{100*num/den:.1f}%"></div></td>')
        parts.append('</tr>')
    parts.append('</table><h3>等预算接续：第2与第6次模型请求后的最好分数</h3><p>每个点是一条续跑；实线为精炼首步，虚线为探索首步。低分更好。颜色区分目标起点与对照。</p>')
    for task in ('graph_colouring','jssp_construct'):
        rs=[r for r in pilot if r['task']==task];values=[r[k] for r in rs for k in ('after_2','after_6')]+[rs[0]['incumbent']]
        lo=min(values);hi=max(values);pad=(hi-lo)*.08 or 1;lo-=pad;hi+=pad
        y=lambda v:165-135*(v-lo)/(hi-lo)
        parts.append(f'<svg viewBox="0 0 620 200" role="img" aria-label="{NAMES[task]}续跑结果"><text x="0" y="16" fill="var(--foreground)" font-size="14">{NAMES[task]}</text>')
        for val,label in ((min(values),str(round(min(values),2))),(max(values),str(round(max(values),2)))):
            parts.append(f'<line x1="60" x2="570" y1="{y(val)}" y2="{y(val)}" stroke="var(--border)"/><text x="4" y="{y(val)+4}" font-size="10" fill="var(--foreground)">{label}</text>')
        for i,r in enumerate(rs):
            col='var(--chart-1)' if r['role']=='feature_target' else 'var(--chart-2)';dash='' if r['first_operator']=='Refine' else 'stroke-dasharray="4 3"'
            a=150+(i-3.5)*5;b=460+(i-3.5)*5
            parts.append(f'<g><title>{html.escape(r["job_id"])}: {r["after_2"]} → {r["after_6"]}</title><line x1="{a}" x2="{b}" y1="{y(r["after_2"])}" y2="{y(r["after_6"])}" stroke="{col}" {dash}/><circle cx="{a}" cy="{y(r["after_2"])}" r="3" fill="{col}"/><circle cx="{b}" cy="{y(r["after_6"])}" r="3" fill="{col}"/></g>')
        parts.append('<text x="130" y="190" font-size="11" fill="var(--foreground)">2次请求</text><text x="440" y="190" font-size="11" fill="var(--foreground)">6次请求</text></svg>')
    parts.append('<p class="key"><span style="color:var(--chart-1)">● 特征目标起点</span>　<span style="color:var(--chart-2)">● 相近适应度对照</span>。每起点每算子仅2条续跑；图着色两起点执行部分相同；JSSP起点分数未完全匹配。此图不估计完整选择策略的优势。</p></body></html>')
    (OUT/'chart.html').write_text(''.join(parts))


if __name__=='__main__':main()
