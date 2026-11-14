"""Freeze committed local run data for the manuscript; never evaluate programs."""
from pathlib import Path
import csv, hashlib, json, statistics, datetime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/evidence'
BATCH='20261114_103407_evotrace_v2_repaired_co4'

def records(path):
    raw=path.read_bytes()
    rows=[]
    for line in raw.splitlines():
        if line.strip(): rows.append(json.loads(line))
    return raw,rows

runs=[]
for p in sorted((ROOT/'experiments_result/evotrace_v2_repaired').glob('*/'+BATCH+'*/native/attempts.jsonl')):
    raw,rs=records(p)
    config=json.loads((p.parent.parent/'run_config.json').read_text())
    evs=[ev for r in rs for ev in r['attempt'].get('evaluations',[])]
    trees=[r['tree_after'] for r in rs if 'tree_after' in r]
    updates=[]; reflections=[]
    for r in rs:
        c=r['attempt'].get('record',{}).get('consolidation')
        if not c: continue
        expected=set(c['triggers'][0]['removed_ids'])
        assigned=[i for a in c.get('path_contractions',[]) for i in a['removed_ids']+a.get('side_ids',a.get('side_removed_ids',[]))]
        assigned += [i for a in c.get('branch_consolidations',[]) for i in a['removed_ids']]
        if set(assigned)!=expected:
            # Versions serialize side attempts as branch_ids.
            assigned=[i for a in c.get('path_contractions',[]) for i in a['removed_ids']+a.get('branch_ids',[])]
            assigned += [i for a in c.get('branch_consolidations',[]) for i in a['removed_ids']]
        assert set(assigned)==expected and len(assigned)==len(set(assigned)), (p,c.get('path_contractions'))
        updates.append(c); reflections.extend(c.get('reflection_calls',[]))
    size=0; attached=0; elapsed=0; tree_series=[]
    for row in rs:
        evaluations=row['attempt'].get('evaluations',[])
        elapsed+=len(evaluations)
        if evaluations and row.get('program') is not None:
            size+=1; attached+=1
        change=row['attempt'].get('record',{}).get('consolidation')
        if change:size-=len(change['triggers'][0]['removed_ids'])
        if 'tree_after' in row:assert size==len(row['tree_after']['nodes'])
        if evaluations:tree_series.append({'evaluation':elapsed,'nodes':size})
    safeconfig={k:config[k] for k in ['task','seed','batch','method_params','task_eval','evaluation_execution','llm']}
    run={'task':config['task'],'seed':config['seed'],'source':str(p.relative_to(ROOT)),
         'sha256':hashlib.sha256(raw).hexdigest(),'attempts':len(rs),'evaluations':len(evs),
         'last_finished_at':rs[-1]['attempt']['finished_at'],'retained_nodes':size,'attached_nodes':attached,
         'updates':len(updates),'path_regions':sum(len(c.get('path_contractions',[])) for c in updates),
         'side_regions':sum(len(c.get('branch_consolidations',[])) for c in updates),
         'removed_nodes':sum(len(c['triggers'][0]['removed_ids']) for c in updates),
         'reflection_calls':len(reflections),'fallbacks':sum(x.get('used_fallback',False) for x in reflections),
         'reflection_input_tokens':sum(x.get('usage',{}).get('prompt_tokens',0) or 0 for x in reflections),
         'reflection_output_tokens':sum(x.get('usage',{}).get('completion_tokens',0) or 0 for x in reflections),
         'region_sizes':[len(a['removed_ids'])+len(a.get('side_removed_ids',a.get('side_ids',a.get('branch_ids',[])))) for c in updates for a in c.get('path_contractions',[])]+[len(a['removed_ids']) for c in updates for a in c.get('branch_consolidations',[])],
         'latest_best':min(ev['evaluation']['fitness'] for ev in evs if ev['evaluation'].get('fitness') is not None),
         'config':safeconfig,'series':[]}
    best=None
    for n,ev in enumerate(evs,1):
        f=ev['evaluation'].get('fitness')
        if f is not None:best=f if best is None else min(best,f)
        run['series'].append({'evaluation':n,'fitness':f,'best':best,'status':ev['status']})
    run['tree_series']=tree_series
    runs.append(run)
assert len(runs)==12,len(runs)
common=(min(r['evaluations'] for r in runs)//10)*10
for r in runs:r['common_best']=r['series'][common-1]['best']
agg=[]
for task in ['tsp_construct','cvrp_aco','fssp_gls','op_aco']:
    sub=[r for r in runs if r['task']==task]; scores=[r['common_best'] for r in sub]
    agg.append({'task':task,'budget':common,'scores':scores,'mean':statistics.mean(scores),'sd':statistics.stdev(scores),
                'evaluations':[r['evaluations'] for r in sub], 'retained':[r['retained_nodes'] for r in sub],
                'path_regions':sum(r['path_regions'] for r in sub),'side_regions':sum(r['side_regions'] for r in sub),
                'removed_nodes':sum(r['removed_nodes'] for r in sub)})
result={'captured_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'batch':BATCH,
        'common_budget':common,'scope':'Committed local prefix; ongoing search, training fitness only; three seeds per task.',
        'runs':runs,'aggregates':agg}
OUT.mkdir(exist_ok=True)
(OUT/'snapshot.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (OUT/'training_scores.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['task','seed','common_budget','best_training_fitness','observed_evaluations','retained_nodes'])
    for r in runs:w.writerow([r['task'],r['seed'],common,r['common_best'],r['evaluations'],r['retained_nodes']])
print(json.dumps({'captured_at':result['captured_at'],'budget':common,'aggregates':agg},ensure_ascii=False,indent=2))
