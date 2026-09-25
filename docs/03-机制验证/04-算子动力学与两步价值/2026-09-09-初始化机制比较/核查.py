"""从仓库根目录运行；只读原始运行档案，结果写入 /tmp/init_comparison.json。"""
import json,math,statistics,csv
from pathlib import Path
from collections import defaultdict
TASKS=['tsp_construct','cvrp_aco','op_aco','online_bin_packing','vrptw_construct']
METHODS=['traceaad_v9_7','traceaad_v9_14','traceaad_v9_19']+[f'traceaad_v10_{i}' for i in range(1,9)]
rows=[]
for method in METHODS:
 for task in TASKS:
  for d in sorted(Path('experiments',method,'results',task).glob('*rep[123]')):
   if method=='traceaad_v9_19' and 'fixed_20260829' not in d.name:continue
   if method=='traceaad_v10_8' and not d.name.startswith('20260909_v108_formal_'):continue
   cfg=d/'run_config.json'
   if not cfg.exists():continue
   c=json.loads(cfg.read_text());p=c.get('method_params',{})
   if p.get('budget',1000)!=1000:continue
   roots=[]; bootstrap=[]
   if (d/'tree_state.json').exists():
    ns=json.loads((d/'tree_state.json').read_text())['nodes'];roots=[n for n in ns if n.get('parent_id') is None]
   elif method=='traceaad_v9_7':
    f=d/'artifacts/candidates.jsonl'
    if not f.exists():continue
    ev=0
    with f.open() as h:
     for l in h:
      x=json.loads(l);ev+=bool(x.get('evaluator_called'))
      if x.get('child_id') is not None and x.get('child_fitness') is not None:
       n=dict(id=x['child_id'],fitness=x['child_fitness'],evaluation_id=ev,code=x['program'],parent_id=x.get('anchor_id'))
       if x.get('anchor_id') is None:roots.append(n)
       if x.get('stage')=='bootstrap':bootstrap.append(n)
   elif (d/'checkpoints/latest.json').exists():
    data=json.loads((d/'checkpoints/latest.json').read_text());ns=data['tree']['algorithms'];roots=[n for n in ns if n.get('parent_id')==0 and n.get('fitness') is not None]
    if (d/'evaluations.csv').exists():
     evmap={int(x['child_id']):int(x.get('eval_count',x.get('slot',0))) for x in csv.DictReader((d/'evaluations.csv').open()) if x.get('child_id') not in ('',None,'None')}
     roots=[{**n,'evaluation_id':evmap.get(n['id'])} for n in roots]
   if not roots:continue
   roots=[n for n in roots if n.get('fitness') is not None and math.isfinite(n['fitness'])]
   label=method.replace('traceaad_','').replace('_','.')
   if method=='traceaad_v10_7':label='v10.7R' if '_v107r_' in d.name else 'v10.7'
   r=dict(method=label,task=task,path=str(d),repeat=c.get('repeat'),n_roots=len(roots),best=max(n['fitness'] for n in roots),mean=statistics.mean(n['fitness'] for n in roots),scores=[n['fitness'] for n in roots],last_root_eval=max(n.get('evaluation_id',0) or 0 for n in roots),bootstrap_scores=[n['fitness'] for n in bootstrap],root_ids=[n['id'] for n in roots])
   rows.append(r)
Path('/tmp/init_comparison.json').write_text(json.dumps(rows,indent=2))
groups=defaultdict(list)
for r in rows:groups[r['method'],r['task']].append(r)
for method in dict.fromkeys(r['method'] for r in rows):
 print(method,[(t,len(groups[method,t]),round(statistics.mean(r['best'] for r in groups[method,t]),4) if groups[method,t] else None) for t in TASKS])
for t in TASKS:
 pairs=[(statistics.mean(r['best'] for r in rs),m,len(rs)) for (m,task),rs in groups.items() if task==t and len(rs)==3]
 print('BEST_INITIAL',t,sorted(pairs,reverse=True)[:4])
