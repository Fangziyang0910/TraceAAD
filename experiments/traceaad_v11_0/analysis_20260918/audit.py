"""Read-only audit of five batches; event-time ranks and 100-evaluation follow-up."""
import ast, bisect, hashlib, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
TASKS=['tsp_construct','cvrp_aco','op_aco','online_bin_packing','vrptw_construct']
BATCHES={'generic':('traceaad_v10_11','20260915_v1011_generic'), 'no_traj':('traceaad_v10_11','20260915_v1011_no_traj_idea'), 'idea_code':('traceaad_v10_11','20260915_v1011_idea_code'), 'rand_ctx':('traceaad_v10_11','20260916_v1011_rand_ctx'), 'v11':('traceaad_v11_0','20260917')}
def mean(v): return statistics.mean(v) if v else None
def key(code): return hashlib.sha256(ast.dump(ast.parse(code),include_attributes=False).encode()).hexdigest()
def audit(run):
 nodes=[json.loads(l) for l in (run/'nodes.jsonl').read_text().splitlines()]; nd={n['id']:n for n in nodes}; nk={n['id']:key(n['code']) for n in nodes}
 events=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]; ev=[e for e in events if e.get('candidate_id') is not None]
 rank_scores=[]; born={}; selected=Counter(); sel_times=defaultdict(list); normal=[]; refs=[]; root={}; depths={}; lineage_counts=Counter(); op=defaultdict(Counter); curves={}
 for n in nodes:
  root[n['id']]=n['id'] if n['parent_id'] is None else root[n['parent_id']]
  depths[n['id']]=0 if n['parent_id'] is None else depths[n['parent_id']]+1
 for e in ev:
  pid=e.get('parent_id'); is_norm=pid is not None and not e.get('repair_of')
  at=e.get('evaluation_id') or e.get('budget_used',0)
  if is_norm:
   f=nd[pid]['fitness']; low=bisect.bisect_left(rank_scores,f); high=bisect.bisect_right(rank_scores,f)
   q=(low+(high-low-1)/2)/(len(rank_scores)-1) if len(rank_scores)>1 else .5
   s=e.get('selection',{}); selected[nk[pid]]+=1; sel_times[nk[pid]].append(at); lineage_counts[root[pid]]+=1
   normal.append({'q':q,'attempts_before':s.get('attempts',s.get('parent_count_before')),'op':e['operator'],'at':at,'key':nk[pid],'parent_gap':(e['best_before']-f)/max(abs(e['best_before']),1e-10),'parent_age':at-nd[pid]['evaluation_id']})
   o=op[e['operator']];o['normal']+=1;o['ok']+=e['status']=='ok';o['frontier']+=bool(e.get('frontier_improved'));o['parent_improved']+=bool(e.get('parent_improved'))
   for rid in e.get('reference_ids',[]):
    rf=nd[rid]['fitness']; better=len(rank_scores)-bisect.bisect_right(rank_scores,rf)
    refs.append({'rank':better+1,'pool':len(rank_scores),'op':e['operator'],'at':at})
  nid=e.get('node_id')
  if nid is not None:
   n=nd[nid]; k=nk[nid];f=n['fitness'];N=len(rank_scores)
   birthq=(bisect.bisect_left(rank_scores,f)+(bisect.bisect_right(rank_scores,f)-bisect.bisect_left(rank_scores,f))/2)/N if N else 1
   if k not in born: born[k]={'q':birthq,'eval':n['evaluation_id'],'op':n['operator'],'frontier':bool(e.get('frontier_improved')),'id':nid}
   bisect.insort(rank_scores,f)
 for b in (100,250,500,750,1000):
  fs=[n['fitness'] for n in nodes if n['evaluation_id']<=b];curves[str(b)]=max(fs) if fs else None
 follow={}
 for group,pred in {'all':lambda n:True,'pivot':lambda n:n['op']=='Pivot','q_lt_075':lambda n:n['q']<.75,'q_ge_095':lambda n:n['q']>=.95}.items():
  items=[(k,n) for k,n in born.items() if n['eval']<=900 and n['op']!='Init' and pred(n)]
  cnt=sum(any(n['eval']<t<=n['eval']+100 for t in sel_times[k]) for k,n in items)
  follow[group]={'eligible':len(items),'developed_100':cnt}
 best=max(nodes,key=lambda n:n['fitness']); chain=[];n=best
 while n is not None: chain.append(n['id']); n=nd.get(n['parent_id'])
 return {'run':str(run.relative_to(ROOT)),'budget':max(e.get('evaluation_id') or 0 for e in ev),'curves':curves,'status':dict(Counter(e['status'] for e in ev)), 'normal_n':len(normal),'q_mean':mean([x['q'] for x in normal]),'q_lt_075':sum(x['q']<.75 for x in normal),'fresh_fraction':mean([x['attempts_before']==0 for x in normal]),'parent_gap_mean':mean([x['parent_gap'] for x in normal]),'top_root_share':max(lineage_counts.values())/len(normal),'unique_parents':len(selected),'parents_at_least5':sum(v>=5 for v in selected.values()),'ref_rank_median':statistics.median(x['rank'] for x in refs) if refs else None,'ref_outside32_fraction':mean([x['rank']>32 for x in refs]),'ref_count':len(refs),'ops':dict(op),'followup':follow,'best':{'id':best['id'],'eval':best['evaluation_id'],'fitness':best['fitness'],'operator':best['operator'],'depth':depths[best['id']],'chain':chain},'depth_mean':mean(list(depths.values())),'valid_codes':len(born),'valid_nodes':len(nodes)}
def main():
 results={}
 for batch,(version,prefix) in BATCHES.items():
  results[batch]={}
  for task in TASKS:
   runs=sorted((ROOT/'experiments'/version/'results'/task).glob(prefix+'*'))
   results[batch][task]=[audit(r) for r in runs if (r/'nodes.jsonl').exists()]
  print('finished',batch,flush=True)
 (OUT/'audit.json').write_text(json.dumps(results,indent=2)+'\n')
 for task in TASKS:
  print('\nTASK',task)
  for batch in BATCHES:
   rows=results[batch][task];print(batch,'n',len(rows),'budget',[r['budget'] for r in rows],'q',round(mean([r['q_mean'] for r in rows]),3),'fresh',round(mean([r['fresh_fraction'] for r in rows]),3),'parents',round(mean([r['unique_parents'] for r in rows]),1),'toproot',round(mean([r['top_root_share'] for r in rows]),3),'ref>32',mean([r['ref_outside32_fraction'] for r in rows if r['ref_outside32_fraction'] is not None]))
if __name__=='__main__':main()
