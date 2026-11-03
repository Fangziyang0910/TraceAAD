"""Post-hoc full-training output checks for three diagnostic examples, not prevalence."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys


def check(row,limit=None):
    import numpy as np
    from traceaad_base.cli import load_task
    from output_probe import equal
    task=load_task(name=row['task']);e=task.evaluate_program.__self__;outputs=[]
    arc_max=[];minimum_ranges=[]
    for code in (row['parent_code'],row['child_code']):
        collected=[]
        for i in range(min(limit or e.n_instance,e.n_instance)):
            random.seed(730241+i);np.random.seed(730241+i);ns={};exec(code,ns);fn=ns[task.entrypoint]
            if row['task']=='op_aco':
                from benchmarks.op_aco.evaluation import gen_prizes,gen_distance_matrix
                coords=e._datasets[i];d=gen_distance_matrix(coords)
                eligible=np.ones(d.shape,bool);np.fill_diagonal(eligible,False);eligible[0,:]=False;eligible[:,0]=False
                arc_max.append(float(np.max((d+d[:,0][None,:])[eligible])))
                collected.append(fn(gen_prizes(coords),d.copy(),float(e.max_len)))
            elif row['task']=='tsp_construct':
                coords,dist=e._datasets[i];current=0;visited=np.zeros(len(coords),bool);visited[0]=True;route=[0]
                for step in range(len(coords)-2):
                    order=np.argsort(dist[current]);unvisited=order[~visited[order]]
                    if row['pair_id']=='tsp_construct_s1_A21':
                        a=dist[current,unvisited];b=dist[unvisited,0];beta=(len(unvisited)/len(coords))**2
                        minimum_ranges.extend([float(np.ptp(a+beta*b)),float(np.ptp(np.sqrt(a*(1+b))))])
                    nxt=fn(current,0,unvisited.copy(),dist.copy());assert nxt in unvisited
                    current=int(nxt);visited[current]=True;route.append(current)
                route.append(int(np.flatnonzero(~visited)[0]));collected.append(route)
            else:
                d=e._instances[i];collected.append(fn(d['processing_times'].copy(),d['machine_order'].copy()))
        outputs.append(copy.deepcopy(collected))
    same=[equal(a,b) for a,b in zip(*outputs)]
    return {'pair_id':row['pair_id'],'task':row['task'],'instances':len(same),'per_instance_equal':same,
            'all_equal':all(same),'max_pair_plus_return_distance':max(arc_max) if arc_max else None,
            'maxlen':e.max_len if arc_max else None,'min_tsp_score_range':min(minimum_ranges) if minimum_ranges else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-root',required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--tsp-sample',action='store_true');a=p.parse_args();sys.path.insert(0,a.base_root)
    from traceaad_base.runtime import isolated_call
    from experiments.traceaad_base.scheduling import ScheduledEvaluator
    from traceaad_base.cli import load_task
    rows=json.loads((a.data/'sample.locked.json').read_text())['sampled_pairs'];results=[]
    pairs=[r['pair_id'] for r in rows if r['task']=='tsp_construct'] if a.tsp_sample else ('op_aco_s0_A10','tsp_construct_s1_A21','jssp_construct_s0_A117')
    for pair in pairs:
        row=next(x for x in rows if x['pair_id']==pair)
        scheduler=ScheduledEvaluator(load_task(name=row['task']),'/tmp/traceaad-1003/scheduler.sock','trajectory-case:'+pair)
        with scheduler.evaluation_context():result,seconds=isolated_call(check,(row,2 if a.tsp_sample else None),60.)
        result['seconds']=seconds;results.append(result);print(json.dumps(result),flush=True)
    (a.data/('tsp_routes.json' if a.tsp_sample else 'case_audit.json')).write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
