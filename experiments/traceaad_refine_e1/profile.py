"""Training-probe profiles, validation and per-run distance matrices for E1-A."""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import profile_core as core
from .prepare import DEFAULT, dump


def candidate(node):
    return {'id':node['id'], 'code':node['code']}


def worker(job):
    task, node = job
    c = candidate(node)
    result = {'node_id':c['id'], 'panels':{}}
    for panel in ['A','B']:
        core._init_worker(task, panel, core.DEFAULT_TRAJECTORY_POINTS[task], core.DEFAULT_TIMEOUT_SECONDS[task])
        result['panels'][panel] = core._profile_candidate(c)
    return result


def inputs(out):
    manifest = json.loads((out/'snapshot.json').read_text())
    result = []
    for run in manifest['runs']:
        folder = out/'snapshot'/run['run_name']
        nodes = json.loads((folder/'tree_state.json').read_text())['nodes']
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        selected = {e['parent_id'] for e in events if e['requested_operator']==e['operator']=='Refine'}
        result.append((run, [n for n in nodes if n['id'] in selected]))
    return result


def validate(out):
    """Two earliest selected parents per task, replayed twice and against evaluator."""
    records = []
    seen = set()
    for run,nodes in inputs(out):
        task=run['task']
        if task in seen: continue
        seen.add(task)
        for node in nodes[:2]:
            for panel in ['A','B']:
                core._init_worker(task,panel,core.DEFAULT_TRAJECTORY_POINTS[task],core.DEFAULT_TIMEOUT_SECONDS[task])
                c=candidate(node)
                a=core._profile_candidate(c);b=core._profile_candidate(c)
                assert a['ok'] and b['ok'], (task,a,b)
                assert a['trajectories']==b['trajectories'] and a['probe_score']==b['probe_score']
                namespace={'np':np};core._reset_program_randomness(-1);exec(node['code'],namespace)
                fn=namespace['priority' if task=='online_bin_packing' else 'heuristics' if task in ['op_aco','cvrp_aco'] else 'select_next_node']
                data=core._GLOBAL_DATA
                scores=[]
                if task=='online_bin_packing':
                    from benchmarks.online_bin_packing.evaluation import OBPEvaluation
                    evaluator=OBPEvaluation()
                else: evaluator=data['evaluator']
                original_seed=getattr(evaluator,'aco_seed',None)
                for i,instance in enumerate(data['instances']):
                    core._reset_program_randomness(i)
                    if task=='online_bin_packing':
                        evaluator._datasets={'probe':instance}
                    else:
                        evaluator._datasets=[instance];evaluator.n_instance=1
                    if original_seed is not None:
                        evaluator.aco_seed=original_seed+data['instance_indices'][i]
                    scores.append(evaluator.evaluate(fn))
                official=float(np.mean(scores))
                assert np.isclose(official,a['probe_score'],rtol=1e-10,atol=1e-10),(task,official,a['probe_score'])
                # Optimized distance must agree with the literal DTW implementation.
                matrix=core.compute_distance_matrix([a,b],prefix_mode=task in core.PREFIX_TASKS)
                assert np.max(np.abs(matrix))==0
                records.append(dict(task=task,node_id=node['id'],panel=panel,
                    official_score=official,profile_score=a['probe_score'],deterministic=True,
                    seconds=a['elapsed_seconds'],probe_metadata=data['probe_metadata']))
                print('validated',task,node['id'],panel,round(a['elapsed_seconds'],2),flush=True)
    dump(out/'validation.json',records)


def profile(out,workers):
    assert (out/'validation.json').exists(), 'Run validation first'
    jobs={}
    for run,nodes in inputs(out):
        for node in nodes:
            key=(run['task'],run['run_name'],str(node['id']))
            path=out/'profiles'/key[0]/key[1]/(key[2]+'.json')
            if not path.exists():jobs[key]=(key[0],node)
    print('profile jobs',len(jobs),'workers',workers,flush=True)
    started=time.time()
    # Bounded CPU workers; no GPU or generation endpoint used.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,job):key for key,job in jobs.items()}
        from concurrent.futures import as_completed
        for i,future in enumerate(as_completed(futures)):
            key=futures[future];result=future.result()
            dump(out/'profiles'/key[0]/key[1]/(key[2]+'.json'),result)
            if i%25==0:print('profiles',i+1,'/',len(jobs),'seconds',round(time.time()-started),flush=True)
    matrices(out)


def matrices(out, ready_only=False):
    stability=[]
    for run,nodes in inputs(out):
        profile_dir=out/'profiles'/run['task']/run['run_name']
        if ready_only and not all((profile_dir/(str(n['id'])+'.json')).exists() for n in nodes):
            continue
        panels={'A':[],'B':[]};valid=[]
        for n in nodes:
            p=json.loads((profile_dir/(str(n['id'])+'.json')).read_text())
            if all(p['panels'][s]['ok'] for s in panels):
                valid.append(n['id'])
                for s in panels:panels[s].append(p['panels'][s])
        folder=out/'distances'/run['run_name'];folder.mkdir(parents=True,exist_ok=True)
        dump(folder/'ids.json',valid)
        values={s:core.compute_distance_matrix(p,prefix_mode=run['task'] in core.PREFIX_TASKS) for s,p in panels.items()}
        for s,m in values.items():
            assert np.isfinite(m).all() and np.allclose(m,m.T) and np.allclose(np.diag(m),0)
            np.save(folder/(s+'.npy'),m)
        if len(valid)>1:
            literal=core.profile_distance(panels['A'][0],panels['A'][-1])
            assert np.isclose(values['A'][0,-1],literal,rtol=1e-5,atol=1e-7)
        np.save(folder/'behavior.npy',(values['A']+values['B'])/2)
        upper=np.triu_indices(len(valid),1)
        corr=float(spearmanr(values['A'][upper],values['B'][upper]).statistic) if len(valid)>2 else None
        k=min(10,len(valid)-1)
        nearest={s:np.argsort(m+np.eye(len(valid))*1e6,axis=1)[:,:k] for s,m in values.items()}
        overlap=float(np.mean([len(set(a)&set(b))/k for a,b in zip(nearest['A'],nearest['B'])])) if k>0 else None
        stability.append(dict(run=run['run_name'],task=run['task'],parents=len(nodes),valid=len(valid),
                              panel_spearman=corr if corr is None or np.isfinite(corr) else None,knn_overlap=overlap))
        print('matrix',run['run_name'],len(valid),'rho',corr,flush=True)
    dump(out/'stability.json',stability)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['validate','profile','matrices'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--workers',type=int,default=6);p.add_argument('--ready-only',action='store_true')
    args=p.parse_args()
    if args.stage=='validate':validate(args.out)
    elif args.stage=='profile':profile(args.out,args.workers)
    else:matrices(args.out,args.ready_only)
