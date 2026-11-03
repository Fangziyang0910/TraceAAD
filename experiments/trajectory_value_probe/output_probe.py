"""Model-free input/output and executed-line probes on locked historical pairs."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import difflib
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def changed_statements(parent,child):
    changed=set()
    for op,_,_,a,b in difflib.SequenceMatcher(None,parent.splitlines(),child.splitlines(),autojunk=False).get_opcodes():
        if op in ('insert','replace'):changed.update(range(a+1,b+1))
    tree=ast.parse(child)
    excluded=(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom)
    return sorted({n.lineno for n in ast.walk(tree) if isinstance(n,ast.stmt) and not isinstance(n,excluded)
                   and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))
                   and n.lineno in changed})


@lru_cache(maxsize=6)
def inputs(task_name):
    import numpy as np
    from traceaad_base.cli import load_task
    task=load_task(name=task_name);e=task.evaluate_program.__self__;cases=[]
    for index in range(2):
        if task_name=='cvrp_aco':
            from benchmarks.cvrp_aco.evaluation import _distance_matrix
            instance=e._datasets[index];demand=instance[:,0].copy();coords=instance[:,1:].copy()
            cases.append((_distance_matrix(coords),coords,demand,e.capacity))
        elif task_name=='op_aco':
            from benchmarks.op_aco.evaluation import gen_prizes,gen_distance_matrix
            coords=e._datasets[index];cases.append((gen_prizes(coords),gen_distance_matrix(coords),float(e.max_len)))
        elif task_name=='tsp_construct':
            coords,dist=e._datasets[index];visited=np.zeros(len(coords),bool);visited[0]=True;current=0
            for step in range(len(coords)-2):
                order=np.argsort(dist[current]);unvisited=order[~visited[order]]
                if step in (0,len(coords)//3,2*len(coords)//3):cases.append((current,0,unvisited.copy(),dist.copy()))
                current=int(unvisited[0]);visited[current]=True
        elif task_name=='graph_colouring':cases.append((e._instances[index]['adjacency'].copy(),))
        elif task_name=='jssp_construct':
            d=e._instances[index];cases.append((d['processing_times'].copy(),d['machine_order'].copy()))
        elif task_name=='fssp_gls':
            from benchmarks.fssp_gls.evaluation import neh
            t=e._instances[index]['processing_times'].astype(float);s=neh(t)
            cases.extend(((s.copy(),t.copy(),t.shape[1],t.shape[0]),(s[::-1].copy(),t.copy(),t.shape[1],t.shape[0])))
    return task.entrypoint,cases


def run_program(code,entry,cases,changed,task_name,trace_enabled=True):
    import numpy as np
    import resource
    # Same memory cap as the frozen evaluation, only inside isolated worker.
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    outputs=[];hits=set();filename='<trajectory-probe>'
    def trace(frame,event,arg):
        if event=='line' and frame.f_code.co_filename==filename and frame.f_lineno in changed:hits.add(frame.f_lineno)
        return trace if frame.f_code.co_filename==filename else None
    compiled=compile(code,filename,'exec')
    for i,args in enumerate(cases):
        random.seed(730241+i);np.random.seed(730241+i);namespace={};args=copy.deepcopy(args)
        try:
            if trace_enabled:sys.settrace(trace)
            exec(compiled,namespace);out=namespace[entry](*args)
        finally:sys.settrace(None)
        if task_name=='graph_colouring':
            from benchmarks.graph_colouring.evaluation import validate_coloring
            validate_coloring(cases[i][0],out)
        elif task_name=='jssp_construct':
            from benchmarks.jssp_construct.evaluation import validate_schedule
            validate_schedule(*cases[i],out)
        elif task_name=='tsp_construct':
            assert isinstance(out,(int,np.integer)) and not isinstance(out,(bool,np.bool_)) and out in cases[i][2]
        elif task_name in ('cvrp_aco','op_aco'):
            arr=np.asarray(out);n=len(cases[i][0]);assert arr.shape==(n,n) and np.isfinite(arr).all()
        elif task_name=='fssp_gls':
            mat,jobs=map(np.asarray,out);t=cases[i][1]
            assert mat.shape==t.shape and np.isfinite(mat).all() and (mat>=0).all()
            assert jobs.ndim==1 and 1<=len(jobs)<=5 and jobs.dtype.kind in 'iu'
            assert len(set(jobs.tolist()))==len(jobs) and (jobs>=0).all() and (jobs<len(t)).all()
        outputs.append(copy.deepcopy(out))
    return {'outputs':outputs,'executed_changed_lines':sorted(hits)}


def equal(a,b):
    import numpy as np
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return bool(np.array_equal(np.asarray(a),np.asarray(b)))


def worker(row,base_root,socket,trace_enabled=True):
    sys.path.insert(0,base_root)
    from traceaad_base.runtime import isolated_call
    from experiments.traceaad_base.scheduling import Lease
    start=time.monotonic();result={k:v for k,v in row.items() if not k.endswith('_code')}
    result['probe_status']='error';result['trace_enabled']=trace_enabled
    try:
        entry,cases=inputs(row['task']);changed=changed_statements(row['parent_code'],row['child_code'])
        result['changed_statement_lines']=changed;result['probe_cases']=len(cases)
        with Lease(socket,'cpu','trajectory-output:'+row['pair_id']) as lease:
            cpu=lease.acquire();original=os.sched_getaffinity(0)
            try:
                os.sched_setaffinity(0,{int(cpu)})
                parent,pt=isolated_call(run_program,(row['parent_code'],entry,cases,[],row['task'],trace_enabled),30.)
                child,ct=isolated_call(run_program,(row['child_code'],entry,cases,changed,row['task'],trace_enabled),30.)
            finally:
                os.sched_setaffinity(0,original);lease.poll(0,[cpu])
        per_case=[equal(a,b) for a,b in zip(parent['outputs'],child['outputs'])]
        result.update(probe_status='success',all_outputs_equal=all(per_case),per_case_equal=per_case,
                      executed_changed_lines=child['executed_changed_lines'],
                      changed_statements_unexecuted=(not child['executed_changed_lines']) if changed and trace_enabled else None,
                      execution_seconds=pt+ct)
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'[-3000:]
    result['elapsed_seconds']=time.monotonic()-start
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-root',required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--socket',default='/tmp/traceaad-1003/scheduler.sock')
    p.add_argument('--retry-output-only',action='store_true');a=p.parse_args()
    plan=json.loads((a.data/'sample.locked.json').read_text());manifest=json.loads((a.data/'manifest.json').read_text())
    assert hashlib.sha256((a.data/'sample.locked.json').read_bytes()).hexdigest()==manifest['sample_sha256']
    destination=a.data/('results-output-only' if a.retry_output_only else 'results');destination.mkdir(exist_ok=True)
    jobs=plan['sampled_pairs']
    if a.retry_output_only:
        originals=[a.data/'results'/(r['pair_id']+'.json') for r in jobs]
        assert all(p.exists() for p in originals),'Finish the original locked sample first'
        # Tracing changes time-limited search behavior. Recover output comparisons
        # without tracing for failures and all clock-dependent source pairs.
        clock_tokens=('time.time','perf_counter','monotonic','process_time')
        jobs=[r for r,p in zip(jobs,originals) if json.loads(p.read_text())['probe_status']!='success'
              or any(token in r['parent_code']+'\n'+r['child_code'] for token in clock_tokens)]
    jobs=[r for r in jobs if not (destination/(r['pair_id']+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(worker,r,a.base_root,a.socket,not a.retry_output_only) for r in jobs]
        for future in as_completed(pending):
            result=future.result();(destination/(result['pair_id']+'.json')).write_text(json.dumps(result,indent=2))
            print(json.dumps({k:result.get(k) for k in ('pair_id','probe_status','all_outputs_equal','changed_statements_unexecuted','error')}),flush=True)


if __name__=='__main__':main()
