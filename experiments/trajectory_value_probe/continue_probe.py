"""Bounded private continuations using an existing frozen Base runtime.

Run as a script so --base-root determines all search/evaluation imports.
No value feature or manually diagnosed opportunity is inserted in generation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text())


def setup(args,task_name):
    sys.path.insert(0,str(args.base_root))
    from traceaad_base.cli import load_task
    from experiments.traceaad_base.run import configure_time_budget
    from experiments.traceaad_base.scheduling import load_environment
    load_environment(Path('/home/fzy/code/LLM4AD/TraceAAD/.env'))
    task=load_task(name=task_name)
    configure_time_budget(task,10.)
    task.memory_limit_bytes=2<<30
    return task


def preflight(args):
    sys.path.insert(0,str(args.base_root))
    from experiments.traceaad_base.scheduling import ScheduledEvaluator,scheduler_status
    from traceaad_base.storage import write_json
    lock=read(args.data/'candidates.locked.json')
    manifest=read(args.data/'manifest.json')
    assert hashlib.sha256((args.data/'candidates.locked.json').read_bytes()).hexdigest()==manifest['candidate_lock_sha256']
    status=scheduler_status(args.socket)
    results=[]
    for c in lock['candidates']:
        prefix=read(args.data/c['prefix_file']);task=setup(args,c['task'])
        expected=prefix['run_config']['task_definition']
        assert task.task_description==expected['task_description']
        assert task.time_limit==expected['time_limit']
        assert task.metadata['problem_size']==expected['metadata']['problem_size']
        assert task.evaluation_seed==expected['evaluation_seed']
        code=next(p['code'] for p in prefix['archive'] if p['id']==c['node_id'])
        evaluator=ScheduledEvaluator(task,args.socket,'trajectory-probe:preflight:'+c['candidate_key'])
        with evaluator.evaluation_context():
            start=time.monotonic();e=evaluator.evaluate(code);seconds=time.monotonic()-start
        matches=e.status=='success' and math.isclose(e.fitness,c['fitness'],rel_tol=1e-10,abs_tol=1e-9)
        row={'candidate_key':c['candidate_key'],'expected':c['fitness'],'evaluation':asdict(e),
             'seconds':seconds,'matches':matches,'code_sha256':hashlib.sha256(code.encode()).hexdigest()}
        results.append(row);print(json.dumps(row),flush=True)
        write_json(args.data/'pilot/preflight.json',{'results':results,'scheduler_instance':status['instance_id'],
                   'condition':'frozen schema Base; train whole-task 160s; no condition details inserted into generation'})
    hashes={str(p.relative_to(args.base_root)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('traceaad_base','benchmarks','experiments/traceaad_base') for p in (args.base_root/directory).rglob('*.py')}
    write_json(args.data/'pilot/source_manifest.json',{'base_root':str(args.base_root),'hashes':hashes,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version})


def run_job(args):
    lock=read(args.data/'candidates.locked.json')
    job=next(j for j in lock['jobs'] if j['job_id']==args.job)
    candidate=next(c for c in lock['candidates'] if c['candidate_key']==job['candidate_key'])
    checked=read(args.data/'pilot/preflight.json')['results']
    assert next(x for x in checked if x['candidate_key']==candidate['candidate_key'])['matches']
    root=args.data/'pilot/jobs'/job['job_id']
    if (root/'result.json').exists():return
    task=setup(args,candidate['task'])
    from traceaad_base import TraceAAD,Config
    from traceaad_base.models import Program,Attempt
    from traceaad_base.storage import write_json
    from experiments.traceaad_base.scheduling import ScheduledModel,ScheduledEvaluator
    prefix=read(args.data/candidate['prefix_file'])
    config=Config(evaluation_budget=job['request_budget'],initial_size=1,seed=20261103+job['repeat'],
                  temperature=1.,max_tokens=16384,model_timeout=600.)
    model=ScheduledModel(args.socket,'trajectory-probe:'+job['job_id'])
    with TraceAAD(task,model,config=config,run_dir=root/'native') as search:
        search.evaluator=ScheduledEvaluator(task,args.socket,'trajectory-probe:'+job['job_id'])
        search.archive={p['id']:Program.from_dict(p) for p in prefix['archive']}
        search.next_program_id=max(search.archive)+1
        search._save()
        original=search.archive[candidate['node_id']]
        current=original; best=original;valid_outcomes=[];checkpoints=[]
        incumbent=min(p.evaluation.fitness for p in search.archive.values() if p.valid)
        start=time.monotonic()

        def checkpoint():
            row={'requests':search.model_calls,'evaluations':search.evaluations_used,
                 'best_fitness':best.evaluation.fitness,'best_id':best.id,
                 'improvement_from_start':original.evaluation.fitness-best.evaluation.fitness,
                 'beats_frozen_incumbent':best.evaluation.fitness<incumbent-1e-9}
            checkpoints.append(row)
            write_json(root/'progress.json',{'job':job,'candidate':candidate['candidate_key'],
                'status':'running','checkpoints':checkpoints,'elapsed_seconds':time.monotonic()-start})
            print(json.dumps({'job_id':job['job_id'],**row}),flush=True)

        def call(attempt,prompt,kind=None):
            if search.model_calls>=job['request_budget']:raise RuntimeError('request budget overrun')
            reply=search._model_call(attempt,prompt,kind)
            parsed=search._parse(attempt,reply) if reply is not None else None
            nonlocal best
            if parsed is not None:
                search._evaluate(attempt,parsed)
                program=search.archive[attempt.program_id]
                if program.valid:
                    valid_outcomes.append({'request':search.model_calls,'id':program.id,'fitness':program.evaluation.fitness,
                        'code_sha256':hashlib.sha256(program.code.encode()).hexdigest()})
                    if program.evaluation.fitness<best.evaluation.fitness-1e-9:best=program
            checkpoint()
            return reply,parsed

        first=True
        while search.model_calls<job['request_budget']:
            op=job['first_operator'] if first else 'Refine'
            a=Attempt(len(search.attempts)+1,current.id,op,None)
            search.attempts.append(a);search._save()
            reply,parsed=call(a,search.prompts.evolution(search.archive,current,op,None))
            if reply is not None and parsed is None and search.model_calls<job['request_budget']:
                a.repair_used=True
                reply,parsed=call(a,search.prompts.repair_format(reply,a.calls[-1]['format_failure']),'format')
            if parsed is not None:
                p=search.archive[a.program_id]
                if not p.valid and not a.repair_used and search.model_calls<job['request_budget']:
                    a.repair_used=True
                    call(a,search.prompts.repair_program(p),p.evaluation.status)
            search._finish(a)
            if a.program_id is not None:
                child=search.archive[a.program_id]
                if child.valid and ((first and op=='Explore') or child.evaluation.fitness<current.evaluation.fitness-1e-9):
                    current=child
            first=False
        result={'job':job,'candidate_key':candidate['candidate_key'],'role':candidate['role'],
                'task':candidate['task'],'status':'completed','requests':search.model_calls,
                'evaluations':search.evaluations_used,'starting_fitness':original.evaluation.fitness,
                'frozen_incumbent':incumbent,'best_fitness':best.evaluation.fitness,
                'checkpoints':checkpoints,'valid_outcomes':valid_outcomes,'elapsed_seconds':time.monotonic()-start,
                'model_endpoints':sorted({c.get('metadata',{}).get('scheduler',{}).get('endpoint','unknown') for a in search.attempts for c in a.calls})}
        assert search.model_calls==job['request_budget'] and search.evaluations_used<=search.model_calls
        write_json(root/'result.json',result)
        (root/'best.py').write_text(best.code)


def launch(args):
    lock=read(args.data/'candidates.locked.json');checked=read(args.data/'pilot/preflight.json')['results']
    allowed={r['candidate_key'] for r in checked if r['matches']}
    jobs=[j for j in lock['jobs'] if j['candidate_key'] in allowed]
    results=[]
    def one(job):
        log=args.data/'pilot/logs'/f"{job['job_id']}.log";log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve()),'job','--base-root',str(args.base_root),
             '--data',str(args.data),'--socket',args.socket,'--job',job['job_id']]
        with log.open('w') as f:
            proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1'})
        return {'job_id':job['job_id'],'returncode':proc.returncode}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(one,j) for j in jobs]):
            r=future.result();results.append(r);print(json.dumps(r),flush=True)
    (args.data/'pilot/launch_results.json').write_text(json.dumps({'jobs':results,'planned':len(lock['jobs']),
        'launched':len(jobs),'finished_at':datetime.now(timezone.utc).isoformat()},indent=2))
    if any(r['returncode'] for r in results):raise RuntimeError('A continuation failed; preserve partial budget, do not silently rerun')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','launch','job'])
    p.add_argument('--base-root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--socket',default='/tmp/traceaad-1003/scheduler.sock');p.add_argument('--job')
    args=p.parse_args();{'preflight':preflight,'launch':launch,'job':run_job}[args.action](args)
