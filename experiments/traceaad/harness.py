"""Capture real EvoTrace V2 traces, render changes, and execute paired replays.

No search is resumed or modified. Quality review uses the saved code and outcomes;
this harness does not assign algorithm quality from wording or code distance.
"""
import argparse
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys


def read_rows(path):
    # Match Store.rows: complete newline-terminated records are committed.
    raw=path.read_bytes()
    if path.suffix=='.gz':
        raw=gzip.decompress(raw)
    for line in raw.split(b'\n')[:-1]:
        yield json.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def capture(args):
    from evotrace_v2.models import Attempt, Edge, Node, Program, TreePopulation
    from evotrace_v2.reflection import clean_prompt_text
    from evotrace_v2.search import deserialize_tree

    out = Path(args.out)
    frozen={p.stem:json.loads(p.read_text()) for p in (out/'cases').glob('*.json')}
    inventory = []
    for task_dir in sorted(Path(args.runs).iterdir()):
        if not task_dir.is_dir():
            continue
        run_dirs = sorted(task_dir.glob(f'{args.batch}*/native'))
        if not run_dirs:
            continue
        candidates = []
        for native in run_dirs:
            cfg = json.loads((native.parent / 'run_config.json').read_text())
            settings = json.loads((native / 'run_config.json').read_text())
            archive = {}
            spec = cfg['task_definition']
            if spec.get('example_code'):
                archive[0] = Program(0, spec['example_code'], spec.get('example_design'), None, None, None, None)
            tree = TreePopulation()
            calls = {}
            raw = (native / 'attempts.jsonl').read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            prefix=out/'prefixes'/native.parent.name/'attempts.jsonl.gz'
            if not prefix.exists():
                prefix.parent.mkdir(parents=True,exist_ok=True)
                prefix.write_bytes(gzip.compress(raw))
            for c in read_rows(native / 'calls.jsonl'):
                calls.setdefault(c['attempt_id'], []).append(c)
            for line in raw.split(b'\n')[:-1]:
                row = json.loads(line)
                a = Attempt(**row['attempt'])
                gen = next((c for c in calls.get(a.id, []) if c['stage'] == 'generate'), None)
                base = {'task': cfg, 'settings': settings, 'run': str(native.parent),
                        'input_sha256': sha, 'attempt_id': a.id,
                        'evaluations_before': tree.evaluations_used}
                if gen:
                    candidates.append({**base, 'role': a.operator, 'phase': 'generation',
                        'parent_id': a.parent_id, 'reference_id': a.reference_id,
                        'tree': asdict(tree), 'archive': {i: asdict(p) for i,p in archive.items()} if a.operator=='Init' else {},
                        'original_call': gen, 'original_program': row['program'],
                        'original_evaluations': a.evaluations, 'context': a.record.get('context', {})})
                for index, call in enumerate(calls.get(a.id, [])):
                    if call['stage'] != 'repair':
                        continue
                    previous = calls[a.id][index-1]
                    failed_program = None
                    if call['repair_kind'] != 'format':
                        parsed = json.loads(previous['response'])
                        failed_program = {**row['program'], 'code':parsed['code'],
                            'design':parsed['design'], 'analysis':parsed['analysis'],
                            'evaluation':a.evaluations[0]['evaluation']}
                    candidates.append({**base, 'role':f'Repair_{call["repair_kind"]}', 'phase':'repair',
                        'original_call':call,'repair_response':previous['response'],
                        'format_failure':a.calls[index-1].get('format_failure'),
                        'failed_program':failed_program,'original_evaluations':a.evaluations})
                p = Program.from_dict(row['program']) if row['program'] else None
                if p:
                    archive[p.id] = p
                    tree.add(Node(p.id, p.code, clean_prompt_text(p.design or ''),
                                  p.evaluation.fitness if p.valid else None, p.evaluation.failure),
                             Edge(a.parent_id, p.id, None if a.operator=='Init' else a.operator, a.reference_id),
                             evaluations=len(a.evaluations))
                c = a.record.get('consolidation')
                if c:
                    for call in c['reflection_calls']:
                        key = 'path_contractions' if call['kind']=='variation' else 'branch_consolidations'
                        field = 'child_id' if call['kind']=='variation' else 'parent_id'
                        region = next(r for r in c[key] if r[field]==call['target_id'])
                        candidates.append({**base, 'role': call['kind'], 'phase': 'reflection',
                            'tree': asdict(tree), 'region': region, 'plan': {k:v for k,v in c.items() if k!='reflection_calls'},
                            'original_call': call, 'tree_after': row['tree_after']})
                    tree = deserialize_tree(row['tree_after'])
        # Fixed early seed; later cases chosen by evidence availability, never fitness.
        chosen = []
        for role in ['Init','Refine','Crossover','Explore']:
            matches = [c for c in candidates if c['phase']=='generation' and c['role']==role
                       and c['task']['seed']==0]
            if matches:
                chosen.append((f'early_{role}', matches[0]))
        init = [c for c in candidates if c['role']=='Init' and c['task']['seed']==0]
        if len(init)>1:
            chosen.append(('initial_diversity', init[-1]))
        for role in ['Refine','Crossover','Explore']:
            matches = [c for c in candidates if c['phase']=='generation' and c['role']==role and c['evaluations_before']>=50]
            if matches:
                def richness(c):
                    t=c['tree']
                    return (sum(bool(n['selection_experience']) for n in t['nodes'].values())+
                            sum(bool(e['variation_experience']) for e in t['edges'].values()),c['evaluations_before'])
                chosen.append((f'developed_{role}', max(matches,key=richness)))
        for role in ['selection','variation']:
            matches=[c for c in candidates if c['phase']=='reflection' and c['role']==role]
            if matches:
                chosen.append((f'reflect_{role}',matches[0]))
                multi=[c for c in matches if len(c['region']['removed_ids'])>1]
                if multi:
                    chosen.append((f'multi_{role}',multi[0]))
        for kind in ['error','timeout','format']:
            matches=[c for c in candidates if c['role']==f'Repair_{kind}']
            if matches:
                chosen.append((f'repair_{kind}',matches[0]))
        for label,c in chosen:
            case=f'{task_dir.name}_{label}'
            if frozen and case not in frozen:
                continue
            c['case']=case
            dest=out/'cases'/f'{case}.json'
            if dest.exists():
                c=json.loads(dest.read_text())
            else:
                write_json(dest,c)
            inventory.append({'case':case,'role':c['role'],'phase':c['phase'],'run':c['run'],
                              'attempt':c['attempt_id'],'evaluations_before':c['evaluations_before']})
    if frozen:
        inventory=[{'case':c['case'],'role':c['role'],'phase':c['phase'],'run':c['run'],
                    'attempt':c['attempt_id'],'evaluations_before':c['evaluations_before']}
                   for c in frozen.values()]
    write_json(out/'inventory.json',inventory)
    print(f'Captured {len(inventory)} real cases',flush=True)


def render(args):
    from evotrace_v2.config import Config
    from evotrace_v2.context import ContextConfig, EvoTracePromptBuilder
    from evotrace_v2.models import Program, CANDIDATE_RESPONSE_FORMAT
    from evotrace_v2.reflection import build_selection_prompt, build_variation_prompt, SELECTION_SCHEMA, VARIATION_SCHEMA
    from evotrace_v2.search import deserialize_tree
    from evotrace_v2.task import Task
    from evotrace_v2.topology import BranchConsolidation, PathContraction

    out=Path(args.out)
    jobs=[]
    for path in sorted((out/'cases').glob('*.json')):
        c=json.loads(path.read_text())
        if not re.search(args.match,c['case']):
            continue
        tree=deserialize_tree(c['tree']) if 'tree' in c else None
        task=Task(evaluate_program=lambda _:None,**c['task']['task_definition'])
        if c['phase']=='repair':
            builder=EvoTracePromptBuilder(task,Config(**c['settings']['config']))
            prompt=(builder.repair_format(c['repair_response'],c['format_failure']) if c['role']=='Repair_format'
                    else builder.repair_program(Program.from_dict(c['failed_program'])))
            schema=CANDIDATE_RESPONSE_FORMAT
        elif c['phase']=='generation':
            builder=EvoTracePromptBuilder(task,Config(**c['settings']['config']))
            if c['role']=='Init':
                prompt=builder.initialization({int(i):Program.from_dict(p) for i,p in c['archive'].items()})
            else:
                # Reproduce recorded window reductions without invoking the tokenizer.
                cfg=ContextConfig(**c['settings']['context'])
                for change in c['context'].get('input_window',{}).get('reductions',[]):
                    from dataclasses import replace
                    cfg=replace(cfg,**{change['field']:change['retained']})
                from evotrace_v2.context import build_evolution_context
                text,instruction,scores,_=build_evolution_context(tree,c['parent_id'],c['role'],c['reference_id'],cfg)
                prompt=builder._render(text,instruction,has_scores=scores)
            schema=CANDIDATE_RESPONSE_FORMAT
        else:
            kw=dict(task_description=task.task_description,score_definition=task.score_definition,
                    function_template=task.function_template)
            if c['role']=='selection':
                prompt=build_selection_prompt(tree,BranchConsolidation(**c['region']),**kw)
                schema=SELECTION_SCHEMA
            else:
                prompt=build_variation_prompt(tree,PathContraction(**c['region']),**kw)
                schema=VARIATION_SCHEMA
        identical=prompt==c['original_call']['prompt']
        if args.require_original and not identical:
            raise ValueError(f'Original prompt replay differs: {c["case"]}')
        jobs.append({'case':c['case'],'phase':c['phase'],'role':c['role'],'prompt':prompt,'schema':schema,
                     'task':c['task'],'original_identical':identical,'arm':args.arm})
        (out/'prompts'/args.arm).mkdir(parents=True,exist_ok=True)
        (out/'prompts'/args.arm/f'{c["case"]}.txt').write_text(prompt)
    write_json(out/f'inputs_{args.arm}.json',jobs)
    print(f'Rendered {len(jobs)} cases; {sum(j["original_identical"] for j in jobs)} identical to actual input')


def handoff(args):
    """Replay actual consolidation with one fresh summary, then render its consumer."""
    from evotrace_v2.config import Config
    from evotrace_v2.context import ContextConfig, EvoTracePromptBuilder, build_evolution_context
    from evotrace_v2.models import CANDIDATE_RESPONSE_FORMAT
    from evotrace_v2.reflection import reflect_and_apply
    from evotrace_v2.search import deserialize_tree
    from evotrace_v2.task import Task
    from evotrace_v2.topology import ConsolidationPlan, PathContraction, BranchConsolidation

    class RecordedModel:
        def __init__(self, replies):
            self.replies=iter(replies)
        def complete(self, *_args, **_kwargs):
            return next(self.replies)

    out=Path(args.out)
    jobs=[]
    for file in sorted((out/'cases').glob('*.json')):
        c=json.loads(file.read_text())
        if c['phase']!='reflection' or not re.search(args.match,c['case']):
            continue
        response_path=out/'responses'/args.arm/f'{c["case"]}_0.json'
        response=json.loads(response_path.read_text())
        if response.get('error'):
            raise ValueError(f'Cannot hand off failed reflection: {response_path}')
        prefix=out/'prefixes'/Path(c['run']).name/'attempts.jsonl.gz'
        trace=prefix if prefix.exists() else Path(c['run'])/'native/attempts.jsonl'
        original=next(r for r in read_rows(trace)
                      if r['attempt']['id']==c['attempt_id'])['attempt']['record']['consolidation']
        # Other regions use exactly the same recorded replies in both arms.
        replies=[response['response'] if x['kind']==c['role'] and x['target_id']==c['original_call']['target_id']
                 else x['response'] for x in original['reflection_calls']]
        p=c['plan']
        plan=ConsolidationPlan(tuple(p['kept_ids']),tuple(PathContraction(**v) for v in p['path_contractions']),
                               tuple(BranchConsolidation(**v) for v in p['branch_consolidations']),tuple(p['triggers']))
        task=Task(evaluate_program=lambda _:None,**c['task']['task_definition'])
        tree,records=reflect_and_apply(deserialize_tree(c['tree']),plan,model=RecordedModel(replies),
            task_description=task.task_description,score_definition=task.score_definition,
            function_template=task.function_template)
        assert all(not r.used_fallback for r in records)
        target=c['original_call']['target_id']
        if c['role']=='variation':
            parent=target
            role='Refine'
        elif target==tree.root_id:
            parent=min((n for n in tree.nodes.values() if n.fitness is not None),key=lambda n:(n.fitness,n.id)).id
            role='Explore'
        else:
            parent=target
            role='Refine'
        body,instruction,scores,view=build_evolution_context(tree,parent,role,config=ContextConfig(**c['settings']['context']))
        prompt=EvoTracePromptBuilder(task,Config(**c['settings']['config']))._render(body,instruction,has_scores=scores)
        selected=next(r.experience for r in records if r.kind==c['role'] and r.target_id==target)
        assert selected in prompt, f'Fresh experience did not reach consumer: {c["case"]}'
        arm=args.arm+'_handoff'
        write_json(out/'handoffs'/arm/f'{c["case"]}.json',{'tree':asdict(tree),'reflection_records':[asdict(r) for r in records],
                   'consumer':view,'prompt':prompt,'response_source':str(response_path)})
        jobs.append({'case':c['case'],'phase':'generation','role':role,'prompt':prompt,'schema':CANDIDATE_RESPONSE_FORMAT,
                     'task':c['task'],'arm':arm,'reflection_source':str(response_path)})
    write_json(out/f'inputs_{args.arm}_handoff.json',jobs)
    print(f'Replayed {len(jobs)} real consolidation-to-generation handoffs')


def execute(job):
    out, source, item, repeat, socket = job
    if source:
        sys.path.insert(0,source)
    from experiments.traceaad.scheduling import ScheduledModel
    dest=Path(out)/'responses'/item['arm']/f'{item["case"]}_{repeat}.json'
    if dest.exists():
        cached=json.loads(dest.read_text())
        if cached.get('prompt_sha256')!=hashlib.sha256(item['prompt'].encode()).hexdigest():
            raise ValueError(f'Cached response has different or unrecorded prompt identity: {dest}; use a fresh arm')
        return f'Kept {dest.name}'
    result={'case':item['case'],'arm':item['arm'],'repeat':repeat,'phase':item['phase'],
            'prompt_sha256':hashlib.sha256(item['prompt'].encode()).hexdigest()}
    stage='request'
    try:
        model=ScheduledModel(socket,'evotrace-harness:'+item['case']).for_role(item['schema'])
        with model.request_context():
            r=model.complete(item['prompt'],temperature=.3 if item['phase']=='reflection' else 1.,
                             max_tokens=1536 if item['phase']=='reflection' else 16384,timeout=600)
        result.update(response=r.content,usage=r.usage,metadata=r.metadata,finish_reason=r.finish_reason)
        stage='parse'
        result['parsed']=json.loads(r.content)
        stage='validation'
        if item['phase']!='reflection':
            from evotrace_v2.models import Candidate
            from evotrace_v2.task import Task
            candidate=Candidate.parse(r.content)
            Task(evaluate_program=lambda _:None,**item['task']['task_definition']).validate_source(candidate.code)
        else:
            assert isinstance(result['parsed'].get('experience'),str) and result['parsed']['experience'].strip()
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        result['failure_stage']=stage
    write_json(dest,result)
    return f'{dest.name}: '+result.get('error','received')


def evaluate(job):
    out,source,item,path,socket=job
    if source:
        sys.path.insert(0,source)
    from evotrace_v2.evaluation import Evaluator
    from evotrace_v2.task import Task
    from experiments.traceaad.scheduling import ScheduledEvaluator
    d=json.loads(Path(path).read_text())
    if 'evaluation' in d or not d.get('parsed',{}).get('code'):
        return f'Kept {Path(path).name}'
    cfg=item['task'];spec=cfg['task_definition']
    module,_,name=spec['metadata']['evaluator'].rpartition('.')
    e=getattr(importlib.import_module(module),name)(**cfg['task_eval'])
    task=Task(evaluate_program=e.evaluate_program,**spec)
    try:
        evaluator=ScheduledEvaluator(task,socket,'evotrace-harness-eval',evaluator_factory=Evaluator)
        with evaluator.evaluation_context():
            result=evaluator.evaluate(d['parsed']['code'])
        d['evaluation']=asdict(result)
    except Exception as exc:
        d['evaluation']={'status':'error','failure':f'{type(exc).__name__}: {exc}'}
    write_json(Path(path),d)
    return f'{Path(path).name}: {d["evaluation"]}'


def full_step(job):
    """Run the real search step, reusing a captured generation and live repairs."""
    out,source,item,repeat,socket=job
    if source:
        sys.path.insert(0,source)
    from contextlib import contextmanager, nullcontext
    from evotrace_v2 import Config, EvoTrace, Task
    from evotrace_v2.allocation import AllocationConfig
    from evotrace_v2.context import ContextConfig
    from evotrace_v2.evaluation import Evaluator
    from evotrace_v2.models import Attempt, ModelReply, Program, CANDIDATE_RESPONSE_FORMAT
    from evotrace_v2.search import deserialize_tree
    from evotrace_v2.topology import TopologyConfig
    from experiments.traceaad.scheduling import ScheduledModel, ScheduledEvaluator
    import evotrace_v2

    out=Path(out)
    dest=out/'steps'/item['arm']/f'{item["case"]}_{repeat}.json'
    digest=hashlib.sha256()
    for file in sorted(Path(evotrace_v2.__file__).parent.glob('*.py')):
        digest.update(file.name.encode()+file.read_bytes())
    digest.update(json.dumps(item['task'],sort_keys=True).encode())
    source_hash=digest.hexdigest()
    if dest.exists():
        saved=json.loads(dest.read_text())
        if (saved.get('source_sha256')!=source_hash or
                saved.get('prompt_sha256')!=hashlib.sha256(item['prompt'].encode()).hexdigest()):
            raise ValueError(f'Existing step has different source or input: {dest}; use a fresh arm')
        return f'Kept {dest.name}'
    response_path=out/'responses'/item['arm']/f'{item["case"]}_{repeat}.json'
    cached=json.loads(response_path.read_text())
    if cached.get('error'):
        raise ValueError(f'Cannot start a step with failed generation: {response_path}')
    case=json.loads((out/'cases'/f'{item["case"]}.json').read_text())
    cfg=case['task'];spec=cfg['task_definition'];settings=case['settings']
    module,_,name=spec['metadata']['evaluator'].rpartition('.')
    evaluator=getattr(importlib.import_module(module),name)(**cfg['task_eval'])
    task=Task(evaluate_program=evaluator.evaluate_program,**spec)
    live=ScheduledModel(socket,'evotrace-harness-step').for_role(CANDIDATE_RESPONSE_FORMAT)

    class CachedFirst:
        model='cached-generation-then-live-repair'
        def __init__(self):
            self.pending=True
            self.calls=[]
        def count_prompt_tokens(self,prompt):
            return live.count_prompt_tokens(prompt)
        @contextmanager
        def request_context(self):
            with (nullcontext() if self.pending else live.request_context()):
                yield
        def complete(self,prompt,**kwargs):
            replayed=self.pending
            if replayed:
                assert prompt==item['prompt'], 'Actual step rebuilt a different generation prompt'
                self.pending=False
                reply=ModelReply(cached['response'],usage=cached.get('usage'),metadata=cached.get('metadata'),
                                 model=cached.get('metadata',{}).get('scheduler',{}).get('model'))
            else:
                reply=live.complete(prompt,**kwargs)
            self.calls.append({'prompt':prompt,'response':reply.content,'replayed':replayed,
                               'usage':reply.usage,'metadata':reply.metadata})
            return reply

    model=CachedFirst()
    search=EvoTrace(task,model,config=Config(**settings['config']),
        topology_config=TopologyConfig(**settings['topology']),allocation_config=AllocationConfig(**settings['allocation']),
        context_config=ContextConfig(**settings['context']),reflection_model=live,
        reflection_temperature=settings['reflection_temperature'],reflection_max_tokens=settings['reflection_max_tokens'])
    prefix=out/'prefixes'/Path(case['run']).name/'attempts.jsonl.gz'
    trace=prefix if prefix.exists() else Path(case['run'])/'native/attempts.jsonl'
    for row in read_rows(trace):
        if row['attempt']['id']>=case['attempt_id']:
            break
        search.attempts.append(Attempt(**row['attempt']))
        if c:=row['attempt']['record'].get('consolidation'):
            search.consolidations_count+=1
            search.reflection_calls_count+=sum(bool(r['prompt_chars']) for r in c['reflection_calls'])
        if row['program']:
            program=Program.from_dict(row['program'])
            search.archive[program.id]=program
    search.tree=deserialize_tree(case['tree'])
    assert search.evaluations_used==case['evaluations_before']
    search.evaluator=ScheduledEvaluator(task,socket,'evotrace-harness-step-eval',evaluator_factory=Evaluator)
    best_before=search.best.evaluation.fitness
    attempt=search.step()
    program=search.archive.get(attempt.program_id)
    result={'case':item['case'],'arm':item['arm'],'repeat':repeat,'source_response':str(response_path),
        'attempt':asdict(attempt),'program':asdict(program) if program else None,'tree_after':asdict(search.tree),
        'calls':model.calls,'best_before':best_before,'best_after':search.best.evaluation.fitness,
        'evaluation':asdict(program.evaluation) if program else {'status':'error','failure':'No evaluated program'},
        'evaluation_budget_spent':len(attempt.evaluations),
        'reflection_fallbacks':sum(r['used_fallback'] for r in attempt.record.get('consolidation',{}).get('reflection_calls',[])),
        'source_sha256':source_hash,
        'prompt_sha256':hashlib.sha256(item['prompt'].encode()).hexdigest()}
    write_json(dest,result)
    return f'{item["case"]}: real step {result["evaluation"]}; evaluations={len(attempt.evaluations)}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['capture','render','handoff','run','evaluate','step'])
    parser.add_argument('--out',required=True)
    parser.add_argument('--runs',default='experiments_result/evotrace_v2_subtree')
    parser.add_argument('--batch',default='20261113_162932_evotrace_v2_subtree_co4')
    parser.add_argument('--source',help='Source root containing evotrace_v2, frozen independently of online runs')
    parser.add_argument('--arm',default='before')
    parser.add_argument('--match',default='.')
    parser.add_argument('--require-original',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--socket',default='/tmp/traceaad-1003/scheduler.sock')
    parser.add_argument('--env-file',action='append',default=[])
    args=parser.parse_args()
    if args.source:
        args.source=str(Path(args.source).resolve())
        sys.path.insert(0,args.source)
    if args.action=='capture':
        return capture(args)
    if args.action=='render':
        return render(args)
    if args.action=='handoff':
        return handoff(args)
    for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[var]='1'
    from experiments.traceaad.scheduling import load_environment
    for path in args.env_file:
        load_environment(path)
    items=[i for i in json.loads((Path(args.out)/f'inputs_{args.arm}.json').read_text()) if re.search(args.match,i['case'])]
    if args.action in ('run','step'):
        if args.action=='step':
            items=[i for i in items if i['phase']=='generation']
        jobs=[(args.out,args.source,i,r,args.socket) for i in items for r in range(args.repeats)]
        work=execute if args.action=='run' else full_step
    else:
        jobs=[(args.out,args.source,i,str(p),args.socket) for i in items if i['phase']!='reflection'
              for p in sorted((Path(args.out)/'responses'/args.arm).glob(f'{i["case"]}_*.json'))]
        work=evaluate
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(work,j) for j in jobs]):
            print(future.result(),flush=True)
    failures=[]
    recorded=0
    for item in items:
        if args.action=='evaluate' and item['phase']=='reflection':
            continue
        directory=Path(args.out)/('steps' if args.action=='step' else 'responses')/args.arm
        paths=([directory/f'{item["case"]}_{r}.json' for r in range(args.repeats)] if args.action in ('run','step')
               else list(directory.glob(f'{item["case"]}_*.json')))
        if not paths:
            failures.append(item['case']+': missing response')
        for path in paths:
            if not path.exists():
                failures.append(path.name+': missing response')
                continue
            d=json.loads(path.read_text())
            recorded+=1
            if d.get('error'):
                failures.append(path.name+': '+d['error'])
            elif args.action=='step' and d.get('reflection_fallbacks'):
                failures.append(path.name+': reflection used fallback')
            elif args.action in ('evaluate','step') and d.get('evaluation',{}).get('status')!='success':
                failures.append(path.name+': '+d.get('evaluation',{}).get('status','missing evaluation'))
    print(f'Recorded {recorded} results; {len(failures)} failed or missing',flush=True)
    if failures:
        for failure in failures:
            print(failure,flush=True)
        raise SystemExit(1)


if __name__=='__main__':
    main()
