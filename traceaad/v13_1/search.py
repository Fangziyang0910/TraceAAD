"""V13.1: frozen actions, forward diagnostics, conditional credit, resumable execution."""
from copy import deepcopy
from dataclasses import asdict
import json
import ast
import math
import tokenize
import time
from pathlib import Path

from core.llm import generate, ModelCallError, SERVICE_WAIT_SECONDS
from core.scheduling import SchedulerError
from traceaad.traceaad import TraceAAD
from traceaad.program import Program
from traceaad.delivery import validate_source
from traceaad.history import code_diff
from traceaad.selection import temperature
from traceaad.storage import read_json, write_json
from traceaad.v13.search import TraceAADV13, CallBudgetExhausted
from traceaad.v13.knowledge import digest, bindings, apply_edits, changed_regions
from .config import Config
from .prompts import prompt, node_view
from .knowledge import (specification, canonical, identity, comparable, valid_symbols,
                        token_template, render_template)
from . import models


class TraceAADV131(TraceAADV13):
    METHOD='traceaad_v13_1'

    def __init__(self, *, config=None, **kwargs):
        TraceAAD.__init__(self,config=config or Config(),**kwargs)
        self.task=kwargs.get('task')
        sources=sorted(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent.parent/'v13/search.py']
        implementation=digest(''.join(p.read_text() for p in sources))
        signature=digest(json.dumps(dict(config=asdict(self.config),task=self.task,
            template=self.template,implementation=implementation),sort_keys=True))
        saved=self.facts.state or {}
        if saved and ('v13_1' not in saved or saved['v13_1']['config_signature']!=signature):
            raise ValueError('V13.1 resume requires the same implementation and configuration')
        self.state=deepcopy(saved.get('v13_1') or dict(config_signature=signature,library={},routes={},
            attempts=[],work=None,reply=None,inflight=None,requests=0,interrupted=[],stages={},
            missing_usage=0,measured_input_tokens=0,measured_output_tokens=0,next_transform=1,
            stop_reason=None,prior=None,scale=None,warm_records=[],open_records=[],condition_cache={},
            evaluation_cache={},match_cache={},diagnostics=[],diagnostic_rounds=0,eligible_rounds=0,
            diagnostic_fallbacks={},search_rounds=0))
        if not saved and self.config.library_path: self._import_library(Path(self.config.library_path))

    def _save(self,**record):
        self.facts.commit({**self.progress.snapshot(),'v13_1':self.state},**record)

    def _ask(self,stage,required,optional=()):
        optional=list(optional)
        text=prompt(self.prompts.common,stage,required,optional)
        while self.llm.count_prompt_tokens(text)>self.config.prompt_tokens:
            if optional:
                optional.pop()
            elif stage=='learn' and required.get('library'):
                # Keep the actual source edit; comparison catalogs are optional.
                required={**required,'library':required['library'][:-1]}
            else:
                raise ValueError('required context exceeds V13.1 prompt limit')
            text=prompt(self.prompts.common,stage,required,optional)
        key=digest(text)
        if self.state['reply'] is not None:
            saved=self.state['reply']
            if saved['key']!=key or saved['stage']!=stage:
                raise RuntimeError('checkpointed response does not match the resumed stage')
            return self._json(saved)
        if self.state['inflight']:
            self.state['interrupted'].append(self.state['inflight']); self.state['inflight']=None
            self._save()
        request=dict(action=stage,input_tokens=self.llm.count_prompt_tokens(text))
        waited=0; pause=2
        while True:
            if self.state['requests']>=self.config.max_model_calls: raise CallBudgetExhausted()
            self.state['requests']+=1
            self.state['inflight']=dict(id=self.state['requests'],stage=stage,key=key)
            self._save()
            try:
                result=generate(self.llm,text,max_tokens=self.config.output_tokens,
                                response_format={'type':'json_object'},retry_limit=0)
                break
            except ModelCallError as exc:
                self._record_calls(exc.calls,request); self.state['inflight']=None; self._save(calls=exc.calls)
                if not exc.transient or waited>=SERVICE_WAIT_SECONDS: raise
                if self.state['requests']>=self.config.max_model_calls: raise CallBudgetExhausted() from exc
                time.sleep(pause); waited+=pause; pause=min(60,pause*2)
        self._record_calls(result['calls'],request)
        self.state['inflight']=None
        self.state['reply']=dict(key=key,stage=stage,content=result.get('content'),finish_reason=result.get('finish_reason'))
        self._save(calls=result['calls'])
        return self._json(self.state['reply'])

    def _context(self,work):
        if work['parent'] is None:
            return [dict(existing_roots=[dict(id=p.id,design=p.idea) for p in self.archive.values()])]
        parent=self.programs[work['parent']]
        path=models.primary_path(parent.id,self.programs)
        recent=[]
        for pid in path[:self.config.history_depth]:
            p=self.programs[pid]; before=self.programs.get(p.parent_id)
            recent.append(dict(id=pid,parent=p.parent_id,score=p.fitness,
                change=p.request.get('trace',{}).get('change',p.idea),
                diff=code_diff(before.code,p.code)[:3000] if before else None))
        branches=[dict(parent=a['parent'],child=a.get('child'),mode=a['mode'],outcome=a.get('outcome'),
                delta=a.get('delta'),transform=a.get('transform'),failure=a.get('failure'))
                for a in self.state['attempts'] if a['parent'] in path and a.get('child') not in path]
        return [dict(path=recent,side_trials=branches[-self.config.history_depth:],
                     note='History is evidence; historical failure is not a ban.')]

    def _start(self):
        if self.budget_used>=self.config.budget or self.state['requests']>=self.config.max_model_calls:
            self.state['stop_reason']='candidate_budget' if self.budget_used>=self.config.budget else 'model_call_budget'
            self.progress.phase='freeze'; self._save(); return
        roots=[p for p in self.archive.values() if p.parent_id is None]
        if self.phase=='roots' and (len(roots)>=self.config.roots or self.progress.init_attempts>=self.config.init_attempt_limit):
            self.progress.phase='search' if roots else 'freeze'
        if self.phase=='freeze': self._save(); return
        work=dict(attempt=self.budget_used+1,parent=None,reference=None,route=None,mode='Init',
                  stage='generate',child=None,transform=None,bindings=[],source='discovery',
                  matched={},match_index=0,repair_count=0)
        if self.phase=='roots': self.progress.init_attempts+=1
        else:
            # Equal route scheduling maintains distinct roots; fitness sampling is the host policy.
            routes=sorted({p.request['route'] for p in self.archive.values()})
            route=self.progress.rng['parent'].choice(routes)
            nodes=[p for p in self.archive.values() if p.request['route']==route]
            beta,_=temperature([p.fitness for p in nodes]); best=min(p.fitness for p in nodes)
            parent=self.progress.rng['parent'].choices(nodes,[math.exp(-beta*(p.fitness-best)) for p in nodes])[0]
            n=self.state['search_rounds']; self.state['search_rounds']+=1
            work.update(parent=parent.id,route=route,mode='Open',round=n+1,
                        forced_free=self.state['prior'] is None or n%10<self.config.free_per_ten,
                        diagnostic_slot=n%10 in range(self.config.free_per_ten,
                            self.config.free_per_ten+self.config.diagnostic_per_ten),
                        candidates=sorted(self.state['library']))
            # Match/measure even on forced-free rounds so opportunity frequency has a full denominator.
            uncached=[]
            for tid in work['candidates']:
                cached=self.state['match_cache'].get(f'{parent.id}:{tid}')
                if cached is None: uncached.append(tid)
                elif cached.get('match'):
                    m=deepcopy(cached['match'])
                    if not self._already_attempted(parent.id,tid,m['bindings']): work['matched'][tid]=m
            work['candidates']=uncached
            if uncached or work['matched']: work['stage']='match'
        self.progress.generations+=1; self.state['work']=work; self._save()

    def _generate_candidate(self,work):
        required=dict(mode=work['mode'])
        if work['parent'] is not None: required['CURRENT_PARENT']=node_view(self.programs[work['parent']])
        reply=self._ask('generate',required,self._context(work))
        if not isinstance(reply.get('code'),str) or not reply['code'].strip(): raise ValueError('missing complete program')
        work['proposal']=dict(design=str(reply.get('design',''))[:1500],analysis=str(reply.get('analysis',''))[:1500])
        self._new_program(reply['code'],reply)

    def _catalog(self,ids=None):
        return [dict(id=tid,version=t['version'],spec=t['spec'],reference_edit=t.get('reference_edit','')) for tid,t in self.state['library'].items()
                if ids is None or tid in ids]

    def _already_attempted(self,parent,tid,regions):
        return any(a.get('mode')=='Transform' and a.get('parent')==parent and a.get('transform')==tid
                   and a.get('bindings')==regions for a in self.state['attempts'])

    def _match(self,work):
        parent=self.programs[work['parent']]
        remaining=work['candidates'][work['match_index']:]
        if not remaining: self._choose(work); return
        batch=remaining[:self.config.match_batch]
        # Shrink catalogs BEFORE losing essential current-parent source.
        while len(batch)>1 and self.llm.count_prompt_tokens(prompt(self.prompts.common,'match',
                dict(CURRENT_PARENT=node_view(parent),candidates=self._catalog(batch))))>self.config.prompt_tokens:
            batch.pop()
        work['current_match_batch']=len(batch)
        reply=self._ask('match',dict(CURRENT_PARENT=node_view(parent),candidates=self._catalog(batch)))
        matches=reply.get('matches'); rejections=reply.get('rejections',{})
        if not isinstance(matches,list) or not isinstance(rejections,dict): raise ValueError('invalid match reply')
        parsed={}
        for match in matches:
            if not isinstance(match,dict): raise ValueError('each match must be an object')
            tid=match.get('id')
            if tid not in batch or tid in parsed: raise ValueError('unknown or duplicate match')
            regions=bindings(match.get('bindings'),parent.code)
            symbols=match.get('symbols',{})
            if not valid_symbols(symbols,parent.code): raise ValueError('invalid bound symbol roles')
            values=match.get('conditions')
            spec=self.state['library'][tid]['spec']
            if not isinstance(values,list) or len(values)!=len(spec['conditions']): raise ValueError('condition arity mismatch')
            measured=[]
            for h,v in zip(spec['conditions'],values):
                if not isinstance(v,dict) or not (v.get('value') is None or type(v['value']) is bool):
                    raise ValueError('condition must be true/false/unknown')
                evidence=v.get('evidence',[])
                supported=isinstance(evidence,list) and bool(evidence) and all(
                    isinstance(x,str) and x.strip() and x in parent.code for x in evidence)
                measured.append(v['value'] if h['kind']=='static' and supported else None)
            key=f'{parent.id}:{tid}'
            cache=self.state['condition_cache'].get(key)
            if cache is not None: measured=cache
            if self._already_attempted(parent.id,tid,regions):
                continue
            parsed[tid]=dict(bindings=regions,symbols=symbols,conditions=measured)
        covered={m.get('id') for m in matches}|set(rejections)
        if set(batch)-covered: raise ValueError('every candidate needs a match or rejection')
        for tid in batch:
            self.state['match_cache'][f'{parent.id}:{tid}']=dict(match=parsed.get(tid),rejection=rejections.get(tid))
        for tid,m in parsed.items(): self.state['condition_cache'].setdefault(f'{parent.id}:{tid}',m['conditions'])
        work['matched'].update(parsed); work.setdefault('rejections',{}).update(rejections)
        work['match_index']+=len(batch); self._advance('match')

    def _pairs(self,parent,tid,match):
        result=[]; transform=self.state['library'][tid]
        for i,r in enumerate(transform['records']):
            q=r['parent']
            if not r.get('credit') or not r['A'] or r.get('external') or q==parent or q not in self.programs:
                continue
            if not comparable(q,parent,r['bindings'],match['bindings'],self.programs): continue
            changed=any(a is not None and b is not None and a!=b for a,b in zip(r['conditions'],match['conditions']))
            d=models.distance(q,parent,self.programs)
            coverage=sum(x.get('transform')==tid and x.get('reference_parent')==q for x in self.state['diagnostics'])
            result.append(dict(record=i,parent=q,changed=changed,distance=d,coverage=coverage))
        return sorted(result,key=lambda x:(not x['changed'],x['distance'],x['coverage'],x['parent']))

    def _predict(self,tid,match,parent):
        t=self.state['library'][tid]
        return models.predictions(t,self.state['prior'] or [1.,1.,1.,1.],match['conditions'],parent,
                                  self.programs,self.config.lineage_decay)

    def _choose(self,work):
        candidates={}
        for tid,m in work['matched'].items():
            cells=self._predict(tid,m,work['parent']); t=self.state['library'][tid]
            probs=models.weights(t['log_weights']); pairs=self._pairs(work['parent'],tid,m)
            candidates[tid]=dict(cells=cells,weights=probs,score=models.disagreement(cells,probs),pairs=pairs)
        qualified=[tid for tid,c in candidates.items() if c['pairs']]
        work['opportunity']=dict(compatible=len(candidates),paired=len(qualified))
        self.state['eligible_rounds']+=bool(qualified)
        selected=None
        if not work['forced_free'] and work['diagnostic_slot']:
            active=[tid for tid in qualified if candidates[tid]['score']>self.config.diagnostic_threshold]
            if active:
                selected=min(active,key=lambda tid:(-candidates[tid]['score'],
                    not candidates[tid]['pairs'][0]['changed'],candidates[tid]['pairs'][0]['distance'],
                    candidates[tid]['pairs'][0]['coverage'],tid))
                work['source']='diagnostic_replay'; self.state['diagnostic_rounds']+=1
            else:
                reason='no_pair' if not qualified else 'low_disagreement'
                self.state['diagnostic_fallbacks'][reason]=self.state['diagnostic_fallbacks'].get(reason,0)+1
                work['diagnostic_fallback']=reason
        if not work['forced_free'] and selected is None:
            rng=self.progress.rng['action']
            prior=self.state['prior']; open_cell=models.cell(prior,self.state['open_records'])
            draws={'Open':models.sample(open_cell,rng)}
            for tid,c in candidates.items():
                index=rng.choices(range(len(c['cells'])),c['weights'])[0]
                draws[tid]=models.sample(c['cells'][index],rng)
            work['draws']=draws; winner=max(draws,key=draws.get)
            selected=None if winner=='Open' else winner
        if selected:
            c=candidates[selected]; t=self.state['library'][selected]
            work.update(mode='Transform',transform=selected,version=t['version'],spec=deepcopy(t['spec']),
                **work['matched'][selected],predictions=c['cells'],model_weights=c['weights'],
                diagnostic_score=c['score'],pair=c['pairs'][0] if c['pairs'] else None)
            if work['source']!='diagnostic_replay': work['source']='ordinary_replay'
            self._advance('edit')
        else: self._advance('generate')

    def _edit(self,work):
        parent=self.programs[work['parent']]; t=self.state['library'][work['transform']]
        if t.get('template') and not work.get('repair_count'):
            try:
                code,edits=render_template(t['template'],work['symbols'],parent.code,work['bindings'])
                work.update(edits=edits,executor='template')
                self._new_program(code,dict(design=t['spec']['name'],analysis='deterministic replay'))
                return
            except (ValueError,SyntaxError,tokenize.TokenError) as exc:
                self._implementation_failure(work,'template cannot bind: '+str(exc))
                return
        required=dict(CURRENT_PARENT=node_view(parent),spec=work['spec'],bindings=work['bindings'],symbols=work['symbols'])
        if work.get('repair_count'): required['previous_failure']=work.get('repair_reason','implementation rejected')
        example=t.get('source_example')
        optional=[dict(source_diff=example['diff'])] if example else []
        reply=self._ask('edit',required,optional)
        code=apply_edits(parent.code,reply.get('edits'),work['bindings'])
        work.update(edits=reply['edits'],executor='specification')
        self._new_program(code,reply)

    def _new_program(self,code,reply):
        work=self.state['work']; parent=self.programs.get(work['parent'])
        p=Program(id=len(self.programs)+1,code=code,idea=str(reply.get('design',''))[:1500],
            analysis=str(reply.get('analysis',''))[:1500],action=work['mode'],generation=self.budget_used,
            parent_id=work['parent'],depth=parent.depth+1 if parent else 0,request_id=self.progress.model_calls,
            request=dict(mode=work['mode'],route=work['route'],transformation=work['transform'],
                version=work.get('version'),bindings=work['bindings'],source=work['source']))
        self.programs[p.id]=p; work['child']=p.id
        try: validate_source(code,self.template)
        except (ValueError,SyntaxError) as exc:
            p.status='invalid_source'; p.failure=dict(kind='invalid_source',error=str(exc))
            self._advance('audit' if work['mode']=='Transform' else 'evaluate',program=p,new=True,candidate=False)
            self._implementation_failure(work,str(exc)); return
        self._advance('audit' if work['mode']=='Transform' else 'evaluate',program=p,new=True,candidate=False)

    def _implementation_failure(self,work,reason):
        p=self.programs.get(work['child'])
        if p:
            p.status='implementation_failed'; p.failure=dict(kind='implementation_failed',error=reason)
        if work['mode']=='Transform' and work['repair_count']<1 and work.get('executor')!='template':
            work.update(repair_count=1,repair_reason=reason,child=None)
            self._advance('edit',**({'program':p,'candidate':False} if p else {}))
        else:
            work.update(outcome='implementation_failed' if work['mode']=='Transform' else 'delivery_failed',failure=reason)
            self._advance('describe',**({'program':p,'candidate':True} if p else {}))

    def _audit(self,work):
        if work.get('pending_spec') is not None:
            self._audit_discovery(work)
            return
        parent,child=self.programs[work['parent']],self.programs[work['child']]
        if work.get('executor')=='template':
            # Only a template reproducing three independently audited instances may bypass the LLM.
            work['audit']=dict(operation_ok=True,dependencies_ok=True,preservation_ok=True,parameters_ok=True,
                               summary='validated deterministic template',reason='same frozen token rule')
            child.request['audit']=work['audit']; self._advance('evaluate',program=child); return
        reply=self._ask('audit',dict(CURRENT_PARENT=dict(code=parent.code),candidate=child.code,
            actual_diff=code_diff(parent.code,child.code),spec=work['spec'],bindings=work['bindings']))
        keys=('operation_ok','dependencies_ok','preservation_ok','parameters_ok')
        if any(type(reply.get(k)) is not bool for k in keys) or not isinstance(reply.get('summary'),str):
            raise ValueError('audit needs all explicit checks')
        work['audit']=reply; child.request['audit']=reply
        if all(reply[k] for k in keys): self._advance('evaluate',program=child)
        else: self._implementation_failure(work,str(reply.get('reason','audit rejected')))

    def _evaluation_key(self,code):
        # Cache is run-local: config signature fixes source, task, dependencies and evaluation protocol.
        return digest(self.state['config_signature']+canonical(code))

    def _evaluate(self,work):
        p=self.programs[work['child']]; key=self._evaluation_key(p.code)
        receipt=self.run_dir/'evaluation_receipts'/f'{p.id}.json'
        saved=read_json(receipt); cached=self.state['evaluation_cache'].get(key)
        if saved:
            if saved['source_sha256']!=digest(p.code): raise RuntimeError('evaluation source mismatch')
            p.__dict__.update(saved['program']); rows=saved['evaluations']
            work['cache_hit']=saved.get('cache_hit',False)
        elif cached is not None:
            old=self.programs[cached]
            for k in ('status','fitness','failure','eval_seconds','eval_wall_seconds','calls','function_seconds'):
                setattr(p,k,deepcopy(getattr(old,k)))
            rows=[]; work.update(cache_hit=True,cached_program=cached)
            write_json(receipt,dict(source_sha256=digest(p.code),program=asdict(p),evaluations=[],cache_hit=True))
        else:
            rows=p.evaluate(self.training,self.template); work['cache_hit']=False
            write_json(receipt,dict(source_sha256=digest(p.code),program=asdict(p),evaluations=rows,cache_hit=False))
        self.state['evaluation_cache'].setdefault(key,p.id)
        parent=self.programs.get(work['parent'])
        work['failure']=p.failure['kind'] if p.failure else None
        if parent is None: work['outcome']='root'
        elif p.valid:
            work['delta']=parent.fitness-p.fitness
            work['outcome']='improved' if work['delta']>self.config.improvement_delta else 'no_improvement'
        else:
            # A includes task validity AND compute constraints; failures update only execution reliability.
            work['outcome']='implementation_failed'
        self._advance('describe',program=p,evaluations=rows,candidate=True)

    def _describe(self,work):
        p=self.programs.get(work['child'])
        if p:
            if work['parent'] is None:
                route=f'r{p.id}'; self.state['routes'][route]=dict(name=p.idea[:160])
            else: route=work['route']
            change=work.get('audit',{}).get('summary') or p.idea or 'see actual parent-child diff'
            p.request.update(route=route,trace=dict(change=change[:1500],lineage='',features=[]))
            work['route']=route
            self._advance('learn' if work['mode']=='Open' and work.get('outcome')=='improved'
                          and not work.get('cache_hit') else 'feedback',program=p)
        else: self._advance('feedback')

    def _learn(self,work):
        parent=self.programs[work['parent']]; child=self.programs[work['child']]
        source=lambda p: {**node_view(p),'code':ast.unparse(ast.parse(p.code))}
        required=dict(parent=source(parent),child=source(child),
            actual_diff=code_diff(source(parent)['code'],source(child)['code']),library=self._catalog())
        if work.get('induction_failure'): required['previous_spec_failure']=work['induction_failure']
        reply=self._ask('learn',required)
        kind=reply.get('kind')
        if kind=='new':
            spec=specification(reply.get('spec'),self.config.max_conditions)
            tid=next((tid for tid,t in self.state['library'].items() if t['identity']==identity(spec)),None)
        elif kind=='attach':
            tid=reply.get('id')
            if tid not in self.state['library']: raise ValueError('unknown equivalent operation')
            spec=self.state['library'][tid]['spec']
        elif kind=='none':
            work['learning_decision']=dict(kind=kind,reason=reply.get('reason'))
            self._advance('feedback'); return
        else: raise ValueError('invalid discovery decision')
        work.update(pending_spec=spec,pending_id=tid,
                    learning_decision=dict(kind=kind,reason=reply.get('reason')))
        self._advance('audit')

    def _audit_discovery(self,work):
        parent=self.programs[work['parent']]; child=self.programs[work['child']]
        reply=self._ask('audit',dict(CURRENT_PARENT=dict(code=ast.unparse(ast.parse(parent.code))),
            candidate=ast.unparse(ast.parse(child.code)),
            actual_diff=code_diff(ast.unparse(ast.parse(parent.code)),ast.unparse(ast.parse(child.code))),
            spec=work['pending_spec'],bindings=changed_regions(parent.code,child.code),
            purpose='Discovery completeness: ALL actual semantic changes must be covered by the specification.'))
        keys=('operation_ok','dependencies_ok','preservation_ok','parameters_ok')
        if any(type(reply.get(k)) is not bool for k in keys): raise ValueError('source audit needs explicit checks')
        if not all(reply[k] for k in keys):
            work['induction_failure']=dict(spec=work.pop('pending_spec'),audit=reply)
            if work.get('induction_retries',0)<1:
                work['induction_retries']=1; self._advance('learn')
            else: self._advance('feedback')
            return
        spec=work.pop('pending_spec');tid=work.get('pending_id')
        if tid is None:
            tid=f't{self.state["next_transform"]}';self.state['next_transform']+=1
            count=len(spec['conditions'])+2
            self.state['library'][tid]=dict(id=tid,version=1,identity=identity(spec),spec=spec,records=[],
                log_weights=[math.log(.5)]+[math.log(.5/(count-1))]*(count-1),examples=[],template=None,
                reference_edit=code_diff(ast.unparse(ast.parse(parent.code)),ast.unparse(ast.parse(child.code))),
                source_example=dict(parent=parent.code,child=child.code,diff=code_diff(parent.code,child.code)),
                discovery=dict(parent=parent.id,child=child.id,delta=work['delta'],attempt=work['attempt'],audit=reply))
        work['discovered_transform']=tid;child.request['discovered_transformation']=tid
        self._advance('feedback',program=child)

    def _freeze_prior(self):
        records=self.state['warm_records']
        n=len(records); valid=sum(r['A'] for r in records); positive=sum(r['A'] and r['delta']>self.config.improvement_delta for r in records)
        v=(1+valid)/(2+n); rho=(1+positive)/(2+valid)
        gains=[r['delta']-self.config.improvement_delta for r in records if r['A'] and r['delta']>self.config.improvement_delta]
        nonzero=[abs(r['delta']) for r in records if r['A'] and r['delta']]
        scale=sum(gains)/len(gains) if gains else sum(nonzero)/len(nonzero) if nonzero else 1.
        self.state.update(prior=[2*v,2*(1-v),2*rho,2*(1-rho)],scale=max(scale,1e-12),
                          prior_frozen_at=self.budget_used)

    def _feedback(self,work):
        valid=work.get('outcome') in ('improved','no_improvement')
        delta=work.get('delta',0.)
        record=dict(attempt=work['attempt'],parent=work['parent'],child=work.get('child'),
            A=valid,Y=valid and delta>self.config.improvement_delta,Z=0.,delta=delta if valid else None,
            source=work['source'],credit=not work.get('cache_hit',False),conditions=work.get('conditions',[]),
            bindings=deepcopy(work['bindings']),outcome=work.get('outcome'),executor=work.get('executor'))
        if self.state['scale']: record['Z']=max(0.,delta-self.config.improvement_delta)/self.state['scale'] if valid else 0.
        if work['mode']=='Open':
            if self.state['prior'] is None:
                if record['credit']:
                    self.state['warm_records'].append(dict(A=valid,delta=delta))
                if len(self.state['warm_records'])>=self.config.warmup: self._freeze_prior()
            elif record['credit']: self.state['open_records'].append(record)
        elif work['mode']=='Transform':
            t=self.state['library'][work['transform']]
            if record['credit']:
                t['log_weights']=[w+models.log_predictive(c,record) for w,c in zip(t['log_weights'],work['predictions'])]
                peak=max(t['log_weights']); t['log_weights']=[w-peak for w in t['log_weights']]
            t['records'].append(record)
            pair=work.get('pair')
            if pair and valid and record['credit']:
                old=t['records'][pair['record']]
                self.state['diagnostics'].append(dict(attempt=work['attempt'],transform=work['transform'],
                    parent=work['parent'],reference_parent=old['parent'],D=delta-old['delta'],
                    relation='local_epistasis' if self.programs[work['parent']].parent_id==old['parent'] else 'background_difference',
                    source=work['source'],conditions_changed=pair['changed']))
            if valid and record['credit'] and work.get('executor')=='specification':
                t['examples'].append(dict(parent=work['parent'],child=work['child'],
                    edits=deepcopy(work['edits']),symbols=deepcopy(work['symbols']),bindings=deepcopy(work['bindings'])))
                self._compile_template(t)
        self._advance('finish')

    def _compile_template(self,t):
        if t['template'] or len(t['examples'])<3: return
        examples=t['examples'][-3:]
        if len({e['parent'] for e in examples})<3: return
        try:
            candidates=[token_template(e['edits'],e['symbols']) for e in examples]
            if not all(c==candidates[0] for c in candidates): return
            for e in examples:
                code,_=render_template(candidates[0],e['symbols'],self.programs[e['parent']].code,e['bindings'])
                if canonical(code)!=canonical(self.programs[e['child']].code): return
            t.setdefault('versions',[]).append(deepcopy({k:t[k] for k in ('version','records','log_weights')}))
            t['version']+=1; t['records']=[]
            count=len(t['spec']['conditions'])+2
            t['log_weights']=[math.log(.5)]+[math.log(.5/(count-1))]*(count-1)
            t['template']=candidates[0]; t['template_validated_on']=[e['child'] for e in examples]
        except (ValueError,SyntaxError,tokenize.TokenError,IndentationError):
            return

    def _finish_attempt(self,work):
        self.state['reply']=None
        self.state['attempts'].append(deepcopy({k:v for k,v in work.items() if k not in
            ('stage','spec','edits','matched','candidates','predictions')}))
        self.state['work']=None; self._save()

    def _stage_error(self,exc):
        work=self.state['work']; stage=work['stage']
        work.setdefault('stage_errors',[]).append(dict(stage=stage,error=str(exc)))
        if stage=='learn' or stage=='audit' and work.get('pending_spec') is not None:
            work.pop('pending_spec',None); self._advance('feedback')
        elif stage=='match':
            # A malformed matching response is not an attempted transformation or performance evidence.
            work['match_index']+=work.get('current_match_batch',1)
            self._advance('match')
        else: self._implementation_failure(work,str(exc))

    def _import_library(self,path):
        payload=read_json(path)
        if not isinstance(payload,dict) or payload.get('schema')!='13.1' or payload.get('task')!=self.task:
            raise ValueError('requires a V13.1 library for the same task')
        prior=payload.get('prior');scale=payload.get('scale')
        if (not isinstance(prior,list) or len(prior)!=4 or any(type(x) not in (int,float) or
                not math.isfinite(x) or x<=0 for x in prior) or type(scale) not in (int,float) or
                not math.isfinite(scale) or scale<=0): raise ValueError('invalid frozen prior or gain scale')
        library=deepcopy(payload['library'])
        if not isinstance(library,dict): raise ValueError('invalid library')
        for tid,t in library.items():
            if tid!=t['id'] or not tid.startswith('t') or not tid[1:].isdigit(): raise ValueError('invalid transform id')
            t['spec']=specification(t['spec'],self.config.max_conditions)
            if identity(t['spec'])!=t['identity']: raise ValueError('library identity mismatch')
            if len(t['log_weights'])!=len(t['spec']['conditions'])+2 or any(
                type(w) not in (int,float) or not math.isfinite(w) for w in t['log_weights']):
                raise ValueError('model weight mismatch')
            for r in t['records']:
                if (type(r.get('A')) is not bool or type(r.get('Y')) is not bool or
                        type(r.get('credit')) is not bool or len(r['conditions'])!=len(t['spec']['conditions']) or
                        any(v is not None and type(v) is not bool for v in r['conditions']) or
                        type(r.get('Z')) not in (int,float) or not math.isfinite(r['Z']) or r['Z']<0):
                    raise ValueError('invalid replay evidence')
                r['external']=True
            t['examples']=[]
        self.state['library']=library; self.state['next_transform']=max((int(k[1:]) for k in library),default=0)+1
        self.state['import']=dict(path=str(path),sha256=digest(path.read_text()))
        # Imported normalized gains retain their frozen source scale and common prior.
        self.state['prior']=payload['prior']; self.state['scale']=payload['scale']
        if not self.state['prior'] or not self.state['scale']: raise ValueError('library has no frozen prior/scale')

    def _export_library(self):
        write_json(self.run_dir/'transformations.json',dict(schema='13.1',task=self.task,run=str(self.run_dir),
            prior=self.state['prior'],scale=self.state['scale'],library=self.state['library']))

    def _summary(self,status,error=None):
        result=super()._summary(status,error)
        n=self.state['search_rounds']
        result.update(diagnostic_opportunity_rounds=self.state['eligible_rounds'],search_rounds=n,
            diagnostic_opportunity_rate=self.state['eligible_rounds']/n if n else 0,
            diagnostic_rounds=self.state['diagnostic_rounds'],diagnostic_fallbacks=self.state['diagnostic_fallbacks'],
            paired_observations=len(self.state['diagnostics']),
            replay_coverage={tid:dict(attempts=len(t['records']),independent=sum(r['credit'] for r in t['records']),
                backgrounds=len({r['parent'] for r in t['records'] if not r.get('external')}),
                model_weights=models.weights(t['log_weights'])) for tid,t in self.state['library'].items()})
        self.facts.save_summary(result); write_json(self.run_dir/'paired_replays.json',self.state['diagnostics'])
        return result
