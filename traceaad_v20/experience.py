"""Trajectory events, reviews and experience notes learned by reflection."""
import difflib
import json

from .embedding import INSTRUCT
from .representation import identity,structural,cosine,ranks,rrf,entry_code
from .prompts import failure_summary
from .selection import path_best

# One standard for scoring and experience: an attempt improves only when it beats
# the best score on its parent's trajectory (the parent and its nearest ancestors).
RESULT_TEXT={'improved':'better than the trajectory best','recovered':'better than the parent, not better than the trajectory best',
             'tied':'equal to the parent','regressed':'worse than the parent'}

UPDATE_FIELDS=['action','experience_id','lesson','applicable_when','new_source_event_ids']
REFLECTION_SCHEMA={'type':'json_schema','json_schema':{'name':'trajectory_review','strict':True,'schema':{
 'type':'object','additionalProperties':False,'required':['analysis','review','updates'],'properties':{
 'analysis':{'type':'string'},'review':{'type':'string'},'updates':{'type':'array','items':{
 'type':'object','additionalProperties':False,'required':UPDATE_FIELDS,'properties':{
 'action':{'type':'string','enum':['create','revise','retire']},'experience_id':{'type':['integer','null']},
 'lesson':{'type':'string'},'applicable_when':{'type':'string'},
 'new_source_event_ids':{'type':'array','items':{'type':'string'}}}}}}}}}
SEGMENT_LIMIT=12


class Experience:
    def __init__(self,task,embedder,config):
        self.task,self.embedder,self.config=task,embedder,config
        self.data={'events':{},'reviews':{},'notes':{},'processed':[],'next_note':1,'segments':[]}
        self.vectors={}
        self.embedding={'requests':0,'errors':0,'usage':{}}
        self.formation=lambda archive,parent:[]
    def vector(self,text,channel='document'):
        if not text or not text.strip() or self.embedder is None:return None
        query=f'Instruct: {INSTRUCT[channel]}\nQuery: {text}' if channel in INSTRUCT else text
        if query in self.vectors:return self.vectors[query]
        self.embedding['requests']+=1
        try:vector=self.embedder.encode(query)
        except Exception:
            self.embedding['errors']+=1;return None
        for k,v in (getattr(self.embedder,'last_usage',None) or {}).items():
            if isinstance(v,(int,float)):self.embedding['usage'][k]=self.embedding['usage'].get(k,0)+v
        self.vectors[query]=vector
        return vector
    def drain(self):
        """Embedding requests since the last call, as one usage row."""
        row={'role':'embedding','status':'error' if self.embedding['errors'] else 'success',**self.embedding}
        self.embedding={'requests':0,'errors':0,'usage':{}}
        return row
    def observe(self,attempt,archive):
        if attempt.operator=='Init' or attempt.program_id is None:return None
        eid=f'A{attempt.id}'
        parent=archive[attempt.parent_id];child=archive[attempt.program_id]
        pf=parent.evaluation.fitness;cf=child.evaluation.fitness if child.valid else None
        reference=attempt.record.get('selection',{}).get('selected',{}).get('T')
        if reference is None:reference=path_best(archive,attempt.parent_id,self.config.ancestor_count)
        if cf is None:result=child.evaluation.status
        else:result='improved' if cf<reference else 'recovered' if cf<pf else 'tied' if cf==pf else 'regressed'
        self.data['events'][eid]={'event_id':eid,'attempt_id':attempt.id,'parent_id':attempt.parent_id,'child_id':child.id,
            'operator':attempt.operator,'reference_id':attempt.reference_id,'status':child.evaluation.status,
            'parent_fitness':pf,'child_fitness':cf,'reference_fitness':reference,'result':result,
            'best_before':attempt.record['context']['best_before'],
            'parent_code':parent.code,'parent_design':parent.design,'code':child.code,'code_identity':identity(child.code),
            'analysis':child.analysis,'design':child.design,
            'failure':failure_summary(child.evaluation.failure or ''),'structural':structural(parent.code)}
        return eid
    def card(self,eid):
        e=self.data['events'][eid]
        text=f"Event {eid}; Algorithm {e['child_id']}; Parent {e['parent_id']}; Operator {e['operator']}\nResult: {RESULT_TEXT.get(e['result'],e['result'])}"
        if e['child_fitness'] is not None:
            text+=f"; parent {e['parent_fitness']} -> {e['child_fitness']}"
            if e.get('reference_fitness') is not None:text+=f"; trajectory best {e['reference_fitness']}"
        else:text+='; '+e['failure']
        review=self.data['reviews'].get(eid)
        return text+f"\nDesign: {e['design']}\n"+('Reviewed change: '+review if review else 'Proposed change: '+str(e['analysis']))
    def note_view(self,note):
        sources=[self.data['events'][i] for i in note['source_event_ids'] if i in self.data['events']]
        selected=[]
        for result,limit in [({'improved'},2),({'recovered','tied','regressed'},2),({'error','timeout'},1)]:
            selected += [e['event_id'] for e in sources if e['result'] in result][-limit:]
        cards=[self.card(i) for i in dict.fromkeys(selected)]
        for segment in reversed(self.data['segments']):
            if any(i in note['source_event_ids'] for i in segment):
                cards.append('Related parent_id sequence:\n'+'\n'.join(self.card(i) for i in segment));break
        return {'id':note['id'],'lesson':note['lesson'],
            'applicable_when':note['applicable_when'],'observations':cards,
            'source_event_ids':note['source_event_ids'],'source_count':len(sources),'visible_event_ids':list(dict.fromkeys(selected))}
    def retrieve(self,parent,need,*,exclude=()):
        notes=[n for n in self.data['notes'].values() if n['active'] and n['id'] not in exclude]
        if not notes:return [],{'channels':{},'rankings':{}}
        state_query=self.vector((parent.design or '')+'\n'+self.review_for(parent.id),'State')
        need_query=self.vector(need,'Need');code=entry_code(parent.code,self.task.entrypoint)
        implementation_query=self.vector(code,'Implementation') if code else None
        ps=structural(parent.code);representatives={};scores={k:{} for k in ['conditions','source_state','need','code','ast']}
        for note in notes:
            nid=note['id'];sources=[self.data['events'][i] for i in note['source_event_ids'] if i in self.data['events']]
            reps=[]
            for good in (True,False):reps += [e for e in sources if (e['result']=='improved')==good][-2:]
            unique={}
            for e in reps:unique.setdefault(identity(e['parent_code']),e)
            reps=list(unique.values())
            representatives[nid]=[{'event_id':e['event_id'],'role':'source_parent','result':e['result']} for e in reps]
            def best(values):
                values=[v for v in values if v is not None];return max(values) if values else None
            scores['conditions'][nid]=cosine(state_query,self.vector(note['applicable_when']))
            scores['source_state'][nid]=best([cosine(state_query,self.vector(e['parent_design'])) for e in reps])
            scores['need'][nid]=best([cosine(need_query,self.vector(note['lesson']))]+[
                cosine(need_query,self.vector((self.data['reviews'].get(e['event_id']) or e['analysis'] or '')+' '+e['result'])) for e in reps])
            scores['code'][nid]=best([cosine(implementation_query,self.vector(entry_code(e['parent_code'],self.task.entrypoint))) for e in reps if entry_code(e['parent_code'],self.task.entrypoint)])
            scores['ast'][nid]=best([cosine(ps,e['structural']) for e in reps])
        ids=[n['id'] for n in notes];routes={k:ranks(v) for k,v in scores.items()}
        state,nstate=rrf([routes['conditions'],routes['source_state']],ids)
        implementation,nimpl=rrf([routes['code'],routes['ast']],ids)
        outer={'State':ranks(state) if nstate else None,'Need':routes['need'],
               'Implementation':ranks(implementation) if nimpl else None}
        total,nchannels=rrf(list(outer.values()),ids)
        complete={k:all(v is not None for v in vals.values()) for k,vals in scores.items()}
        available={'State':complete['conditions'] or complete['source_state'],'Need':complete['need'],'Implementation':complete['code'] or complete['ast']}
        comparable=any(all(v is not None for v in route.values()) for route in scores.values())
        if not nchannels and not comparable:return [],{'channels':available,'active_rankings':{k:v is not None for k,v in outer.items()},'rankings':outer,'scores':scores,'representatives':representatives}
        # Equal rankings keep every note eligible; no similarity cutoff.
        order=sorted(ids,key=lambda i:(-total[i],i))[:3]
        return [self.note_view(self.data['notes'][str(i)]) for i in order],{'channels':available,'active_rankings':{k:v is not None for k,v in outer.items()},'rankings':outer,'order':order,'scores':scores,'representatives':representatives}
    def review_for(self,pid):
        ids=[e['event_id'] for e in self.data['events'].values() if e['child_id']==pid]
        return self.data['reviews'].get(ids[-1],'') if ids else ''
    def segment(self,latest,attempts):
        """Choose the trajectory segment for a learning reflection, or none."""
        events=self.data['events'];processed=set(self.data['processed'])
        pending=[e for i,e in events.items() if i not in processed]
        options=[]
        for origin in events.values():
            if origin['operator'] not in ('Explore','Crossover'):continue
            follow=[e for e in pending if e['parent_id']==origin['child_id']]
            if len(follow)>=2:options.append((0,min(e['attempt_id'] for e in follow),[origin]+follow))
        grouped={}
        for e in pending:grouped.setdefault(e['parent_id'],[]).append(e)
        for es in grouped.values():
            if len({(e['code_identity'],e['status'],e['child_fitness']) for e in es})>=3:
                options.append((1,min(e['attempt_id'] for e in es),es))
        e=events[latest]
        if e['child_fitness'] is not None and e['child_fitness']<e['best_before']:options.append((2,e['attempt_id'],[e]))
        if attempts%10==0 and pending:
            oldest=min(pending,key=lambda e:e['attempt_id']);options.append((3,oldest['attempt_id'],grouped[oldest['parent_id']]))
        if not options:return []
        es=sorted(min(options,key=lambda o:(o[0],o[1]))[2],key=lambda e:e['attempt_id'])
        if len(es)<=SEGMENT_LIMIT:return [e['event_id'] for e in es]
        scored=[e for e in es if e['child_fitness'] is not None]
        required=[es[-1]]
        if scored:required+=[min(scored,key=lambda e:e['child_fitness']),max(scored,key=lambda e:e['child_fitness'])]
        required+=[e for e in es if e['child_fitness'] is None][-1:]
        if es[0]['operator'] in ('Explore','Crossover'):required.append(es[0])
        picked=dict.fromkeys(e['event_id'] for e in required)
        for e in reversed(es):
            if len(picked)>=SEGMENT_LIMIT:break
            picked.setdefault(e['event_id'])
        return sorted(picked,key=lambda i:events[i]['attempt_id'])
    def apply(self,raw,latest,segment,shown):
        """Store the review; in a learning reflection also apply the note updates."""
        try:response=json.loads(raw)
        except ValueError:return False
        if response.get('review'):self.data['reviews'][latest]=response['review']
        if not segment:return True
        visible={n['id'] for n in shown}
        for u in response.get('updates',[]):
            sources=[i for i in u.get('new_source_event_ids',[]) if i in self.data['events']]
            if u.get('action')=='create':
                if not sources:continue
                nid=self.data['next_note'];self.data['next_note']+=1
                self.data['notes'][str(nid)]={'id':nid,'active':True,'lesson':u['lesson'],'applicable_when':u['applicable_when'],
                    'source_event_ids':list(dict.fromkeys(sources)),'history':[]}
                continue
            note=self.data['notes'].get(str(u.get('experience_id')))
            if note is None or not note['active'] or note['id'] not in visible:continue
            note['history'].append({k:v for k,v in note.items() if k!='history'})
            if u['action']=='retire':note['active']=False
            else:note.update(lesson=u['lesson'],applicable_when=u['applicable_when'],
                source_event_ids=list(dict.fromkeys(note['source_event_ids']+sources)))
        self.data['processed']=list(dict.fromkeys(self.data['processed']+segment))
        self.data['segments'].append(list(segment))
        return True
    def reflect(self,attempt,archive,call):
        """Review the latest change; when a segment is ready, also learn notes from it."""
        latest=self.observe(attempt,archive)
        if latest is None:return
        event=self.data['events'][latest]
        segment=self.segment(latest,attempt.id)
        shown=list(attempt.record['context'].get('experiences',[]))
        ancestors=[]
        if segment:
            root=self.data['events'][segment[0]];parent=archive[root['parent_id']]
            extra,_=self.retrieve(parent,self.card(segment[-1]),exclude=[n['id'] for n in shown])
            shown+=extra
            for p in self.formation(archive,parent):
                ancestors.append(f'Algorithm {p.id}; parent_id={p.parent_id}; Operator {p.operator}; Score {p.evaluation.fitness if p.valid else p.evaluation.status}; Design {p.design}; Change {self.review_for(p.id) or p.analysis}')
        def source(e):
            delta='\n'.join(difflib.unified_diff(e['parent_code'].splitlines(),e['code'].splitlines(),n=3))
            content=delta if len(delta)<len(e['code']) else e['code']
            return 'Parent complete code:\n'+e['parent_code']+'\nEvaluated change:\n'+content
        text=f'{self.task.task_description}\nFunction:\n{self.task.function_template}\nScore: {self.task.score_definition}\n[Latest Evaluated Change]\n{self.card(latest)}\n{source(event)}'
        if segment:
            text+='\n[Learning Segment]\nParent complete code:\n'+root['parent_code']+'\n'+'\n'.join(ancestors)+'\n'+'\n'.join(self.card(i) for i in segment if i!=latest)
            text+='\n[Experience]\n'+json.dumps(shown,ensure_ascii=False)
            for i in dict.fromkeys(i for n in shown for i in n['visible_event_ids']):
                if i!=latest:text+='\n[Source material '+i+']\n'+source(self.data['events'][i])
        text+='\n[Instruction]\nBriefly describe the actual changed computation and its role in the returned result. Separate observed outcomes from explanations. '
        text+=('Compare visible same-parent attempts and parent_id relations. Propose or revise a short reusable lesson only when useful. Cite only visible events. Preserve observations. ' if segment else 'Use local-only mode: updates must be empty. ')
        text+='Return exactly one JSON object with string analysis, string review, and updates array. Each update has action(create/revise/retire), experience_id, lesson, applicable_when, new_source_event_ids. Use null for an unused id. Empty review or updates are valid.'
        raw=call(text,'learning_reflection' if segment else 'local_reflection')
        accepted=raw is not None and self.apply(raw,latest,segment,shown)
        attempt.record['reflection']={'segment':segment,'notes':[n['id'] for n in shown],'accepted':accepted}
