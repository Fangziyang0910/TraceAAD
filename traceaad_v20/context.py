"""Deterministic trajectory views and random choice from a fused reference shortlist."""
import json
from .representation import identity,structural,cosine,ranks,rrf,changes
from .prompts import PromptBuilder,failure_summary

COMMON='Develop a candidate to improve solution quality for the stated task. Read the current code as the implementation being modified. Historical designs, reviews, and experience notes are interpretations or proposals; evaluated results report outcomes, not the cause of those outcomes.'
REFINE='Refine the current decision mechanism. Identify a specific limitation or opportunity and change the computation that addresses it, keeping the main mechanism as the basis of the new method. Make coordinated changes where the improvement requires them, and implement the complete function.'
EXPLORE='Develop an alternative decision mechanism for the required function. Introduce or reorganize a computation that changes how the method uses the available information to make its decisions, and explain that difference from the current method. The alternative need not retain the current representation or computation pattern. Implement the complete function.'
CROSSOVER='Adapt a useful computation from the reference to work with the current method. Identify the decision it will change and how the two sources contribute to the resulting procedure, adjusting their representations or order of computation where needed. Implement the complete function with that interaction in its actual decision process. If the full reference code does not offer a coherent adaptation, briefly say that the reference was not used and implement your proposed candidate without claiming a fusion.'
GUIDANCE={
    'ancestors':'Use the formation path to understand earlier changes to the current computation.',
    'children':'Compare the modifications already tried from this parent and their outcomes. A useful next proposal may extend an improvement, revise an unsuccessful implementation, or test a different combination.',
    'failed':'Use execution feedback to address the reported implementation issue; it does not establish that every realization of the idea is unsuitable.',
    'experience':'Use a note when its conditions hold or when your proposed change can establish them. Adapt the idea to the current computation; leave it unused when it adds no useful guidance. A note can also inform further development of a computation already present.'}
DUTY='In analysis, briefly identify the computation you will change and why the change is worth trying. Cite an Algorithm or Experience id only when a specific observation informed that choice; a proposal may also follow from the task and current code without a historical precedent. Describe expected benefits as expectations, not measured improvements.'
SECTIONS=[('ancestors',('ancestors',),'[Ancestor History]'),('children',('children_best','children'),'[Evaluated Offspring]'),
          ('failed',('failed',),'[Failed Offspring]'),('experience',('experience',),'[Relevant Experience]')]
OPERATORS={'Refine':REFINE,'Explore':EXPLORE,'Crossover':CROSSOVER}


class Context:
    def __init__(self,task,config,experience):self.task,self.config,self.experience=task,config,experience
    def ancestors(self,archive,parent):
        path=[];pid=parent.parent_id
        while pid not in (None,0):
            p=archive[pid];path.append(p);pid=p.parent_id
        recent=path[:3];far=path[3:];ids=[p.id for p in far]
        distances={};gains={}
        for p in far:
            before=archive.get(p.parent_id);distances[p.id]=None if before is None else 1-cosine(structural(p.code),structural(before.code)) if cosine(structural(p.code),structural(before.code)) is not None else None
            gains[p.id]=abs(p.evaluation.fitness-before.evaluation.fitness) if p.valid and before and before.valid else None
        fused,_=rrf([ranks(distances),ranks(gains)],ids)
        chosen={p.id for p in recent}|set(sorted(ids,key=lambda i:(-fused[i],ids.index(i)))[:5])
        return [p for p in reversed(path) if p.id in chosen],path
    def cover(self,candidates,selected,number):
        while candidates and len(selected)<number:
            ids=[p.id for p in candidates];rv={};sv={}
            all_programs=candidates+selected
            vec={p.id:self.experience.vector(self.experience.review_for(p.id) or p.design or '') for p in all_programs}
            ast={p.id:structural(p.code) for p in all_programs}
            for p in candidates:
                rv[p.id]=min((1-cosine(vec[p.id],vec[q.id]) for q in selected),default=0) if all(vec[q.id] is not None for q in all_programs) else None
                sv[p.id]=min((1-cosine(ast[p.id],ast[q.id]) for q in selected),default=0) if all(ast[q.id] is not None for q in all_programs) else None
            fused,n=rrf([ranks(rv),ranks(sv)],ids)
            winner=min(candidates,key=lambda p:(-fused[p.id],-p.id))
            selected.append(winner);candidates=[p for p in candidates if p.id!=winner.id]
        return selected
    def histories(self,archive,parent,operator):
        if not self.config.trajectory_context:return PromptBuilder(self.task,self.config).histories(archive,parent),{}
        ancestors,path=self.ancestors(archive,parent)
        unique={}
        for p in sorted(archive.values(),key=lambda p:p.id):
            if p.parent_id==parent.id and p.evaluation is not None:
                key=(identity(p.code),p.evaluation.status,p.evaluation.fitness,failure_summary(p.evaluation.failure or ''))
                unique[key]=p
        valid=[p for p in unique.values() if p.valid];failed=[p for p in unique.values() if not p.valid]
        selected=[]
        if valid:selected=[min(valid,key=lambda p:(p.evaluation.fitness,-p.id))]
        remaining=[p for p in valid if p not in selected]
        if operator=='Refine' and remaining:
            latest=max(remaining,key=lambda p:p.id);selected.append(latest);remaining.remove(latest)
        selected=self.cover(remaining,selected,3)
        classes={}
        for p in sorted(failed,key=lambda p:-p.id):classes.setdefault((p.evaluation.status,failure_summary(p.evaluation.failure or '').split(' at ')[0]),[]).append(p)
        failures=[rows[0] for rows in classes.values()][:2]
        if len(failures)==1:
            rest=[p for p in failed if p.id!=failures[0].id]
            if rest:failures.append(max(rest,key=lambda p:p.id))
        return (ancestors,sorted(selected,key=lambda p:p.id),sorted(failures,key=lambda p:p.id)),{'ancestor_path':[p.id for p in path],
            'ancestors':[p.id for p in ancestors],'children':[p.id for p in selected],'failed':[p.id for p in failures],
            'total_valid':sum(p.parent_id==parent.id and p.valid for p in archive.values()),'total_failed':sum(p.parent_id==parent.id and p.evaluation is not None and not p.valid for p in archive.values())}
    def reference(self,archive,parent,rng):
        reps={}
        for p in sorted((p for p in archive.values() if p.valid),key=lambda p:(p.evaluation.fitness,p.id)):reps.setdefault(identity(p.code),p)
        all_reps=list(reps.values());median=sorted(p.evaluation.fitness for p in all_reps)
        median=(median[(len(median)-1)//2]+median[len(median)//2])/2
        candidates=[p for key,p in reps.items() if key!=identity(parent.code)]
        if not candidates:return None,{'reason':'no_distinct_reference'}
        ids=[p.id for p in candidates];pool=[p for p in candidates if p.evaluation.fitness<=median];relaxed=not pool
        if not pool:pool=candidates
        outside=[p for p in candidates if p not in pool]
        pv=self.experience.vector(parent.design or '');ps=structural(parent.code)
        ds={p.id:1-cosine(pv,self.experience.vector(p.design or '')) if pv is not None and self.experience.vector(p.design or '') is not None else None for p in candidates}
        ss={p.id:1-cosine(ps,structural(p.code)) if cosine(ps,structural(p.code)) is not None else None for p in candidates}
        diff,nd=rrf([ranks(ds),ranks(ss)],ids)
        quality=ranks({p.id:-p.evaluation.fitness for p in candidates});difference=ranks(diff) if nd else None
        outer=[r for r in (quality,difference) if r is not None];total,_=rrf(outer,ids)
        poolids={p.id for p in pool};chosen=[]
        for route in outer:
            winner=min(poolids,key=lambda i:(route[i],i))
            if winner not in chosen:chosen.append(winner)
        limit=3 if outside else 4
        for i in sorted(poolids,key=lambda i:(-total[i],i)):
            if len(chosen)>=limit:break
            if i not in chosen:chosen.append(i)
        if outside:
            extra=min(outside,key=lambda p:(-diff[p.id],p.evaluation.fitness,p.id));chosen.append(extra.id)
        chosen=chosen[:4];selected=rng.choice(chosen)
        return archive[selected],{'shortlist':chosen,'quality_pool':sorted(poolids),'relaxed':relaxed,
            'ranks':outer,'selected':selected}
    def card(self,archive,p,*,current=False):
        parent=archive.get(p.parent_id);lines=[f'Algorithm {p.id}',f'Parent: Algorithm {p.parent_id}',f'Operator: {p.operator}']
        if p.reference_id is not None:lines.append(f'Reference: Algorithm {p.reference_id}')
        if p.valid:
            if parent and parent.valid:
                a,b=parent.evaluation.fitness,p.evaluation.fitness;result='improved' if b<a else 'tied' if b==a else 'regressed'
                lines.append(f'Result relative to Algorithm {parent.id}: {a} -> {b}; {result}')
            else:lines.append(f'Score: {p.evaluation.fitness}')
        else:lines.append('Result: '+p.evaluation.status+'; '+failure_summary(p.evaluation.failure or ''))
        lines.append(f'Design: {p.design}')
        review=self.experience.review_for(p.id)
        lines.append(('Origin proposal: '+str(p.analysis)) if current else ('Reviewed change: '+review if review else 'Proposed change: '+str(p.analysis)))
        if current:lines.append('```python\n'+p.code+'\n```')
        elif parent:
            unit=changes(parent.code,p.code)
            if unit['shown']:lines.append('Changed computation: '+json.dumps(unit))
        return '\n'.join(lines)
    def render(self,archive,parent,operator,reference,notes):
        """Return header-free records; assemble() adds one header per section."""
        (ancestors,children,failed),view=self.histories(archive,parent,operator)
        sections=[]
        if ancestors:
            full=list(reversed(view.get('ancestor_path',[])))
            previous=None
            for p in ancestors:
                index=full.index(p.id) if p.id in full else 0
                omitted=max(0,index-previous-1) if previous is not None else index
                sections.append(('ancestors',self.card(archive,p)+f'\nOmitted intermediate nodes before this record: {omitted}; actual parent_id: {p.parent_id}'))
                previous=index
        best_child=min(children,key=lambda p:(p.evaluation.fitness,-p.id)) if children else None
        for p in children:sections.append(('children_best' if p is best_child else 'children',self.card(archive,p)))
        for p in failed:sections.append(('failed',self.card(archive,p)))
        for n in notes:sections.append(('experience',json.dumps(n,ensure_ascii=False)))
        required=[]
        if reference:required.append('[Reference Algorithm]\n'+self.card(archive,reference,current=True))
        required.append('[Current Algorithm]\n'+self.card(archive,parent,current=True))
        view['parent_id']=parent.id
        return sections,required,view
    @staticmethod
    def assemble(sections,view):
        blocks=[]
        for name,kinds,header in SECTIONS:
            records=[text for kind,text in sections if kind in kinds]
            if not records:continue
            if name=='children':header+=f'\nDirect attempts from Algorithm {view.get("parent_id")}: {view.get("total_valid",len(records))} valid; showing {len(records)}.'
            if name=='failed':header+=f'\nFailed direct attempts from Algorithm {view.get("parent_id")}: {view.get("total_failed",len(records))}; showing {len(records)}.'
            blocks.append(header+'\n\n'+'\n\n'.join(records))
        return '\n\n'.join(blocks)
    @staticmethod
    def instruction(operator,sections):
        present={kind for kind,_ in sections}
        parts=[COMMON,OPERATORS[operator]]
        for name,kinds,_ in SECTIONS:
            if present & set(kinds):parts.append(GUIDANCE[name])
        parts.append(DUTY)
        return '\n'.join(parts)
