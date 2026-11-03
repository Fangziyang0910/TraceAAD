"""Past-only, deterministic trajectory features. Unknown semantics stay unknown."""
from __future__ import annotations
import ast
from collections import Counter
import copy
from functools import lru_cache
import hashlib
import math
import re
import statistics as st


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fitness(p):
    e=(p or {}).get('evaluation') or {}; f=e.get('fitness')
    return float(f) if e.get('status')=='success' and isinstance(f,(float,int)) and math.isfinite(f) else None


class Normalize(ast.NodeTransformer):
    def __init__(self,numbers=False): self.numbers=numbers
    def generic_visit(self,node):
        node=super().generic_visit(node)
        if isinstance(getattr(node,'body',None),list):
            node.body=[n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        return node
    def visit_Constant(self,node):
        if self.numbers and isinstance(node.value,(int,float,complex)) and not isinstance(node.value,bool):
            return ast.copy_location(ast.Constant(value='NUMBER'),node)
        return node


def compare_bags(a,b):
    keys=a.keys()|b.keys(); den=sum(max(a.get(k,0),b.get(k,0)) for k in keys)
    return 1-sum(min(a.get(k,0),b.get(k,0)) for k in keys)/den if den else 0.


def text_similarity(a,b):
    x=Counter(re.findall(r'[a-z][a-z0-9_]+',(a or '').lower()))
    y=Counter(re.findall(r'[a-z][a-z0-9_]+',(b or '').lower()))
    den=math.sqrt(sum(v*v for v in x.values())*sum(v*v for v in y.values()))
    return sum(v*y.get(k,0) for k,v in x.items())/den if den else None


def constant_value(node,env):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float,bool)): return node.value
    if isinstance(node,ast.Name): return env.get(node.id)
    if isinstance(node,ast.UnaryOp):
        v=constant_value(node.operand,env)
        if v is not None:
            if isinstance(node.op,ast.USub): return -v
            if isinstance(node.op,ast.Not): return not v
    if isinstance(node,ast.Compare) and len(node.ops)==1:
        a=constant_value(node.left,env);b=constant_value(node.comparators[0],env)
        if a is not None and b is not None:
            op=node.ops[0]
            if isinstance(op,ast.Lt): return a<b
            if isinstance(op,ast.LtE): return a<=b
            if isinstance(op,ast.Gt): return a>b
            if isinstance(op,ast.GtE): return a>=b
            if isinstance(op,ast.Eq): return a==b
            if isinstance(op,ast.NotEq): return a!=b
    return None


def projected_tree(tree,graph_n=None):
    """Only prune proven simple guards under recorded input-size assumptions."""
    tree=copy.deepcopy(tree);checks=[]
    dynamic=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'eval','exec','globals','locals','getattr','setattr'} for n in ast.walk(tree))
    for function in [n for n in tree.body if isinstance(n,ast.FunctionDef)]:
        env={}
        assignments=Counter(n.id for n in ast.walk(function) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store))
        input_name=function.args.args[0].arg if function.args.args else None
        # Input aliases or reassignments make this narrow size deduction unknown.
        shape_mutation=any(isinstance(n,ast.Attribute) and n.attr in {'resize','shape'} and isinstance(n.ctx,ast.Store) for n in ast.walk(function)) or any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='resize' for n in ast.walk(function))
        if graph_n is not None and not dynamic and not shape_mutation and function.name=='solve' and assignments[input_name]==0:
            for stmt in function.body:
                if not isinstance(stmt,ast.Assign) or len(stmt.targets)!=1: continue
                target=stmt.targets[0];expr=stmt.value
                is_shape=isinstance(expr,ast.Attribute) and isinstance(expr.value,ast.Name) and expr.value.id==input_name and expr.attr=='shape'
                if isinstance(target,(ast.Tuple,ast.List)) and is_shape:
                    for t in target.elts:
                        if isinstance(t,ast.Name) and assignments[t.id]==1: env[t.id]=graph_n
                if not isinstance(target,ast.Name) or assignments[target.id]!=1: continue
                is_len=(isinstance(expr,ast.Call) and isinstance(expr.func,ast.Name) and expr.func.id=='len' and len(expr.args)==1 and isinstance(expr.args[0],ast.Name) and expr.args[0].id==input_name)
                is_dim=(isinstance(expr,ast.Subscript) and isinstance(expr.value,ast.Attribute) and isinstance(expr.value.value,ast.Name) and expr.value.value.id==input_name and expr.value.attr=='shape' and isinstance(expr.slice,ast.Constant) and expr.slice.value in (0,1))
                if is_len or is_dim: env[target.id]=graph_n
        class Prune(ast.NodeTransformer):
            def visit_FunctionDef(self,n): return n  # Do not carry outer bindings into nested scopes.
            def visit_If(self,n):
                v=constant_value(n.test,env)
                if isinstance(v,bool):
                    checks.append({'line':n.lineno,'condition':ast.unparse(n.test),'value':v,'bindings':dict(env)})
                    out=[]
                    for s in (n.body if v else n.orelse):
                        q=self.visit(s)
                        out.extend(q if isinstance(q,list) else [q])
                    return out
                return self.generic_visit(n)
        out=[]
        for stmt in function.body:
            val=Prune().visit(stmt)
            out.extend(val if isinstance(val,list) else [val])
        function.body=out
        if not dynamic:
            defs={n.name:n for n in function.body if isinstance(n,ast.FunctionDef)}
            main=[n for n in function.body if not isinstance(n,ast.FunctionDef)]
            used={n.id for s in main for n in ast.walk(s) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
            while True:
                expanded=used|{n.id for name in used & defs.keys() for n in ast.walk(defs[name]) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
                if expanded==used: break
                used=expanded
            def safely_discardable(n):
                args=n.args.posonlyargs+n.args.args+n.args.kwonlyargs
                defaults=n.args.defaults+[v for v in n.args.kw_defaults if v is not None]
                return not n.decorator_list and n.returns is None and all(a.annotation is None for a in args) and all(isinstance(v,ast.Constant) for v in defaults)
            function.body=[n for n in function.body if not isinstance(n,ast.FunctionDef) or n.name in used or not safely_discardable(n)]
    return tree,checks,dynamic


@lru_cache(maxsize=16000)
def inspect_code(code,graph_n=None):
    try: tree=Normalize().visit(ast.parse(code))
    except (SyntaxError,RecursionError): return {'parse_ok':False,'hash':digest(code),'shape':None,'projection':None,'checks':[],'nodes':0,'subtrees':Counter(),'calls':Counter(),'return_identifiers':set()}
    nodes=list(ast.walk(tree))
    bag=Counter(type(n).__name__+':'+','.join(type(c).__name__ for c in ast.iter_child_nodes(n)) for n in nodes)
    calls=Counter(ast.unparse(n.func) for n in nodes if isinstance(n,ast.Call))
    # Syntactic dependency closure only; mutations/aliases prevent semantic guarantees.
    deps={};ret=set()
    for n in nodes:
        if isinstance(n,ast.Assign):
            inputs={x.id for x in ast.walk(n.value) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
            for target in n.targets:
                for x in ast.walk(target):
                    if isinstance(x,ast.Name):deps.setdefault(x.id,set()).update(inputs)
        if isinstance(n,ast.Return):ret|={x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
    while True:
        new=ret|set().union(*(deps.get(k,set()) for k in ret)) if ret else set()
        if new==ret: break
        ret=new
    projected,checks,dynamic=projected_tree(tree,graph_n)
    return {'parse_ok':True,'hash':digest(ast.dump(tree)),'shape':digest(ast.dump(Normalize(True).visit(copy.deepcopy(tree)))),
        'projection':digest(ast.dump(projected)),'checks':checks,'dynamic_unknown':dynamic,'nodes':len(nodes),
        'subtrees':bag,'calls':calls,'return_identifiers':ret,
        'loops':sum(isinstance(n,(ast.For,ast.While,ast.comprehension)) for n in nodes)}


def edge_features(parent,child,graph_n=None):
    p=inspect_code(parent.get('code',''),graph_n);c=inspect_code(child.get('code',''),graph_n)
    known=p['parse_ok'] and c['parse_ok']
    same=known and p['hash']==c['hash']
    projected=known and p['projection']==c['projection']
    return {'same_ast':bool(same),'numeric_only':bool(known and not same and p['shape']==c['shape']),
        'known_inactive_only':bool(known and not same and projected and (p['checks'] or c['checks'])),
        'semantic_relation':'same_normalized_ast' if same else 'same_under_recorded_guards' if projected and known else 'unknown',
        'ast_distance':compare_bags(p['subtrees'],c['subtrees']) if known else None,
        'node_retention':min(1,c['nodes']/p['nodes']) if known and p['nodes'] else None,
        'call_retention':sum((p['calls']&c['calls']).values())/sum(p['calls'].values()) if p['calls'] else None,
        'return_identifier_retention':len(p['return_identifiers']&c['return_identifiers'])/len(p['return_identifiers']) if p['return_identifiers'] else None,
        'design_similarity':text_similarity(parent.get('design'),child.get('design')),
        'analysis_similarity':text_similarity(parent.get('analysis'),child.get('analysis')),
        'guard_checks':c['checks']}


def dynamics(values):
    out={'observations':len(values)}
    for w in (2,4,8):
        v=values[-w:]; q=[-x for x in v]
        if len(v)<2:
            out[f'slope_{w}']=None;continue
        slope=st.median((q[j]-q[i])/(j-i) for i in range(len(q)) for j in range(i+1,len(q)))
        residual=[x-i*slope for i,x in enumerate(q)];center=st.median(residual)
        out.update({f'slope_{w}':slope,f'residual_mad_{w}':st.median(abs(x-center) for x in residual),
            f'range_{w}':max(v)-min(v),f'max_drawdown_{w}':max(max(q[:i+1])-x for i,x in enumerate(q))})
    out['short_long_slope_difference']=(out.get('slope_2')-out.get('slope_8')) if len(values)>=2 else None
    out['trailing_flat']=0
    for x,y in zip(values[-2::-1],values[:0:-1]):
        if abs(x-y)>1e-9:break
        out['trailing_flat']+=1
    out['steps_since_path_best']=len(values)-1-min(range(len(values)),key=lambda i:(values[i],-i)) if values else None
    return out


def node_features(pid,archive,attempts,metadata,versions):
    p=archive[pid]; pf=fitness(p)
    graph_n=metadata.get('problem_size') if metadata.get('name')=='graph_colouring' else None
    ids=[];current=pid
    while current and current in archive and len(ids)<9:
        ids.append(current);current=archive[current]['parent_id']
    ids.reverse()
    values=[fitness(archive[i]) for i in ids if fitness(archive[i]) is not None]
    birth=next((a for a in attempts if a.get('program_id')==pid),None)
    parent=archive.get(p.get('parent_id')); parent_f=fitness(parent)
    edge=edge_features(parent,p,graph_n) if parent else {}
    trials=[a for a in attempts if a['parent_id']==pid]
    coverage={}
    for op in ['all','Refine','Explore','Crossover']:
        subset=[a for a in trials if op=='all' or a['operator']==op]
        edges=[edge_features(p,archive[a['program_id']],graph_n) for a in subset if a.get('program_id') in archive]
        valid=[a for a in subset if fitness(archive.get(a.get('program_id'))) is not None]
        gains=[pf-fitness(archive[a['program_id']]) for a in valid]
        useful=[e for e in edges if not e['same_ast'] and not e['known_inactive_only']]
        hist=[inspect_code(archive[a['program_id']]['code'],graph_n) for a in subset if a.get('program_id') in archive]
        # Effective diversity in normalized shallow-subtree count space; descriptive, not sample size.
        hs=hist[-32:]
        norm=[math.sqrt(sum(v*v for v in h['subtrees'].values())) or 1 for h in hs]
        sumk2=0.
        for i,h in enumerate(hs):
            for j,g in enumerate(hs):
                dot=sum(v*g['subtrees'].get(k,0) for k,v in h['subtrees'].items())/(norm[i]*norm[j])
                sumk2+=math.exp(-4*max(0,2-2*dot))
        coverage[op]={'attempts':len(subset),'valid':len(valid),'improving':sum(v>1e-9 for v in gains),
            'same_ast':sum(e['same_ast'] for e in edges),'known_inactive_only':sum(e['known_inactive_only'] for e in edges),
            'numeric_only':sum(e['numeric_only'] for e in edges),'different_but_semantics_unknown':len(useful),
            'distinct_shapes':len({h['shape'] for h in hist if h['shape']}),
            'effective_subtree_diversity':len(hs)**2/sumk2 if sumk2 else None,
            'positive_gain_max':max([0.]+gains),'eval_cost':sum(len(a['evaluations']) for a in subset),
            'request_cost':sum(len(a['calls']) for a in subset),
            'repair_attempts':sum(a['repair_used'] for a in subset)}
    birth_evals=(birth or {}).get('evaluations',[])
    repair_edge=None
    if len(birth_evals)>1 and birth_evals[0]['id'] in versions:
        initial=versions[birth_evals[0]['id']].get('candidate')
        if initial:repair_edge=edge_features(initial,p,graph_n)
    last=birth_evals[-1] if birth_evals else {}
    return {'node_id':pid,'fitness':pf,'ancestor_ids':ids,'dynamics':dynamics(values),'coverage':coverage,
        'formation':{'operator':p.get('operator'),'parent_fitness':parent_f,'fitness_delta':pf-parent_f if parent_f is not None else None,
                     'repair_used':bool((birth or {}).get('repair_used')),**edge},
        'repair_change':repair_edge,
        'cost':{'last_evaluation_seconds':last.get('seconds'),'evaluation_limit_seconds':metadata.get('settings',{}).get('timeout_seconds'),
                'birth_requests':len((birth or {}).get('calls',[])),'birth_evaluations':len(birth_evals)},
        'assumptions':{'graph_n':graph_n,'semantic_unknown_is_not_failure':True}}
