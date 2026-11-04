"""Source identity, AST structure, changed units and rank fusion."""
import ast
from collections import Counter
import hashlib
import math


def identity(code):
    try:
        tree=ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
                first=node.body[0]
                if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):node.body=node.body[1:]
        text=ast.dump(tree,include_attributes=False)
    except (SyntaxError,ValueError):text=code.strip()
    return hashlib.sha256(text.encode()).hexdigest()


def structural(code):
    try:tree=ast.parse(code)
    except (SyntaxError,ValueError):return None
    counts=Counter()
    def visit(node):
        children=list(ast.iter_child_nodes(node))
        for child in children:
            counts[type(node).__name__+'/'+type(child).__name__]+=1
            for grand in ast.iter_child_nodes(child):counts[type(node).__name__+'/'+type(child).__name__+'/'+type(grand).__name__]+=1
            visit(child)
    visit(tree)
    transformed={k:math.log1p(v) for k,v in counts.items()};norm=math.sqrt(sum(v*v for v in transformed.values()))
    return {k:v/norm for k,v in transformed.items()} if norm else None


def cosine(a,b):
    if a is None or b is None:return None
    if isinstance(a,dict):return sum(v*b.get(k,0.) for k,v in a.items())
    if len(a)!=len(b):return None
    return sum(x*y for x,y in zip(a,b))


def ranks(scores, *, reverse=True):
    if not scores or any(v is None for v in scores.values()) or len(set(scores.values()))<2:return None
    ordered=sorted(scores,key=lambda k:((-scores[k] if reverse else scores[k]),k));out={};last=object();rank=0
    for i,k in enumerate(ordered,1):
        if scores[k]!=last:rank=i;last=scores[k]
        out[k]=rank
    return out


def rrf(rankings, ids):
    active=[r for r in rankings if r is not None and set(r)==set(ids)]
    return {i:sum(1/(60+r[i]) for r in active) for i in ids},len(active)


def units(code):
    try:tree=ast.parse(code)
    except (SyntaxError,ValueError):return []
    return [(n.name,ast.get_source_segment(code,n)) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]


def entry_code(code,name):return next((source for n,source in units(code) if n==name),None)


def changes(before,after):
    try:
        old=ast.parse(before);new=ast.parse(after)
    except (SyntaxError,ValueError):return {'total':0,'shown':[],'status':'unparseable'}
    def items(tree,source):
        return [(n.name,ast.dump(s,include_attributes=False),ast.get_source_segment(source,s))
            for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for s in n.body]
    a,b=items(old,before),items(new,after);ad={x[1] for x in a};bd={x[1] for x in b};changed=[]
    for name,stamp,text in a:
        if stamp not in bd:changed.append({'function':name,'before':text,'after':None})
    for name,stamp,text in b:
        if stamp not in ad:changed.append({'function':name,'before':None,'after':text})
    return {'total':len(changed),'shown':changed[:2],'status':'parsed'}
