"""Frozen identities, comparable AST bindings, and deterministic token templates."""
import ast
from copy import deepcopy
from difflib import SequenceMatcher
import io
import keyword
import json
import tokenize
from traceaad.v13.knowledge import specification as base_specification, digest, apply_edits


def specification(value, max_conditions=3):
    if not isinstance(value,dict): raise ValueError('specification must be an object')
    value=deepcopy(value)
    parameters=value.get('parameters')
    if isinstance(parameters,dict) and parameters:
        # Preserve named semantic choices exactly when a model uses JSON instead of prose.
        value['parameters']=json.dumps(parameters,ensure_ascii=False,sort_keys=True,allow_nan=False)
    spec=base_specification(value,{})
    policy=value.get('binding_policy')
    if not isinstance(policy,str) or not policy.strip():
        raise ValueError('missing fixed binding policy')
    conditions=value.get('conditions')
    if not isinstance(conditions,list) or len(conditions)>max_conditions:
        raise ValueError('invalid frozen conditions')
    for h in conditions:
        if not isinstance(h,dict) or h.get('kind') not in ('static','runtime') or any(
                not isinstance(h.get(k),str) or not h[k].strip() for k in ('name','predicate','rationale')):
            raise ValueError('invalid condition proposition')
    return {**spec,'binding_policy':policy.strip(),'conditions':deepcopy(conditions)}


def canonical(code):
    try:
        return ast.dump(ast.parse(code),include_attributes=False)
    except SyntaxError:
        return code


def identity(spec):
    import json
    return digest(json.dumps({k:spec[k] for k in ('operation','parameters','binding_policy')},sort_keys=True))


def ast_path(code, region):
    """Smallest containing statement; paths survive unambiguous statement insertion/deletion."""
    raw_start=code.index(region)
    start=raw_start+len(region)-len(region.lstrip()); end=raw_start+len(region.rstrip())
    lines=code.splitlines(keepends=True)
    offsets=[0]
    for line in lines: offsets.append(offsets[-1]+len(line))
    found=[]
    def visit(node,path):
        if isinstance(node,ast.stmt):
            # ast column offsets are UTF-8 byte offsets.
            lo=offsets[node.lineno-1]+len(lines[node.lineno-1].encode()[:node.col_offset].decode())
            hi=offsets[node.end_lineno-1]+len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
            if lo<=start and end<=hi: found.append((hi-lo,path))
        for field,value in ast.iter_fields(node):
            if isinstance(value,ast.AST): visit(value,path+[(field,None)])
            elif isinstance(value,list):
                for i,child in enumerate(value):
                    if isinstance(child,ast.AST): visit(child,path+[(field,i)])
    visit(ast.parse(code),[])
    return min(found,key=lambda x:x[0])[1] if found else None


def map_path(before,after,path):
    a,b=ast.parse(before),ast.parse(after)
    out=[]
    for field,index in path:
        if type(a) is not type(b) or not hasattr(b,field): return None
        if isinstance(a,(ast.FunctionDef,ast.AsyncFunctionDef)) and a.name!=b.name: return None
        aa,bb=getattr(a,field),getattr(b,field)
        if index is None:
            a,b=aa,bb; out.append((field,None)); continue
        if not isinstance(bb,list): return None
        keys=lambda seq:[ast.dump(x,include_attributes=False) if isinstance(x,ast.AST) else repr(x) for x in seq]
        mapped=None
        for tag,i,j,k,l in SequenceMatcher(None,keys(aa),keys(bb),autojunk=False).get_opcodes():
            if i<=index<j and (tag=='equal' or tag=='replace' and j-i==l-k==1):
                mapped=k+index-i; break
        if mapped is None: return None
        a,b=aa[index],bb[mapped]; out.append((field,mapped))
    return out if type(a) is type(b) else None


def comparable(q,p,old_bindings,new_bindings,programs):
    if [x['role'] for x in old_bindings]!=[x['role'] for x in new_bindings]: return False
    def path(pid):
        out=[]
        while pid is not None:
            out.append(pid); pid=programs[pid].parent_id
        return out
    qa,pa=path(q),path(p)
    common=next((i for i in qa if i in pa),None)
    if common is None: return False
    chain=qa[:qa.index(common)+1]+list(reversed(pa[:pa.index(common)]))
    for old,new in zip(old_bindings,new_bindings):
        anchor=ast_path(programs[q].code,old['old'])
        target=ast_path(programs[p].code,new['old'])
        if anchor is None or target is None: return False
        for a,b in zip(chain,chain[1:]):
            anchor=map_path(programs[a].code,programs[b].code,anchor)
            if anchor is None: return False
        if anchor!=target: return False
    return True


def token_template(edits,symbols):
    reverse={v:'ROLE_'+k for k,v in symbols.items()}
    def tokens(text):
        return [(t.type,reverse.get(t.string,t.string) if t.type==tokenize.NAME else t.string)
                for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type not in (tokenize.ENCODING,tokenize.ENDMARKER)]
    return [{k:tokens(e[k]) for k in ('old','new')} for e in edits]


def render_template(template,symbols,parent,regions):
    mapping={'ROLE_'+k:v for k,v in symbols.items()}
    # Token patterns ignore whitespace, but edits use exact slices of the actual bound region.
    def meaningful(ts): return [(t,str(v)) for t,v in ts if t not in
        (tokenize.NL,tokenize.NEWLINE,tokenize.INDENT,tokenize.DEDENT,tokenize.COMMENT)]
    edits=[]
    for item in template:
        pattern=meaningful([(t,mapping.get(v,v)) for t,v in item['old']])
        matches=[]
        for region in regions:
            text=region['old']; ts=list(tokenize.generate_tokens(io.StringIO(text).readline))
            actual=[t for t in ts if t.type not in (tokenize.ENDMARKER,tokenize.NL,tokenize.NEWLINE,
                      tokenize.INDENT,tokenize.DEDENT,tokenize.COMMENT)]
            for i in range(len(actual)-len(pattern)+1):
                if [(t.type,t.string) for t in actual[i:i+len(pattern)]]==pattern:
                    offsets=[0]
                    for line in text.splitlines(keepends=True): offsets.append(offsets[-1]+len(line))
                    first,last=actual[i],actual[i+len(pattern)-1]
                    old=text[offsets[first.start[0]-1]+first.start[1]:offsets[last.end[0]-1]+last.end[1]]
                    matches.append(old)
        if len(matches)!=1: raise ValueError('template pattern not uniquely bound')
        new=tokenize.untokenize([(t,mapping.get(v,v)) for t,v in item['new']]).strip('\n')
        edits.append(dict(old=matches[0],new=new))
    code=apply_edits(parent,edits,regions)
    ast.parse(code)
    return code,edits


def valid_symbols(symbols,code):
    names={t.string for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type==tokenize.NAME}
    return isinstance(symbols,dict) and all(isinstance(k,str) and k.isidentifier() and
        isinstance(v,str) and v.isidentifier() and not keyword.iskeyword(v) and v in names
        for k,v in symbols.items()) and len(set(symbols.values()))==len(symbols)
