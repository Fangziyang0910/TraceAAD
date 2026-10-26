"""Small conjugate outcome models, prequential scores, and forward disagreement."""
import math
from collections import deque


def distances(a, programs):
    if a not in programs:
        return {}
    graph = {}
    for p in programs.values():
        for q in (p.parent_id, p.reference_id):
            if q is not None:
                graph.setdefault(p.id, set()).add(q)
                graph.setdefault(q, set()).add(p.id)
    queue, seen = deque([a]), {a:0}
    while queue:
        p = queue.popleft()
        for q in graph.get(p, ()):
            if q not in seen:
                seen[q]=seen[p]+1; queue.append(q)
    return seen


def distance(a,b,programs):
    return distances(a,programs).get(b)


def primary_path(pid, programs):
    result = []
    while pid is not None:
        result.append(pid)
        pid = programs[pid].parent_id
    return result


def weights(log_weights):
    peak = max(log_weights)
    values = [math.exp(x-peak) for x in log_weights]
    total = sum(values)
    return [x/total for x in values]


def cell(prior, records, *, condition=None, value=None, parent=None, programs=None, decay=.5):
    av,bv,ar,br = prior
    k,b = 3.,2.
    near=distances(parent,programs) if parent is not None else {}
    for r in records:
        if not r.get('credit'):
            continue
        w = 1.
        if condition is not None and r['conditions'][condition] is not value:
            continue
        if parent is not None:
            d = None if r.get('external') else near.get(r['parent'])
            if d is None:
                continue
            w = decay**d
        if not r['A']:
            bv += w
        else:
            av += w
            if r['Y']:
                ar += w; k += w; b += w*r['Z']
            else:
                br += w
    return [av,bv,ar,br,k,b]


def predictions(transform, prior, conditions, parent, programs, decay):
    records=transform['records']
    global_cell=cell(prior, records)
    cells=[global_cell]
    for j, value in enumerate(conditions):
        cells.append(global_cell[:] if value is None else cell(prior, records, condition=j, value=value))
    cells.append(cell(prior, records, parent=parent, programs=programs, decay=decay))
    return cells


def mean(c):
    av,bv,ar,br,k,b=c
    return av/(av+bv)*ar/(ar+br)*b/(k-1)


def sample(c, rng):
    av,bv,ar,br,k,b=c
    return rng.betavariate(av,bv)*rng.betavariate(ar,br)/rng.gammavariate(k,1/b)


def log_predictive(c, record):
    av,bv,ar,br,k,b=c
    result=math.log((av if record['A'] else bv)/(av+bv))
    if record['A']:
        result+=math.log((ar if record['Y'] else br)/(ar+br))
        if record['Y']:
            result+=math.log(k)+k*math.log(b)-(k+1)*math.log(b+record['Z'])
    return result


def disagreement(cells, probabilities):
    means=[mean(c) for c in cells]
    avg=sum(w*m for w,m in zip(probabilities,means))
    return sum(w*(m-avg)**2 for w,m in zip(probabilities,means))
