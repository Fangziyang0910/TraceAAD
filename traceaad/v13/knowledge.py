"""DAG views, grounded edits and outcome-conditioned transformation selection."""

import hashlib
import io
import tokenize
from difflib import SequenceMatcher

SPEC_FIELDS = ('name', 'applicability', 'operation', 'dependencies', 'parameters', 'preserve', 'checks')
APPLIED = ('improved', 'no_improvement')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def ancestors(node_id, programs):
    seen, pending = set(), [node_id]
    while pending:
        pid = pending.pop()
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        node = programs[pid]
        pending.append(node.parent_id)
        if node.action == 'Crossover':
            pending.append(node.reference_id)
    return seen


def syntax_tokens(code):
    try:
        return tuple((t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(code).readline)
                     if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                                       tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER))
    except (tokenize.TokenError, IndentationError):
        return ()


def unchanged_regions(code, regions):
    tokens = syntax_tokens(code)
    for region in regions:
        block = syntax_tokens(region['old'])
        if not block or not any(tokens[i:i+len(block)] == block for i in range(len(tokens)-len(block)+1)):
            return False
    return bool(regions)


def bindings(value, code):
    if not isinstance(value, list) or not value:
        raise ValueError('bindings must name nonempty grounded regions')
    result = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get('role'), str) or not item['role'].strip():
            raise ValueError('each binding needs a computational role')
        old = item.get('old')
        if not isinstance(old, str) or not old.strip() or code.count(old) != 1:
            raise ValueError('each bound region must occur exactly once in the actual parent')
        result.append({'role': item['role'], 'old': old})
    return result


def changed_regions(before, after):
    """Ground an observed source edit directly; only new recipients need model binding."""
    lines = before.splitlines(keepends=True)
    regions = []
    for tag, start, end, _, _ in SequenceMatcher(None, lines, after.splitlines(keepends=True), autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if start == end:
            start, end = max(0, start-1), min(len(lines), end+1)
        block = ''.join(lines[start:end])
        while (not block.strip() or before.count(block) != 1) and (start > 0 or end < len(lines)):
            start, end = max(0, start-1), min(len(lines), end+1)
            block = ''.join(lines[start:end])
        if not any(r['old'] == block for r in regions):
            regions.append(dict(role='observed source change', old=block))
    return bindings(regions, before)


def apply_edits(code, edits, regions):
    if not isinstance(edits, list) or not edits:
        raise ValueError('edits must be nonempty')
    # All patches refer to the original parent, so overlapping/sequential escape is rejected.
    patches = []
    allowed = [(code.index(r['old']), code.index(r['old']) + len(r['old'])) for r in regions]
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError('invalid patch')
        old, new = edit.get('old'), edit.get('new')
        if not isinstance(old, str) or not old or not isinstance(new, str) or code.count(old) != 1:
            raise ValueError('each old patch must occur exactly once')
        start, end = code.index(old), code.index(old) + len(old)
        if not any(lo <= start and end <= hi for lo, hi in allowed):
            raise ValueError('edit outside matched computation/dependency regions')
        patches.append((start, end, new))
    patches.sort()
    if any(a[1] > b[0] for a, b in zip(patches, patches[1:])):
        raise ValueError('overlapping patches')
    for start, end, new in reversed(patches):
        code = code[:start] + new + code[end:]
    return code


def specification(value, library, *, own_id=None):
    if not isinstance(value, dict):
        raise ValueError('specification must be an object')
    value = dict(value)
    for key in SPEC_FIELDS:
        field = value.get(key)
        if isinstance(field, list) and field and all(isinstance(s, str) and s.strip() for s in field):
            value[key] = '\n'.join(field)
    if any(not isinstance(value.get(k), str) or not value[k].strip() for k in SPEC_FIELDS):
        raise ValueError('specification needs all computational fields')
    required = value.get('requires', [])
    if (not isinstance(required, list) or any(type(x) is not str or x not in library for x in required)
            or own_id in required or len(set(required)) != len(required)):
        raise ValueError('invalid prerequisite transformation ids')
    def visit(tid, seen):
        if tid == own_id or tid in seen:
            raise ValueError('cyclic transformation prerequisites')
        for dependency in library[tid]['spec']['requires']:
            visit(dependency, seen | {tid})
    for tid in required:
        visit(tid, set())
    return {**{k: value[k].strip() for k in SPEC_FIELDS}, 'requires': required}


def used_on_path(node_id, programs, attempts, library=None):
    lineage = ancestors(node_id, programs)
    used = {a['transform'] for a in attempts
            if a.get('child') in lineage and a.get('outcome') in APPLIED and a.get('transform')}
    pending = list(used)
    while pending:
        tid = pending.pop()
        t = (library or {}).get(tid, {})
        for related in [t.get('family', tid), *t.get('components', [])]:
            if related not in used:
                used.add(related)
                pending.append(related)
    return used


def eligible(parent, programs, library, attempts):
    lineage = ancestors(parent.id, programs)
    used = used_on_path(parent.id, programs, attempts, library)
    result = []
    for tid, transform in library.items():
        if (tid in used or transform.get('family', tid) in used
                or set(transform.get('components', [])) & used
                or set(transform['spec']['requires']) - used):
            continue
        family = transform.get('family', tid)
        failed = [a for a in attempts if library.get(a.get('transform'), {}).get('family') == family and a.get('parent') in lineage
                  and a.get('outcome') == 'no_improvement']
        if any(unchanged_regions(parent.code, a.get('bindings', [])) for a in failed):
            continue
        result.append(tid)
    return result


def evidence_rate(records, *, route=None):
    comparable = [r for r in records if r['outcome'] in APPLIED]
    similar = [r for r in comparable if r.get('route') == route] if route else []
    sample = similar or comparable
    return (1 + sum(r['outcome'] == 'improved' for r in sample)) / (2 + len(sample))


def distribution(keys, records, field, window, floor=0):
    samples = [[r for r in records if r.get(field) == k and r.get('mode') != 'Init'][-window:] for k in keys]
    # Search allocation counts every spent opportunity, unlike transformation performance evidence.
    rates = [(1 + sum(r['outcome'] == 'improved' for r in sample)) / (2 + len(sample)) for sample in samples]
    total = sum(rates)
    return [floor / len(keys) + (1-floor) * rate / total for rate in rates]


def trial_view(node_id, programs, attempts):
    lineage = ancestors(node_id, programs)
    groups = {}
    for row in attempts:
        failed = (row.get('outcome') == 'no_improvement' or
                  row.get('outcome') == 'implementation_failed' and row.get('mode') in ('Open', 'Crossover'))
        if row.get('parent') not in lineage or not failed:
            continue
        child = programs.get(row.get('child'))
        key = row.get('transform') or (child.request.get('trace', {}).get('key') if child else None) or 'unclassified'
        group = groups.setdefault(key, dict(key=key, count=0, examples=[]))
        group['count'] += 1
        group['examples'] = (group['examples'] + [dict(parent=row['parent'], child=row.get('child'),
                                                     failure=row.get('failure'))])[-3:]
    return list(groups.values())
