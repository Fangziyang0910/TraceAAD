"""Read-only V10.6 prefix audit; run with python3, no evaluator or model calls."""
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[4]
PREFIX = 350
BIRTH_CUTOFF = 175


def identities(code):
    # Keep names, constants, docstrings, imports and the entire evaluated module.
    return code, ast.dump(ast.parse(code), include_attributes=False)


def concentration(counts):
    total = sum(counts.values())
    return dict(uses=total, distinct=len(counts),
                top_share=max(counts.values(), default=0) / total if total else 0)


def audit(path):
    raw = path.read_bytes()
    state = json.loads(raw)
    config = json.loads(path.with_name('run_config.json').read_text())
    assert state['budget_used'] >= PREFIX
    assert state['mechanism']['generation'] == 'idea_code_then_implementation_idea'
    nodes = {n['id']: n for n in state['nodes'] if n['evaluation_id'] <= PREFIX}
    events = []
    with path.with_name('events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line)
            if e['budget_used'] > PREFIX:
                break
            events.append(e)
            if e.get('evaluation_id') == PREFIX:
                break
    assert [e['evaluation_id'] for e in events if e.get('evaluation_id') is not None] == list(range(1, PREFIX + 1))
    assert len({e['candidate_id'] for e in events}) == len(events)
    assert {e['node_id'] for e in events if e.get('node_id') is not None} == set(nodes)
    ids = {i: identities(n['code']) for i, n in nodes.items()}
    seen = [set(), set()]
    visited = set()
    parent_uses = Counter()
    program_uses = [Counter(), Counter()]
    donor_uses = Counter()
    prompt_uses = Counter()
    response = defaultdict(Counter)
    conditioned_response = defaultdict(Counter)
    duplicate = Counter()
    early_pivots = {}
    continuations = defaultdict(list)
    best = None
    initial_best = None
    frontier_gain = 0.0
    rows = []
    for e in events:
        pid, did, nid = e.get('parent_id'), e.get('donor_id'), e.get('node_id')
        op = e['operator']
        condition = None
        if pid is not None:
            repeated = prompt_uses[(op, pid)] > 0
            condition = 'repeat_prompt' if repeated else 'first_prompt'
            conditioned_response[f'{op}:{condition}']['attempts'] += 1
            conditioned_response[f'{op}:{condition}']['evaluations'] += int(e.get('evaluation_id') is not None)
        donor_condition = None
        if did is not None:
            gap = e['donor_fitness'] - e['parent_fitness']
            donor_condition = 'stronger' if gap > 0 else 'weaker' if gap < 0 else 'equal'
            conditioned_response[f'Fuse:donor_{donor_condition}']['attempts'] += 1
        c = response[op]
        c['attempts'] += 1
        c['evaluations'] += int(e.get('evaluation_id') is not None)
        c[e['status']] += 1
        c['eval_seconds'] += e.get('eval_seconds') or 0
        c['llm_seconds'] += e.get('llm_seconds') or 0
        assert e.get('best_before') == best
        if pid is not None:
            assert pid in visited
            assert e['parent_fitness'] == nodes[pid]['fitness']
            parent_uses[pid] += 1
            for j in range(2):
                if e['selection']['parent_count_before'] == 0 and program_uses[j][ids[pid][j]]:
                    duplicate[f'fresh_id_previously_used_program_{j}'] += 1
                program_uses[j][ids[pid][j]] += 1
            prompt_uses[(op, pid)] += 1
            if did is not None:
                assert did in visited and e['donor_fitness'] == nodes[did]['fitness']
                donor_uses[did] += 1
                for j in range(2):
                    duplicate[f'fuse_same_inputs_{j}'] += int(ids[pid][j] == ids[did][j])
            if pid in early_pivots and op in ('Refine', 'Fuse') and e.get('evaluation_id') is not None:
                continuations[pid].append(e)
        if nid is None:
            continue
        node = nodes[nid]
        q = node['fitness']
        assert q == e['fitness'] and node['evaluation_id'] == e['evaluation_id']
        for j in range(2):
            duplicate[f'archive_revisit_{j}'] += int(ids[nid][j] in seen[j])
            if pid is not None:
                duplicate[f'child_equals_parent_{j}'] += int(ids[nid][j] == ids[pid][j])
            if did is not None:
                duplicate[f'child_equals_donor_{j}'] += int(ids[nid][j] == ids[did][j])
            seen[j].add(ids[nid][j])
        if pid is not None:
            dp = q - e['parent_fitness']
            db = q - max(e['parent_fitness'], e['donor_fitness'] if did is not None else e['parent_fitness'])
            gf = max(0, q - best)
            c['parent_improved'] += int(dp > 0)
            c['both_improved'] += int(db > 0)
            c['frontier_improved'] += int(gf > 0)
            c['equal_parent'] += int(dp == 0)
            c['degraded'] += int(dp < 0)
            groups = [f'{op}:{condition}']
            if donor_condition is not None:
                groups.append(f'Fuse:donor_{donor_condition}')
            for group in groups:
                conditioned_response[group].update(valid=1, parent_improved=int(dp > 0),
                                                   both_improved=int(db > 0), frontier_improved=int(gf > 0))
            rows.append(dict(candidate_id=e['candidate_id'], evaluation_id=e['evaluation_id'],
                             operator=op, parent_delta=dp, both_delta=db, frontier_gain=gf))
            if op == 'Pivot' and dp < 0 and e['evaluation_id'] <= BIRTH_CUTOFF:
                early_pivots[nid] = e['parent_fitness']
        if best is not None:
            frontier_gain += max(0, q - best)
        else:
            initial_best = q
        best = max(q, best) if best is not None else q
        visited.add(nid)
    assert abs(frontier_gain - (best - initial_best)) < 1e-8
    recovery = Counter(total=len(early_pivots))
    recovery_details = []
    for nid, target in early_pivots.items():
        follow = continuations[nid]
        recovery['direct_RF_evaluations'] += len(follow)
        recovered = next((j for j, e in enumerate(follow, 1)
                          if e['fitness'] is not None and e['fitness'] >= target), None)
        recovery['untested' if not follow else 'recovered' if recovered else 'tested_not_recovered'] += 1
        if follow:
            recovery_details.append(dict(node_id=nid, target=target,
                follow=[{k: e.get(k) for k in ['evaluation_id', 'operator', 'fitness', 'donor_fitness', 'both_improved', 'frontier_improved']} for e in follow]))
    return dict(run=path.parent.name, task=config['task'], backend=config['backend'],
                prefix=PREFIX, valid_nodes=len(nodes), source=str(path.relative_to(ROOT)),
                unique_raw=len(seen[0]), unique_ast=len(seen[1]), duplicate=dict(duplicate),
                parents=concentration(parent_uses), parent_program_raw=concentration(program_uses[0]),
                parent_program_ast=concentration(program_uses[1]), donors=concentration(donor_uses),
                prompt_attempts=sum(prompt_uses.values()), unique_prompts=len(prompt_uses),
                response={op: dict(counts) for op, counts in response.items()},
                conditioned_response={key: dict(counts) for key, counts in conditioned_response.items()},
                early_pivot_direct_recovery=dict(recovery), recovery_details=recovery_details, response_rows=rows)


if __name__ == '__main__':
    started = time.perf_counter()
    a, b = identities('x = 1\n'), identities('# comment\nx=1\n')
    assert a[0] != b[0] and a[1] == b[1]
    assert a[1] != identities('x=2\n')[1]
    paths = sorted((ROOT / 'experiments/traceaad_v10_6/results').glob('*/*revised*/tree_state.json'))
    assert len(paths) == 15
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), prefix=PREFIX,
                  pivot_birth_cutoff=BIRTH_CUTOFF, identity_order=['raw', 'ast'], runs=[audit(p) for p in paths])
    result['elapsed_seconds'] = time.perf_counter() - started
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for run in result['runs']:
        print(run['run'], run['valid_nodes'], run['unique_raw'], run['unique_ast'],
              run['duplicate'], run['early_pivot_direct_recovery'])
