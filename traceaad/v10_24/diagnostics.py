"""Read development outcomes from committed policy checkpoints, without re-evaluation."""

from collections import Counter


def development_stats(facts):
    policy = (facts.state or {}).get('v1024', {})
    units, blocks = policy.get('units', []), policy.get('blocks', [])
    costs, reasons = Counter(), Counter()
    for block in blocks:
        costs[block['purpose']] += block['spent']
        reasons[block['allocation_reason'] or 'scheduled'] += 1
    trials = len(facts.attempts)
    new_questions = sum(u['purpose'] == 'new_question' and bool(u['trial_ids']) for u in units)
    delivery = {}
    for mode in ('edit', 'full'):
        rows = [a for a in facts.attempts.values() if a.get('delivery_mode') == mode]
        delivery[mode] = {'attempts': len(rows),
            'delivery_failed': sum(a['status'] == 'delivery_failed' for a in rows),
            'evaluated_candidates': sum(a.get('entered_evaluation', False) for a in rows),
            'new_valid': sum(a['status'] == 'valid' for a in rows),
            **{field: sum(a['delivery_cost'][field] for a in rows)
               if all(a.get('delivery_cost', {}).get(field) is not None for a in rows) else None
               for field in ('model_calls', 'input_tokens', 'output_tokens')}}
    result = {
        'kind': 'finite_development_requests', 'candidate_cost_by_purpose': dict(costs),
        'delivery': delivery,
        'allocation_reasons': dict(reasons), 'reserved': policy.get('reserved', 0),
        'new_question_proposals': new_questions,
        'new_questions_per_100_candidates': 100 * new_questions / trials if trials else None,
        'settled_frontier_gain': sum(b['frontier_gain'] or 0 for b in blocks),
        'units': [{**{k: u.get(k) for k in ('id', 'purpose', 'axis', 'question', 'anchor_id', 'proposal_id',
                   'champion_id', 'worktip_id', 'pending_failure', 'status', 'closure_reason', 'anchor_delta',
                   'block_ids', 'trial_ids', 'linked_unit_ids', 'events')},
                   'cost': len(u['trial_ids'])} for u in units],
        'blocks': blocks,
    }
    if blocks and all(b['closed_after'] is not None for b in blocks):
        best = min(p['fitness'] for p in facts.valid.values())
        result['gain_accounting_error'] = result['settled_frontier_gain'] - (blocks[0]['start_frontier'] - best)
    return result
