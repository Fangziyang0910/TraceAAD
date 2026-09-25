import json
from types import SimpleNamespace

from experiments.infra.equivalent_backends import allocate_anywhere, prepare_resume


def test_route_update_preserves_search_and_is_restartable(tmp_path):
    state = {'version': 1081, 'mechanism': {'llm': {'base_url': 'old', 'model': 'alias',
             'temperature': 1}},
             'budget_used': 665, 'nodes': [1, 2], 'rng_state': [3, [4], None]}
    original = json.loads(json.dumps(state))
    (tmp_path/'tree_state.json').write_text(json.dumps(state))
    (tmp_path/'pending_candidate.json').write_text('selected candidate, exact prompt and rng')
    (tmp_path/'evaluations.jsonl').write_text('durable receipts')
    (tmp_path/'run_config.json').write_text(json.dumps({'backend': 'old', 'llm': {'model': 'alias'}}))
    item = SimpleNamespace(run_dir=tmp_path, backend='new')
    profiles = {'new': SimpleNamespace(base_url='new-url', model='new-alias', no_proxy='localhost')}
    for _ in range(2):
        prepare_resume(item, profiles)
        result = json.loads((tmp_path/'tree_state.json').read_text())
        result['mechanism']['llm'].update(base_url='old', model='alias')
        assert result == original
        assert (tmp_path/'pending_candidate.json').read_text() == 'selected candidate, exact prompt and rng'
        assert (tmp_path/'evaluations.jsonl').read_text() == 'durable receipts'
    assert json.loads((tmp_path/'checkpoint_before_service_routing.json').read_text()) == original
    assert json.loads((tmp_path/'run_config.json').read_text())['backend'] == 'new'


def test_allocator_adapter_preserves_original_rows():
    rows = [{'run_name': 'paused', 'backend': 'busy', 'status': 'queued'},
            {'run_name': 'active', 'backend': 'busy', 'status': 'running'}]
    def pinned_allocator(plan, available, backend_pool):
        assert plan[1] is rows[1]
        assert backend_pool == ('free', 'other')
        return [(r, b) for r in plan if r['status'] == 'queued'
                for b, n in available.items() if n and r['backend'] in (None, b)]
    assignments = allocate_anywhere(pinned_allocator, rows, {'free': 1}, ('free', 'other'))
    assert assignments == [(rows[0], 'free')]
    assert assignments[0][0] is rows[0] and rows[0]['backend'] == 'busy'
