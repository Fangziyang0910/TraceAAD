import pytest
from experiments.infra.raw_objectives import restore_record


def test_restore_uses_each_instance_reference_and_is_idempotent():
    original = {'valid': True, 'score': -25., 'instances': [
        {'instance_index': 0, 'valid': True, 'score': -50.},
        {'instance_index': 1, 'valid': True, 'score': 0.}]}
    rows = [{'reference': 10.}, {'reference': 100.}]
    result = restore_record(original, rows)
    assert result['score'] == 52.5
    assert original['score'] == -25.
    assert restore_record(result, rows) == result


def test_aggregate_deviation_cannot_be_converted():
    with pytest.raises(ValueError, match='every instance'):
        restore_record({'valid': True, 'score': -25., 'instances': []}, [{'reference': 10.}])


def test_relative_heldout_cache_is_recomputed( tmp_path, monkeypatch):
    import json
    from experiments.infra import evaluate
    from traceaad.v12_1.storage import read_json
    config = {'task': 'jssp_construct', 'method_params': {}, 'revision': 'old'}
    (tmp_path/'run_config.json').write_text(json.dumps(config))
    previous = {'task': 'jssp_construct', 'split': 'test_20', 'scale': '20', 'variant': '',
                'key': 'code-key', 'node_id': 1, 'fitness': -12.}
    (tmp_path/'heldout.json').write_text(json.dumps([previous]))
    chosen = {'program': 'source', 'node_id': 1, 'sample_order': 1}
    # The generic entry derives the content hash itself.
    import hashlib
    previous['key'] = hashlib.sha256(b'source').hexdigest()
    (tmp_path/'heldout.json').write_text(json.dumps([previous]))
    monkeypatch.setattr(evaluate, 'pick_best_sample', lambda *a, **k: (chosen, []))
    monkeypatch.setattr(evaluate, 'load_run_summary', lambda *a, **k: {'status': 'finished'})
    monkeypatch.setattr(evaluate, 'heldout_identity', lambda *a: 'verified')
    calls = []

    class Evaluator:
        protocol = 'raw-protocol'

        def evaluate(self, code, key):
            calls.append(key)
            return {'fitness': 42., 'evaluations': [{'valid': True, 'score': 42., 'seconds': 0.}]}

    monkeypatch.setattr(evaluate, 'heldout_evaluator', lambda *a, **k: (
        Evaluator(), {'score_metric': 'raw_objective'}))
    result = evaluate.evaluate_run(tmp_path, ['test_20'])[0]
    assert calls == [previous['key']]
    assert result['fitness'] == 42.
    assert read_json(tmp_path/'heldout.json')[0]['score_metric'] == 'raw_objective'
