from experiments.traceaad_v20 import scheduling as routing
from traceaad_base import ModelReply
from traceaad_v20.experience import REFLECTION_SCHEMA


def test_role_schema_uses_same_pool_and_generation_settings(monkeypatch):
    endpoints = [{'name': 'server3', 'base_url': 'http://one/v1', 'model': 'qwen'}]
    state = {'instance_id': 'a', 'endpoints': endpoints}
    monkeypatch.setattr(routing, 'scheduler_status', lambda p: state)

    class Lease:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def status(self): return state
        def acquire(self): return '0:0'
        def poll(self, *a, **k): pass
    monkeypatch.setattr(routing, 'Lease', Lease)
    sent = []

    class Client:
        def __init__(self, *a, **kw): sent.append(kw)

        def complete(self, prompt, **kw):
            sent.append(kw)
            return ModelReply('{}', usage={'total_tokens': 3})
    model = routing.ScheduledModel('/fake', 'run', client_factory=Client)
    reflection = model.for_role(REFLECTION_SCHEMA)
    with reflection.request_context():
        reply = reflection.complete('briefly', temperature=1., max_tokens=16384, timeout=600)
    assert sent[0]['request_options']['response_format'] == REFLECTION_SCHEMA
    assert sent[0]['request_options']['chat_template_kwargs'] == {'enable_thinking': False}
    assert sent[1]['max_tokens'] == 16384
    assert model.request_options['response_format'] == routing.CANDIDATE_RESPONSE_FORMAT


def test_common_default_and_explicit_historical_backend():
    from experiments.infra.base import DEFAULT_BACKEND, HOSTED_BACKENDS, resolve_backend
    assert DEFAULT_BACKEND == 'server3'
    assert resolve_backend('deepseek', None, None, None).model == 'deepseek-v4.1-flash'
    assert 'deepseek' in HOSTED_BACKENDS and 'server3' not in HOSTED_BACKENDS
