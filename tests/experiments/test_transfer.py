import pytest

from experiments.transfer.run import apply_edits, decode_candidate, json_reply
from experiments.transfer.analyze import outcome, summarize


def test_transfer_rejects_ambiguous_edits_and_preserves_unedited_code():
    code = 'def f(x):\n    return x + 1\n'
    changed = apply_edits(code, [{'old': 'x + 1', 'new': 'x + 2'}])
    assert changed == 'def f(x):\n    return x + 2\n'
    for edits in ([{'old': '', 'new': 'x'}], [{'old': 'x', 'new': 'y'}],
                  [{'old': 'absent', 'new': 'x'}], []):
        with pytest.raises(ValueError):
            apply_edits(code, edits)
    with pytest.raises(ValueError, match='incomplete'):
        json_reply({'content': '{}', 'finish_reason': 'length'})
    parent = {'code': code}
    cell = {'arm': 'constrained'}
    assert decode_candidate(cell, parent, {'applicable': False, 'edits': []}, code) == (code, 'not_applicable')
    with pytest.raises(ValueError):
        decode_candidate(cell, parent, {'applicable': False, 'edits': [{'old': '1', 'new': '2'}]}, code)
    with pytest.raises(ValueError):
        decode_candidate(cell, parent, {'applicable': True, 'bindings': 'f',
                         'edits': [{'old': 'def f(x)', 'new': 'def g(x)'}]}, code)


def test_transfer_keeps_failures_in_improvement_denominator():
    rows = []
    for fitness in (9, 10, None):
        result = outcome({'fitness': fitness}, {'fitness': 10})
        rows.append({'status': 'generated', **{f'{s}_{k}': v for s in ('train', 'heldout')
                                              for k, v in result.items()}})
    summary = summarize(rows)
    assert summary['n'] == 3
    assert summary['heldout']['improved'] == 1
    assert summary['heldout']['valid'] == 2
    assert summary['heldout']['mean_gain'] == .05
    with pytest.raises(ValueError):
        outcome({'fitness': 9}, {'fitness': None})
