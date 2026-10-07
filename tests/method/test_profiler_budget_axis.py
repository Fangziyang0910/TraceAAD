import json

from baselines.profiler import ProfilerBase
from core import Function


def make_function(score):
    function = Function(name='heuristic', args='x', body='    return x')
    function.score = score
    function.operator = 'op'
    return function


def candidates(run_dir):
    rows = [json.loads(line) for line in (run_dir / 'events.jsonl').read_text().splitlines()]
    return [row for row in rows if row['kind'] == 'candidate']


def test_records_default_to_their_order_on_the_budget_axis(tmp_path):
    profiler = ProfilerBase(tmp_path)
    profiler.register_function(make_function(3.0), program='def heuristic(x):\n    return x\n')
    profiler.register_function(make_function(2.0), program='def heuristic(x):\n    return x + 1\n')
    profiler.write_run_summary()

    assert [row['budget_used'] for row in candidates(tmp_path)] == [1, 2]
    assert json.loads((tmp_path / 'summary.json').read_text())['budget_used'] == 2


def test_methods_can_place_records_at_the_evaluation_that_produced_them(tmp_path):
    profiler = ProfilerBase(tmp_path)
    for position, score in ((0, 5.0), (7, 4.0), (31, 3.0)):
        profiler.register_function(make_function(score), program=f'def heuristic(x):\n    return {score}\n',
                                   budget_used=position)
    profiler.write_run_summary(budget_used=40)

    rows = candidates(tmp_path)
    assert [row['candidate_id'] for row in rows] == [1, 2, 3]
    assert [row['budget_used'] for row in rows] == [0, 7, 31]
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['budget_used'] == 40 and summary['candidate_count'] == 3
