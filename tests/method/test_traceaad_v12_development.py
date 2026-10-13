"""V12.0 short development, repair, budget closure and resume behavior."""

import pytest

from experiments.infra.diagnose_search import diagnostics
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v12_0 import BaseConfig, TraceAADV120

BOOM = "Design: fails\n```python\ndef score(x):\n    raise ValueError('boom')\n```"


def design(value):
    return f"Design: return {value}\n```python\ndef score(x):\n    return {value}\n```"


def method(tmp_path, *answers, budget=5, develop=1):
    return TraceAADV120(evaluation=TinyEvaluation(), llm=TokenLLM(*answers), run_dir=tmp_path,
                        config=BaseConfig(budget=budget, roots=1, development_probability=develop))


def proposed(tmp_path, *answers, **options):
    search = method(tmp_path, response(8), *answers, **options)
    search._roots()
    search._roots()
    search._ordinary_search("Explore")
    return search


@pytest.mark.parametrize('proposal,develop,status', [
    (design(5.5), 0, 'valid'), (response(8), 1, 'duplicate')])
def test_proposal_closes_without_development(tmp_path, proposal, develop, status):
    search = proposed(tmp_path, proposal, develop=develop)
    search._search()
    record = search.facts.explorations[1]
    assert search.attempts == 2 and search.attempts_table[2]['status'] == status
    assert record['developed'] is False and record['development_attempts'] == []
    assert record['first_score'] == (-5.5 if status == 'valid' else None)


def test_development_starts_from_the_best_program_reached(tmp_path):
    search = proposed(tmp_path, design(5.5), design(6.5), design(4.5), design(7.5))
    for _ in range(4):
        search._search()
    steps = [a for a in search.attempts_table.values() if a['id'] > 2]
    assert [(a['action'], a['parent_id'], a['exploration']['step']) for a in steps] == [
        ('Refine', 2, 1), ('Refine', 3, 2), ('Refine', 3, 3)]
    record = search.facts.explorations[1]
    assert (record['first_score'], record['best_score'], record['best_id']) == (-5.5, -7.5, 5)
    assert record['search_best_score'] == -8 and record['development_attempts'] == [3, 4, 5]
    assert search._open_exploration() is None
    report = diagnostics(search.facts, search.config.budget, search.progress.init_attempts)
    assert report['explorations']['developed'] == report['explorations']['development_improved'] == 1
    assert report['explorations']['development_generations'] == 3


def test_repaired_proposal_is_developed_from_its_repaired_version(tmp_path):
    search = proposed(tmp_path, BOOM, design(3.5), design(4.5))
    assert search.attempts_table[3]['repair_of'] == 2 and search.attempts_table[3]['status'] == 'valid'
    search._search()
    assert search.attempts_table[4]['parent_id'] == 3
    assert search.attempts_table[4]['exploration'] == {'id': 1, 'step': 1}


def test_resume_continues_the_open_exploration(tmp_path):
    search = proposed(tmp_path, design(5.5), design(6.5))
    search._search()
    resumed = method(tmp_path, design(4.5), design(7.5))
    assert resumed._open_exploration()['best']['id'] == 3
    resumed._search()
    resumed._search()
    assert [(a['parent_id'], a['exploration']['step']) for a in resumed.attempts_table.values() if a['id'] > 3] == [
        (3, 2), (3, 3)]


def test_exhausted_budget_closes_the_open_exploration(tmp_path):
    search = proposed(tmp_path, design(5.5), design(6.5), budget=3)
    search._search()
    search._search()
    assert search.phase == 'freeze' and search.attempts == 3
    assert search.facts.explorations[1]['development_attempts'] == [3]
