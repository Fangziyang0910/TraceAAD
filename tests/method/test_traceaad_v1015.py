"""V10.15's identity, budget and selection invariants."""

import json
import math
import random

import pytest

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v10_15 import Config, TraceAADV1015
from traceaad.v10_15.canonical import canonical, key
from traceaad.v10_15.delivery import DeliveryError, SourceError, parse_response
from traceaad.v10_15.history import change_summary, code_diff
from traceaad.v10_15.prompts import PromptBuilder
from traceaad.v10_15.selection import (choose_explore_references, choose_reference, probabilities,
                                      sample_parent, score_classes)


class SelectionEvaluation(TinyEvaluation):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1) + self.offset


def method(tmp_path, *answers, budget=10, selection=False, **config):
    return TraceAADV1015(
        evaluation=TinyEvaluation(),
        selection_evaluation=SelectionEvaluation(100) if selection else None,
        llm=TokenLLM(*answers), run_dir=tmp_path,
        config=Config(budget=budget, **config))


def test_canonical_identity_discards_comments_docs_and_formatting():
    a = '"module"\n# comment\ndef score(x):\n    "doc"\n    return x+1\n'
    b = 'def score( x ):\n return x + 1 # another comment\n'
    assert canonical(a) == canonical(b)
    assert key(canonical(a)) == key(canonical(b))
    assert canonical('class C:\n "doc"\n') == 'class C:\n    pass\n'


def test_parent_distribution_is_over_score_classes_with_fixed_ess():
    p, beta, ess, target = probabilities([float(q) for q in range(20)])
    assert beta > 0 and sum(p) == pytest.approx(1)
    assert target == 8 and ess == pytest.approx(8)
    assert probabilities([1.0]) == ([1.0], 0.0, 1.0, 1.0)
    # Few classes: ESS cannot exceed the class count, so sampling is uniform.
    assert probabilities([0.0, 1.0, 2.0])[0] == [1/3] * 3
    # 200 rewrites tied at the top form one class: the tie gets one class's share,
    # and runners-up keep theirs instead of dropping to zero (V10.15's tie mode).
    nodes = [{'id': i, 'fitness': 2.0} for i in range(200)] + [
        {'id': 200 + i, 'fitness': float(i) / 10} for i in range(3)]
    classes = score_classes(nodes)
    assert [len(c) for c in classes] == [200, 1, 1, 1]
    assert [m[0]['id'] for m in classes] == [0, 202, 201, 200]
    draws = [sample_parent(nodes, random.Random(seed))[1] for seed in range(400)]
    top_share = sum(d['class_size'] == 200 for d in draws) / len(draws)
    assert 0.25 < top_share < 0.5
    assert all(d['classes'] == 4 and d['eligible'] == 203 for d in draws)


def test_finalists_are_one_per_score_class(tmp_path):
    answers = [response(v) for v in (1, 2, 3, 4, 5, 6, 7, 8)]
    answers += [f"Idea: same\n```python\ndef score(x):\n    return {v}  # variant\n    pass\n```"
                for v in ('8.0', '4 + 4', '16 / 2')]
    m = method(tmp_path, *answers, budget=11, selection=True)
    m.run()
    fitness = {n['id']: n['fitness'] for n in m.archive.values()}
    assert sorted(fitness.values()).count(8) == 4
    assert m.finalists == [8, 7, 6, 5, 4]
    assert len({fitness[i] for i in m.finalists}) == 5
    same = [a for a in m.facts.tables['attempt'].values() if a.get('same_as_parent')]
    assert all(m.archive[a['parent_id']]['fitness'] == a['fitness'] for a in same)
    diagnostics = json.loads((tmp_path / 'diagnostics.json').read_text())
    assert diagnostics['score_classes'] == 8 and diagnostics['top_class_size'] == 4
    assert diagnostics['new_frontiers_by_action']['Init'] == 8


def test_unclosed_final_block_is_accepted_only_when_complete():
    template = TinyEvaluation().template_program
    code, idea, meta = parse_response('Idea: close enough\n```python\ndef score(x):\n    return 3\n',
                                      'stop', template)
    assert 'return 3' in code and idea == 'close enough'
    assert meta['strategy'] == 'unclosed_final_block'
    with pytest.raises(DeliveryError):  # truncated body
        parse_response('```python\ndef score(x):\n    return (', 'stop', template)
    with pytest.raises(DeliveryError):  # target missing
        parse_response('```python\ndef helper(x):\n    return 1\n', 'stop', template)
    with pytest.raises(DeliveryError):  # not a finished completion
        parse_response('```python\ndef score(x):\n    return 3\n', 'length', template)


def test_numeric_change_and_complete_diff():
    a = 'def score(x):\n    return x + 0.15 + 3\n'
    b = 'def score(x):\n    return x + 0.12 + 4\n'
    assert change_summary(a, b) == 'numeric constants only, in score: 0.15 → 0.12; 3 → 4'
    assert change_summary('def score(x):\n return x - 1\n',
                          'def score(x):\n return x - 2\n') == 'numeric constants only, in score: 1 → 2'
    assert change_summary('def score(x):\n return x + -1\n',
                          'def score(x):\n return x + 2\n') == 'numeric constants only, in score: -1 → 2'
    structural = 'def score(x):\n    y = x + 1\n    return y\n'
    assert '+2/−1 lines in score' in change_summary(a, structural)
    assert '+    return y' in code_diff(a, structural)
    long_code = 'def score(x):\n' + ''.join(f'    x += {i}\n' for i in range(100)) + '    return x\n'
    diff = code_diff(a, long_code)
    assert len(diff.splitlines()) > 60
    assert '+    x += 99' in diff and '+    return x' in diff
    assert 'more diff lines not shown' not in diff


def test_delivery_strict_finish_and_single_repair_payload():
    template = TinyEvaluation().template_program
    with pytest.raises(DeliveryError):
        parse_response(response(1), None, template)
    with pytest.raises(DeliveryError):
        parse_response('Idea: only a thought', 'stop', template)
    with pytest.raises(SourceError) as exc:
        parse_response('Idea: fix this\n```python\ndef wrong(x): return x\n```', 'stop', template)
    assert 'def wrong' in exc.value.code
    code, idea, meta = parse_response('Idea: use two\nCode:\n```python\ndef score(x): return 1\n```\n'
                                      '```python\ndef score(x): return 2\n```', 'stop', template)
    assert 'return 2' in code and idea == 'use two'
    assert meta['block_indices'] == [1]


def test_analysis_is_discarded_and_the_design_is_kept():
    template = TinyEvaluation().template_program
    reply = ("Analysis: The current rule ignores the input sign.\nDesign: maybe add one? No.\n"
             "Design: Add two to the input.\n"
             "Code:\n```python\ndef score(x):\n    return x + 2\n```")
    code, design, _ = parse_response(reply, 'stop', template)
    assert 'return x + 2' in code and design == 'Add two to the input.'


def test_design_label_and_earlier_labels_parse():
    template = TinyEvaluation().template_program
    reply = ("Design: Add two to the input.\nThe sum is returned.\n"
             "Code:\n```python\ndef score(x):\n    return x + 2\n```")
    code, design, _ = parse_response(reply, 'stop', template)
    assert 'return x + 2' in code and design == 'Add two to the input.\nThe sum is returned.'
    for label in ('Idea', 'Thought'):
        assert parse_response(f'{label}: add three\nCode:\n```python\ndef score(x):\n    return x + 3\n```',
                              'stop', template)[1] == 'add three'
    # A description after the code wins over earlier notes.
    notes = ("Idea: add one\nCode:\n```python\ndef score(x):\n    return x + 2\n```\n"
             "Design: The algorithm adds two.")
    assert parse_response(notes, 'stop', template)[1] == 'The algorithm adds two.'


def test_template_import_completion_retains_submitted_source():
    from benchmarks.tsp_construct.template import template_program

    raw = ('def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):\n'
           '    return int(np.min(unvisited_nodes))\n')
    code, _, metadata = parse_response(f'```python\n{raw}```', 'stop', template_program)
    assert code.startswith('import numpy as np\n')
    assert metadata['submitted_code'] == raw.rstrip('\n')
    assert metadata['template_additions'] == ['np']


def test_whole_search_selects_on_independent_set_and_records_provenance(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 11)), selection=True)
    result = m.run()
    assert result['status'] == 'finished'
    assert result['budget_used'] == 10 and result['num_roots'] == 8
    assert result['num_nodes'] == 10 and result['selection_evaluations'] == 5
    assert result['search_evaluations'] == 10 and result['model_calls'] == 10
    assert result['best']['fitness'] == 10 and result['best']['selection_fitness'] == 110
    assert m.facts.tables['request'][9]['parent_id'] in range(1, 9)
    assert m.facts.tables['attempt'][9]['status'] == 'valid'
    assert (tmp_path / 'best_program.py').exists()
    assert json.loads((tmp_path / 'selection.json').read_text())['selected_node'] == 10
    assert '[Task]' in m.facts.tables['request'][9]['prompt']
    assert '[Current Algorithm]' in m.facts.tables['request'][9]['prompt']


def test_invalid_source_gets_exactly_one_paid_repair(tmp_path):
    invalid = 'Idea: try division\n```python\ndef score(x)\n return x\n```'
    m = method(tmp_path, invalid, response(2), budget=2)
    result = m.run()
    assert result['budget_used'] == result['model_calls'] == 2
    assert result['num_nodes'] == 1 and result['num_roots'] == 1
    assert m.facts.tables['attempt'][1]['status'] == 'invalid_source'
    assert m.facts.tables['attempt'][2]['repair_of'] == 1
    assert m.facts.tables['node'][2]['repaired']


def test_known_failure_and_repair_do_not_recurse(tmp_path):
    bad = 'Idea: divide\n```python\ndef score(x): return 1/0\n```'
    m = method(tmp_path, bad, bad, bad, budget=3)
    result = m.run()
    assert result['status'] == 'no_valid_root'
    assert [m.facts.tables['attempt'][i]['status'] for i in (1, 2, 3)] == [
        'runtime_error', 'known_failure', 'known_failure']
    assert result['search_evaluations'] == 1
    assert m.facts.tables['attempt'][2]['repair_of'] == 1
    assert m.facts.tables['attempt'][3]['repair_of'] is None


def test_duplicate_is_paid_without_another_evaluation(tmp_path):
    m = method(tmp_path, response(1), response(1), budget=2)
    result = m.run()
    assert result['budget_used'] == 2 and result['search_evaluations'] == 1
    assert result['num_nodes'] == 1
    assert m.facts.tables['attempt'][2]['status'] == 'duplicate'


def test_initialization_cap_includes_repair_generations(tmp_path):
    wrong = 'Idea: wrong interface\n```python\ndef other(x): return x\n```'
    m = method(tmp_path, *([wrong] * 16), budget=20)
    result = m.run()
    assert result['status'] == 'no_valid_root'
    assert result['budget_used'] == result['init_attempts'] == 16
    assert result['search_evaluations'] == 0
    assert sum(a['repair_of'] is not None for a in m.facts.tables['attempt'].values()) == 8


def test_too_long_parent_is_removed_without_spending_budget(tmp_path):
    m = TraceAADV1015(evaluation=TinyEvaluation(), llm=TokenLLM(response(1)),
                       run_dir=tmp_path, config=Config(budget=2, max_input_tokens=255))
    m._roots()
    assert m.attempts == 1
    m.phase = 'search'
    m.action_rng.choices = lambda *args, **kwargs: ['Refine']
    m._search()
    assert m.attempts == 1 and m.too_long == {1}
    m._search()
    assert m.phase == 'freeze'


def test_service_retry_does_not_spend_candidate_budget(tmp_path):
    class FlakyLLM(TokenLLM):
        def draw_sample_with_details(self, prompt, **kwargs):
            if not self.calls:
                self.calls.append((prompt, kwargs))
                raise ConnectionError('temporary')
            return super().draw_sample_with_details(prompt, **kwargs)

    m = TraceAADV1015(evaluation=TinyEvaluation(), llm=FlakyLLM(response(1)),
                        run_dir=tmp_path, config=Config(budget=1))
    result = m.run()
    assert result['budget_used'] == 1 and result['model_calls'] == 2
    assert result['service_failures'] == 1


def test_reference_lineage_filter_and_copy_status(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), response(2), budget=9)
    for _ in range(8):
        m._roots()
    assert m.phase == 'roots'
    parent = m.archive[8]
    reference, info = choose_reference(parent, m.archive, m.reference_rng)
    assert reference is not None and not info['relaxed_lineage']
    # Submit the exact chosen reference, independent of its sampled ID.
    m.llm.responses = iter([f"Idea: copy\n```python\n{reference['code']}```"])
    request = m.prompts.build('Crossover', parent, reference=reference)
    m._attempt(request, parent=parent, action='Crossover', reference=reference)
    assert m.facts.tables['attempt'][9]['status'] == 'copied_reference'
    assert len(m.archive) == 8 and m.evaluation_calls == 8


def test_reference_excludes_ancestor_and_descendant_then_relaxes():
    nodes = {
        1: {'id': 1, 'parent_id': None, 'key': '1', 'code': 'def score(x): return x', 'fitness': 1},
        2: {'id': 2, 'parent_id': 1, 'key': '2', 'code': 'def score(x): return x+1', 'fitness': 2},
        3: {'id': 3, 'parent_id': 2, 'key': '3', 'code': 'def score(x): return x+2', 'fitness': 3},
        4: {'id': 4, 'parent_id': None, 'key': '4', 'code': 'def score(x): return x*2', 'fitness': 4},
    }
    picked, info = choose_reference(nodes[2], nodes, random.Random(0))
    assert picked['id'] == 4 and not info['relaxed_lineage']
    picked, info = choose_reference(nodes[2], {k: v for k, v in nodes.items() if k != 4},
                                    random.Random(0))
    assert picked['id'] == 3 and info['relaxed_lineage']


def test_explore_references_exclude_lineage_and_duplicate_visible_ideas():
    from tests.method.test_traceaad_v1015_prompts import node

    root = node(1)
    parent = node(2, parent=root)
    descendant = node(3, parent=parent)
    others = [node(i) for i in range(4, 10)]
    others[0]['idea'] = 'Use capacity slack.'
    others[1]['idea'] = '  USE  capacity slack. '
    others[2]['idea'] = ''
    others[3]['idea'] = parent['idea']
    others[4]['code'] = 'def score(x):\n    return min(abs(x), 5)\n'
    archive = {n['id']: n for n in [root, parent, descendant, *others]}
    references, info = choose_explore_references(parent, archive, random.Random(7))
    assert not info['relaxed_lineage']
    assert {n['id'] for n in references} == {5, 8, 9}
    assert references[0]['id'] == 8
    again, _ = choose_explore_references(parent, archive, random.Random(7))
    assert [n['id'] for n in references] == [n['id'] for n in again]
    lineage = {n['id']: n for n in [root, parent, descendant]}
    relaxed, info = choose_explore_references(parent, lineage, random.Random(7))
    assert relaxed and info['relaxed_lineage']


def test_score_classes_absorb_floating_point_noise_without_chaining():
    nodes = [{'id': 1, 'fitness': 14.704000000000002}, {'id': 2, 'fitness': 14.703999999999999},
             {'id': 3, 'fitness': 14.7}, {'id': 4, 'fitness': 14.704000000000002 - 5e-9}]
    classes = score_classes(nodes)
    assert [[n['id'] for n in c] for c in classes] == [[1, 2, 4], [3]]
    assert probabilities([c[0]['fitness'] for c in classes])[0] == [0.5, 0.5]
    # 1e-9-relative steps must not chain distinct scores into one class.
    ladder = [{'id': i, 'fitness': 1.0 + i * 6e-10} for i in range(5)]
    assert len(score_classes(ladder)) >= 2


def test_explore_references_compare_whole_ideas():
    from tests.method.test_traceaad_v1015_prompts import node

    parent = node(1)
    shared = 'Construct routes greedily from the depot and score every feasible customer. ' * 5
    a, b = node(2), node(3)
    a['idea'] = shared + 'Rank customers by regret insertion cost.'
    b['idea'] = shared + 'Rank customers by a capacity shadow price.'
    b['code'] = 'def score(x):\n    return abs(x) - 3\n'
    archive = {n['id']: n for n in (parent, a, b)}
    references, _ = choose_explore_references(parent, archive, random.Random(0))
    assert {n['id'] for n in references} == {2, 3}


def test_explore_references_take_one_card_per_score_class():
    from tests.method.test_traceaad_v1015_prompts import node

    parent = node(1)
    others = [node(i) for i in range(2, 6)]
    for i, other in enumerate(others):
        other['idea'] = f'Distinct idea {i}.'
        other['code'] = f'def score(x):\n    return x + {i} * {i}\n'
    others[1]['fitness'] = others[2]['fitness'] = 42.0  # reworded, same behaviour
    archive = {n['id']: n for n in [parent, *others]}
    references, _ = choose_explore_references(parent, archive, random.Random(0))
    assert len(references) == 3
    assert [n['fitness'] for n in references].count(42.0) == 1


def test_search_explore_displays_and_records_archive_references(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 10)), budget=9, explore_cards=4)
    for _ in range(8):
        m._roots()
    m.phase = 'search'
    m.action_rng.choices = lambda *args, **kwargs: ['Explore']
    m._search()
    request = m.facts.tables['request'][9]
    attempt = m.facts.tables['attempt'][9]
    assert len(request['explore_reference_ids']) == 4
    assert request['parent_id'] not in request['explore_reference_ids']
    assert attempt['explore_reference_ids'] == request['explore_reference_ids']
    assert request['history_edge_ids'] == [] and 'Earlier Ideas' not in request['prompt']
    assert request['explore_reference_selection']['selected_ids'] == request['explore_reference_ids']


def test_distinct_selection_protocol_is_required(tmp_path):
    with pytest.raises(ValueError, match='distinct frozen'):
        TraceAADV1015(evaluation=TinyEvaluation(), selection_evaluation=TinyEvaluation(),
                       llm=TokenLLM(), run_dir=tmp_path, config=Config(budget=1))


def test_prompt_contract_and_crossover_context_fallback():
    parent = {"id": 1, "key": "a", "code": "def score(x):\n    return x\n",
              "score": 1, "fitness": 1, "parent_id": None,
              "action": "Init", "idea": "Use the input.", "depth": 0}
    reference = {**parent, "id": 2, "key": "b", "code": "\n".join(
        [f"v{i} = {i}" for i in range(450)]) + "\ndef score(x): return x + v1\n"}
    archive = {1: parent, 2: reference}
    builder = PromptBuilder(TokenLLM(), None, TinyEvaluation(), archive,
                            Config(max_input_tokens=500))
    prompt = builder.build("Crossover", parent, reference=reference)
    assert prompt['action'] == 'Refine'
    assert 'crossover_context_fallback' in prompt['trims']
    assert '[Reference Algorithm]' not in prompt['prompt']
    assert '[How the Current Algorithm Was Formed]' in prompt['prompt']
    assert '[Your Task: Refine]' in prompt['prompt']
    assert '[Evaluation]' in prompt['prompt']


def test_resume_does_not_regenerate_completed_attempts(tmp_path):
    m = method(tmp_path, response(1), budget=1)
    assert m.run()['budget_used'] == 1
    resumed = method(tmp_path, budget=1)
    assert resumed.run()['budget_used'] == 1
    assert resumed.model_calls == 1


@pytest.mark.parametrize('task', [
    'tsp_construct', 'vrptw_construct', 'online_bin_packing', 'cvrp_aco', 'op_aco'])
def test_real_task_template_can_run_through_search_and_selection(tmp_path, task):
    from experiments.traceaad_v10_14_3.preflight import small_task

    train, selection = small_task(task), small_task(task, seed=11)
    if task in {'vrptw_construct', 'online_bin_packing'}:
        train.timeout_seconds = selection.timeout_seconds = 20
    candidate = f"Idea: use the supplied baseline\n```python\n{train.template_program.strip()}\n```"
    m = TraceAADV1015(evaluation=train, selection_evaluation=selection,
                       llm=TokenLLM(candidate), run_dir=tmp_path,
                       task=task, config=Config(budget=1))
    result = m.run()
    assert result['status'] == 'finished', result
    assert result['budget_used'] == result['model_calls'] == 1
    assert result['search_evaluations'] == result['selection_evaluations'] == 1
    assert result['best']['key'] == key((tmp_path / 'best_program.py').read_text())


def test_aco_wrong_output_shape_is_invalid_output(tmp_path):
    from experiments.traceaad_v10_14_3.preflight import small_task

    train = small_task('cvrp_aco')
    bad = ('Idea: return a small matrix\n```python\n'
           'import numpy as np\n'
           'def heuristics(distance_matrix, coordinates, demands, capacity):\n'
           '    return np.zeros((1, 1))\n```')
    m = TraceAADV1015(evaluation=train, llm=TokenLLM(bad), run_dir=tmp_path,
                       task='cvrp_aco', config=Config(budget=1))
    assert m.run()['status'] == 'no_valid_root'
    assert m.facts.tables['attempt'][1]['status'] == 'invalid_output'
