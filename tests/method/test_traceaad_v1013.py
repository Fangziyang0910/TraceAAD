import json
import random
import pytest
from pathlib import Path

from core import Evaluation
from traceaad.v10_13 import TraceAADV1013
from traceaad.v10_13.parsing import parse_candidate, template_target
from traceaad.v10_13.prompts import OPERATOR_INSTRUCTIONS, PromptBuilder
from traceaad.v10_13.selection import (
    PARENT_UNIFORM_PROBABILITY,
    quality_distribution,
    sample_parent,
    reference_shortlist,
)
from traceaad.v10_13.storage import read_journal
from traceaad.v10_13.tree import Node


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program="def score(x):\n    pass",
                         task_description="Return a numeric score.", safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 7

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"content": next(self.responses), "finish_reason": "stop", "usage": {}}


def response(value, heading="Idea:"):
    return (
        f"{heading} Try a direct decision rule that returns {value}.\n"
        "Code:\n```python\n"
        f"def score(x):\n    return {value}\n"
        "```"
    )


def make_method(path, llm=None, **kwargs):
    return TraceAADV1013(
        evaluation=TinyEvaluation(), llm=llm or FakeLLM(), run_dir=path,
        budget=kwargs.pop("budget", 1000), n_roots=kwargs.pop("n_roots", 1), **kwargs,
    )


def add_node(method, fitness, *, parent_id=None, idea="idea", code=None):
    code = code or f"def score(x):\n    return {fitness}"
    return method.tree.add(
        code=code, idea=idea, fitness=fitness,
        evaluation_id=len(method.tree.nodes) + 1, parent_id=parent_id,
        operator="Init",
    )


def test_quality_allocator_uses_v1010_ess_and_operator_independent_exploration():
    nodes = [Node(index, f"def score(x):\n    return {index}", "idea", index)
             for index in range(12)]
    probabilities, stats = quality_distribution(nodes)
    assert abs(stats["quality_ess"] - 8.0) < 1e-6
    _, selection = sample_parent(nodes, "Pivot", random.Random(0))
    assert selection["parent_ess"] > stats["quality_ess"]
    assert selection["parent_probability"] >= PARENT_UNIFORM_PROBABILITY / len(nodes)
    for operator in ('Refine', 'Tune', 'Fuse'):
        _, local = sample_parent(nodes, operator, random.Random(0))
        assert local['parent_probability'] == selection['parent_probability']


def test_reference_shortlist_excludes_parent_and_its_exact_copies():
    nodes = [
        Node(0, "def score(x):\n    return x", "a", 1),
        Node(1, "def score(x):\n    return x", "copy", 9),
        Node(2, "def score(x):\n    return x + 1", "b", 2),
    ]
    references, stats = reference_shortlist(nodes, nodes[0], random.Random(2))
    assert [node.id for node in references] == [2]
    assert stats['distinct_reference_pool'] == 1


def test_prompts_are_short_and_operator_specific_without_behavior_checklist(tmp_path):
    method = make_method(tmp_path, budget=1)
    parent = add_node(method, 3)
    donor = add_node(method, 4, code="def score(x):\n    return x + 1")
    builder = PromptBuilder(
        method.llm, method.task_contract, max_tokens=20000, history_depth=3,
        lookup=method.tree.nodes.get, all_nodes=method.tree.all_nodes,
    )
    text = builder.build(parent, "Fuse", donor)
    assert "Host Algorithm" in text and "Reference Algorithm" in text
    assert "Idea: idea" in text
    assert "Do not mechanically concatenate" in text
    assert "Change/Evidence/Behavior/Preserve" not in text
    assert "candidate ranking" not in text
    assert "promising improvement" in OPERATOR_INSTRUCTIONS["Refine"]
    assert "influential parameters" in OPERATOR_INSTRUCTIONS["Tune"]
    history = builder.build(parent, "Refine")
    assert "Recent Design History" not in history  # root has no parent edge


def test_parser_accepts_simple_markdown_idea_heading():
    interface, _ = template_target("def score(x):\n    pass")
    parsed, error = parse_candidate(response(4, "**Idea:**"), "stop", interface,
                                    "def score(x):\n    pass")
    assert error is None
    assert parsed.idea.startswith("Try a direct")


def test_repair_does_not_allocate_a_second_parent_opportunity(tmp_path):
    llm = FakeLLM(response(1), "bad output", response(3))
    method = make_method(tmp_path, llm, budget=2)
    method.run()
    events = read_journal(method.storage.events_path)
    assert [event["status"] for event in events] == ["ok", "invalid_output", "ok"]
    assert events[1]["parent_selected"] is True
    assert events[2]["parent_selected"] is False
    assert method.tree.parent_selections == 1


def test_checkpoint_contains_only_evolutionary_state_and_resumes(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    method.run()
    state = json.loads((Path(tmp_path) / "tree_state.json").read_text())
    assert state["mechanism"]["method"] == "v1013"
    assert "route_stats" not in state
    assert all("attempts" in node for node in state["nodes"])

    resumed = make_method(tmp_path, FakeLLM(), budget=2)
    resumed.run()
    assert resumed.llm.calls == []
    assert resumed.tree.parent_selections == method.tree.parent_selections


def test_operator_instructions_are_algorithmic_moves():
    assert "promising improvement" in OPERATOR_INSTRUCTIONS["Refine"]
    assert "Calibrate influential parameters" in OPERATOR_INSTRUCTIONS["Tune"]
    assert "central assumption" in OPERATOR_INSTRUCTIONS["Pivot"]
    assert "complementary signal" in OPERATOR_INSTRUCTIONS["Fuse"]


def test_idea_never_blocks_valid_code_and_supports_small_structured_fields():
    interface, _ = template_target('def score(x):\n    pass')
    for metadata in (None, 'long ' * 2000, {'mechanism': 'use x', 'change': 'add one',
                                          'transfer': 'a reusable offset', 'unknown': 12}):
        parsed, error = parse_candidate(json.dumps({'mode': 'full', 'idea': metadata,
            'code': 'def score(x):\n    return x + 1'}), 'stop', interface, 'def score(x):\n    pass')
        assert error is None and 'return x + 1' in parsed.program_code
    parsed, error = parse_candidate('```python\ndef score(x):\n    return x\n```',
                                    'stop', interface, 'def score(x):\n    pass')
    assert not error and parsed.idea == ''


def test_edit_mode_exact_base_unique_blocks_and_interface():
    from traceaad.v10_13.parsing import code_hash
    base = 'def score(x):\n    return x + 1'
    interface, _ = template_target('def score(x):\n    pass')
    request = {'mode': 'edit', 'base_hash': code_hash(base),
               'edits': [{'search': 'return x + 1', 'replacement': 'return x + 2'}]}
    def parse(value, code=base):
        return parse_candidate(json.dumps(value), 'stop', interface,
                               'def score(x):\n    pass', base_code=code)
    parsed, error = parse(request)
    assert not error and parsed.mode == 'edit' and 'x + 2' in parsed.program_code
    assert parse({**request, 'base_hash': 'wrong'})[1].startswith('edit_error')
    assert parse({**request, 'edits': [{'search': 'x', 'replacement': 'y'}]})[1].startswith('edit_error')
    assert parse({**request, 'edits': [{'search': 'def score(x)', 'replacement': 'def score(y)'}]})[1].startswith('signature_error')
    assert parse({**request, 'code': base})[1].startswith('edit_error')
    without_hash = {k: v for k, v in request.items() if k != 'base_hash'}
    parsed, error = parse(without_hash)
    assert not error and parsed.base_hash == code_hash(base)


def test_one_reference_mixes_quality_and_uniform_without_semantic_categories():
    from traceaad.v10_13.selection import reference_shortlist
    nodes = [Node(i, f'def score(x):\n    return x + {i}', f'mechanism {i}', float(i)) for i in range(30)]
    selected, stats = reference_shortlist(nodes, nodes[0], random.Random(2))
    assert len(selected) == 1
    assert set(stats['reference_sources'].values()) <= {'quality', 'uniform'}
    assert all(n.id != 0 for n in selected)
    observed = {n.id for seed in range(100) for n in
                reference_shortlist(nodes, nodes[0], random.Random(seed))[0]}
    assert min(observed) < 5 and max(observed) == 29


def test_local_trials_are_inline_compact_and_exact_parent_specific(tmp_path):
    method = make_method(tmp_path)
    parent = add_node(method, 3)
    good = add_node(method, 4, parent_id=parent.id, idea='raise offset')
    bad = add_node(method, 2, parent_id=parent.id, idea='lower offset')
    outsider = add_node(method, 9, idea='unrelated')
    add_node(method, 10, parent_id=outsider.id, idea='do not show')
    read = method.prompts.build_development(parent, 'Tune')
    assert 'Optional context' not in read.prompt
    assert read.trial_ids == [good.id, bad.id]
    assert 'do not show' not in read.prompt and 'raise offset' in read.prompt
    assert '```diff' not in read.prompt
    assert good.code not in read.prompt


class ScriptedLLM(FakeLLM):
    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        value = next(self.responses)
        return {'content': value(prompt) if callable(value) else value,
                'finish_reason': 'stop', 'usage': {}}


def test_inline_context_and_edit_cost_one_call_and_parent_opportunity(tmp_path):
    def edit(prompt):
        assert 'base_hash' not in prompt and 'mode":"context' not in prompt
        return json.dumps({'mode': 'edit',
                          'edits': [{'search': 'return 1', 'replacement': 'return 2'}]})
    llm = ScriptedLLM(response(1), edit)
    method = make_method(tmp_path, llm, budget=2)
    method.run()
    events = read_journal(method.storage.events_path)
    assert len(events) == 2 and events[-1]['context_reads'] == 0
    assert events[-1]['output_mode'] == 'edit'
    assert method.tree.parent_selections == 1 and method.evaluations_used == 2
    assert len(llm.calls) == 2
    assert method.tree.best().fitness == 2


class CountingEvaluation(TinyEvaluation):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return super().evaluate_program(program_str, callable_func, **kwargs)


@pytest.mark.parametrize('point', ['record_call', 'record_evaluation', 'record_node', 'record_event',
                                   'scheduled', 'generated', 'parsed', 'evaluated', 'committing', 'clear'])
def test_resume_does_not_repeat_llm_or_evaluation_after_durable_stage(tmp_path, point):
    evaluation = CountingEvaluation()
    llm = FakeLLM(response(1), response(2))
    method = TraceAADV1013(evaluation=evaluation, llm=llm, run_dir=tmp_path, budget=2, n_roots=1)
    fired = False
    if point.startswith('record_'):
        original = getattr(method.storage, point)
        def crash(record):
            nonlocal fired
            original(record)
            identity = record.id if point == 'record_node' else record.get('candidate_id')
            if identity == (1 if point == 'record_node' else 2) and not fired:
                fired = True
                raise OSError('injected after durable write')
        setattr(method.storage, point, crash)
    else:
        original = method._save_checkpoint
        def crash():
            nonlocal fired
            original()
            pending = method.pending
            stage = pending['stage'] if pending else 'clear'
            identity = pending['candidate']['candidate_id'] if pending else method.completed_candidates
            if identity == 2 and stage == point and not fired:
                fired = True
                raise OSError('injected after checkpoint')
        method._save_checkpoint = crash
    with pytest.raises(OSError):
        method.run()
    resumed = TraceAADV1013(evaluation=evaluation, llm=llm, run_dir=tmp_path, budget=2, n_roots=1)
    resumed.run()
    assert fired and evaluation.calls == 2 and len(llm.calls) == 2
    assert [e['evaluation_id'] for e in read_journal(resumed.storage.events_path)] == [1, 2]
    assert [n['id'] for n in read_journal(resumed.storage.nodes_path)] == [0, 1]
    assert resumed.tree.parent_selections == 1


def test_missing_evaluation_receipt_blocks_automatic_retry(tmp_path):
    from traceaad.v10_13.traceaad import UncertainEvaluationError
    evaluation = CountingEvaluation()
    llm = FakeLLM(response(1))
    method = TraceAADV1013(evaluation=evaluation, llm=llm, run_dir=tmp_path, budget=1, n_roots=1)
    def missing_receipt(record):
        raise OSError('lost before durable receipt')
    method.storage.record_evaluation = missing_receipt
    with pytest.raises(OSError):
        method.run()
    resumed = TraceAADV1013(evaluation=evaluation, llm=llm, run_dir=tmp_path, budget=1, n_roots=1)
    with pytest.raises(UncertainEvaluationError):
        resumed.run()
    assert evaluation.calls == 1 and len(llm.calls) == 1
    assert resumed.evaluations_used == 1
    summary = json.loads(resumed.storage.summary_path.read_text())
    assert summary['status'] == 'uncertain_evaluation'
    assert summary['budget_reserved'] == 1 and summary['search_evaluations'] == 0
    assert summary['evaluation_calls_with_receipts'] == 0


def test_old_checkpoint_rejected_before_journal_mutation(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    method.run()
    state = json.loads(method.storage.state_path.read_text())
    state['mechanism']['prompt_policy'] = 'old_frozen_revision'
    method.storage.state_path.write_text(json.dumps(state))
    method.storage.events_path.write_bytes(method.storage.events_path.read_bytes() + b'torn')
    before = method.storage.events_path.read_bytes()
    with pytest.raises(ValueError, match='different revision'):
        make_method(tmp_path, budget=1).run()
    assert method.storage.events_path.read_bytes() == before


def test_edit_preserves_exact_program_order_and_can_remove_template_import():
    from traceaad.v10_13.parsing import code_hash
    template = 'import math\n\ndef score(x):\n    pass'
    base = 'import math\n\ndef score(x):\n    return x\n\nexample = score(1)'
    interface, _ = template_target(template)
    parsed, error = parse_candidate(json.dumps({
        'mode': 'edit', 'base_hash': code_hash(base),
        'edits': [{'search': 'import math\n\n', 'replacement': ''}],
    }), 'stop', interface, template, base_code=base)
    assert not error and 'import math' not in parsed.program_code
    scope = {}
    exec(parsed.program_code, scope)
    assert scope['example'] == 1


def test_reference_sampling_is_not_restricted_to_elites():
    nodes = [Node(i, f'def score(x):\n    return {i}', 'idea', float(i)) for i in range(40)]
    seen, sources = set(), set()
    for seed in range(100):
        chosen, stats = reference_shortlist(nodes, nodes[-1], random.Random(seed))
        seen.add(chosen[0].id)
        sources.update(stats['reference_sources'].values())
    assert sources == {'quality', 'uniform'} and min(seen) < 5 and max(seen) == 38


def test_initial_and_repair_prompts_only_offer_full_output(tmp_path):
    from traceaad.v10_13.parsing import build_repair_prompt
    method = make_method(tmp_path)
    assert '"edits"' not in method.prompts.build_initial()
    repair = build_repair_prompt('task', json.dumps({
        'mode': 'full', 'idea': 'irrelevant ' * 10000,
        'code': 'def score(x):\n    return broken',
    }), {'error': 'undefined name'}, base_code='large unrelated parent')
    assert '"edits"' not in repair and 'base_hash' not in repair
    assert 'irrelevant' not in repair and 'large unrelated parent' not in repair
    assert 'return broken' in repair


@pytest.mark.parametrize('wrapper', [
    'Explanation.\n```json\n{}\n```',
    '```json\n{}',
    'Explanation.\n<tool_call>\n{}\n</tool_call>',
    'Draft:\n```python\ndef score(x):\n    return x - 1\n```\n'
    'Draft two:\n```python\ndef score(x):\n    return x - 2\n```\nFinal:\n{}',
])
def test_unique_wrapped_proposal_preserves_exact_code(wrapper):
    template = 'def score(x):\n    pass'
    interface, _ = template_target(template)
    payload = json.dumps({'mode': 'full', 'code': 'def score(x):\n    return x + 7'})
    parsed, error = parse_candidate(wrapper.format(payload), 'stop', interface, template)
    assert not error and parsed.program_code == 'def score(x):\n    return x + 7'


def test_conflicting_json_proposals_are_not_silently_selected():
    template = 'def score(x):\n    pass'
    interface, _ = template_target(template)
    first = json.dumps({'mode': 'full', 'code': 'def score(x):\n    return x'})
    second = json.dumps({'mode': 'full', 'code': 'def score(x):\n    return x + 1'})
    parsed, error = parse_candidate(first + '\n' + second, 'stop', interface, template)
    assert parsed is None and 'conflicting JSON' in error
    # Repeating the identical object is not a conflicting implementation.
    assert parse_candidate(first + '\n' + first, 'stop', interface, template)[1] is None


def test_existing_single_python_block_keeps_code_first_interpretation():
    template = 'def score(x):\n    pass'
    interface, _ = template_target(template)
    other = json.dumps({'mode': 'full', 'code': 'def score(x):\n    return x + 2'})
    text = 'Idea: direct rule\n```python\ndef score(x):\n    return x + 1\n```\n' + other
    parsed, error = parse_candidate(text, 'stop', interface, template)
    assert not error and parsed.program_code == 'def score(x):\n    return x + 1'


def test_code_string_json_is_not_a_second_proposal_and_bad_code_stays_invalid():
    template = 'def score(x):\n    pass'
    interface, _ = template_target(template)
    code = 'def score(x):\n    text = \'{"mode":"edit","edits":[]}\'\n    return x'
    payload = json.dumps({'mode': 'full', 'code': code})
    assert parse_candidate('Final: ' + payload, 'stop', interface, template)[1] is None
    assert parse_candidate(code, 'stop', interface, template)[1] is None
    assert parse_candidate('```python\n' + code + '\n```', 'stop', interface, template)[1] is None
    bad = json.dumps({'mode': 'full', 'code': 'def score(x):\n    return -1e6.0'})
    assert parse_candidate('Final: ' + bad, 'stop', interface, template)[1].startswith('syntax_error')


def test_missing_closing_python_fence_accepts_complete_code_only():
    template = 'def score(x):\n    pass'
    interface, _ = template_target(template)
    assert parse_candidate('```python\ndef score(x):\n    return x', 'stop', interface, template)[1] is None
    assert parse_candidate('```python\ndef score(x):\n    return (', 'length', interface, template)[1].startswith('syntax_error')


def test_context_material_matches_operator_without_read_requests(tmp_path):
    method = make_method(tmp_path)
    parent = add_node(method, 1)
    local = method.prompts.build_development(parent, 'Refine')
    assert 'Local trials' not in local.prompt and '"mode":"context"' not in local.prompt
    donor = add_node(method, 2)
    cross = method.prompts.build_development(parent, 'Fuse', references=[donor])
    assert donor.code in cross.prompt and cross.reference_program == donor
    assert '"mode":"context"' not in cross.prompt
    pivot = method.prompts.build_development(parent, 'Pivot', references=[donor])
    assert donor.code not in pivot.prompt and not pivot.reference_ids


@pytest.mark.parametrize('point', ['none', 'call_journal', 'call_scheduled'])
def test_inline_reference_and_resume_preserve_one_parent_opportunity(tmp_path, point):
    import re
    evaluation = CountingEvaluation()
    selected = []
    def propose(prompt):
        assert '# Reference Algorithm' in prompt and 'mode":"context' not in prompt
        donor_id = int(re.search(r'Node (\d+)', prompt.split('# Reference Algorithm')[1]).group(1))
        selected.append(donor_id)
        return json.dumps({'mode': 'full', 'donor_id': donor_id,
                           'code': 'def score(x):\n    return 3'})
    llm = ScriptedLLM(response(1), response(2), propose)
    def create():
        return TraceAADV1013(evaluation=evaluation, llm=llm, run_dir=tmp_path,
                            budget=3, n_roots=2, seed=0)
    method = create()
    if point == 'call_journal':
        original = method.storage.record_call
        def crash(record):
            original(record)
            if record['call_id'] == '3:1':
                raise OSError('after response')
        method.storage.record_call = crash
    elif point == 'call_scheduled':
        original = method._save_checkpoint
        def crash():
            original()
            pending = method.pending
            if pending and pending['stage'] == 'scheduled' and pending['candidate']['candidate_id'] == 3:
                raise OSError('before call')
        method._save_checkpoint = crash
    if point != 'none':
        with pytest.raises(OSError):
            method.run()
        method = create()
    method.run()
    event = read_journal(method.storage.events_path)[-1]
    assert event['operator'] == 'Fuse'
    assert event['donor_id'] == event['loaded_reference_id'] == selected[0]
    assert event['context_reads'] == 0 and event['llm_calls'] == 1
    assert method.tree.parent_selections == 1 and evaluation.calls == 3 and len(llm.calls) == 3


def test_obsolete_context_request_gets_one_repair_without_another_parent(tmp_path):
    request = json.dumps({'mode': 'context', 'trials': True})
    method = make_method(tmp_path, ScriptedLLM(response(1), request, response(2)), budget=2)
    method.run()
    events = read_journal(method.storage.events_path)
    assert [e['status'] for e in events] == ['ok', 'invalid_output', 'ok']
    assert events[1]['reason'].startswith('context_error')
    assert events[-1]['repair_of'] == 2 and not events[-1]['parent_selected']
    assert method.tree.parent_selections == 1 and method.evaluations_used == 2


def test_failed_edit_execution_repairs_reconstructed_code_without_reattributing_parent(tmp_path):
    def edit(prompt):
        return json.dumps({'mode': 'edit',
            'edits': [{'search': 'return 1', 'replacement': 'return 1 / 0'}]})
    def repair(prompt):
        assert 'return 1 / 0' in prompt and '# Edit base' not in prompt
        assert 'Local edit:' not in prompt
        return response(2)
    method = make_method(tmp_path, ScriptedLLM(response(1), edit, repair), budget=3)
    method.run()
    events = read_journal(method.storage.events_path)
    assert [e['status'] for e in events] == ['ok', 'eval_failed', 'ok']
    assert events[-1]['parent_id'] == events[-2]['parent_id'] == 0
    assert method.tree.parent_selections == 1 and method.evaluations_used == 3


def test_context_capacity_drops_whole_reference_not_code_fragment(tmp_path):
    method = make_method(tmp_path)
    parent = add_node(method, 1)
    donor = add_node(method, 2, code='def score(x):\n' + '    # long comment\n' * 1000 + '    return 2')
    baseline = method.prompts.build_development(parent, 'Fuse')
    method.prompts.max_tokens = method.prompts.count(baseline.prompt) + 30
    read = method.prompts.build_development(parent, 'Fuse', references=[donor])
    assert read.reference_program is None
    assert read.fallback_reason == 'reference_implementation_exceeds_context'
    assert 'long comment' not in read.prompt and parent.code in read.prompt
    assert method.prompts.count(read.prompt) <= method.prompts.max_tokens


def test_writer_lock_and_journal_conflicts_fail_closed(tmp_path):
    from traceaad.v10_13.storage import RunStorage
    first, second = RunStorage(tmp_path), RunStorage(tmp_path)
    with first.writer_lock():
        with pytest.raises(RuntimeError, match='another writer'):
            with second.writer_lock():
                pass
    first.record_evaluation({'evaluation_id': 1, 'candidate_id': 1})
    first.record_evaluation({'evaluation_id': 1, 'candidate_id': 1})
    with pytest.raises(ValueError, match='conflicting'):
        first.record_evaluation({'evaluation_id': 1, 'candidate_id': 2})
    assert len(read_journal(first.evaluations_path)) == 1


def test_torn_final_record_preserved_before_recovery(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    method.run()
    path = method.storage.events_path
    path.write_bytes(path.read_bytes() + b'{"incomplete')
    resumed = make_method(tmp_path, budget=1)
    resumed.run()
    assert len(read_journal(path)) == 1 and resumed.llm.calls == []
    assert next(tmp_path.glob('events.jsonl.torn-*')).read_bytes() == b'{"incomplete'


@pytest.mark.parametrize('journal', ['events.jsonl', 'nodes.jsonl', 'evaluations.jsonl'])
def test_missing_committed_journal_record_is_detected(tmp_path, journal):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    method.run()
    (tmp_path / journal).write_text('')
    with pytest.raises(ValueError, match='inconsistent'):
        make_method(tmp_path, budget=1).run()


def test_repair_does_not_claim_inherited_donor_without_new_declaration(tmp_path):
    import re
    def failure(prompt):
        donor_id = int(re.search(r'Node (\d+)', prompt.split('# Reference Algorithm')[1]).group(1))
        return json.dumps({'mode': 'full', 'donor_id': donor_id,
                           'code': 'def score(x):\n    return 1 / 0'})
    method = make_method(tmp_path, ScriptedLLM(response(1), response(2), failure, response(3)),
                         budget=4, n_roots=2, seed=0)
    method.run()
    events = read_journal(method.storage.events_path)
    assert events[-2]['donor_id'] is not None
    assert events[-1]['donor_id'] is None and events[-1]['donor_fitness'] is None
    assert events[-1]['repair_of'] == 3 and events[-1]['context_reads'] == 0
    assert method.tree.parent_selections == 1


def test_infrastructure_failure_at_budget_limit_cannot_be_marked_finished(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    def unavailable(code):
        raise RuntimeError('evaluator infrastructure unavailable')
    method.evaluator.evaluate_program_with_details = unavailable
    with pytest.raises(RuntimeError, match='infrastructure failed'):
        method.run()
    assert method.evaluations_used == 1
    assert json.loads(method.storage.summary_path.read_text())['status'] == 'error'


@pytest.mark.parametrize('kind', ['prepare_error', 'infrastructure_error'])
@pytest.mark.parametrize('crash_point', [None, 'receipt', 'evaluated', 'committing', 'clear'])
def test_confirmed_infrastructure_failure_releases_budget_once_and_stays_blocked(
        tmp_path, kind, crash_point):
    from core.evaluate import EvaluationOutcome
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    calls = []
    def unavailable(code):
        calls.append(code)
        return EvaluationOutcome(None, failure_kind=kind, error='worker startup failed')
    method.evaluator.evaluate_program_with_details = unavailable
    if crash_point == 'receipt':
        original = method.storage.record_evaluation
        def crash(record):
            original(record)
            raise OSError('after durable receipt')
        method.storage.record_evaluation = crash
    elif crash_point in ('evaluated', 'committing', 'clear'):
        original = method._save_checkpoint
        def crash():
            original()
            stage = method.pending['stage'] if method.pending else 'clear'
            if method.evaluation_attempts and stage == crash_point:
                raise OSError('after fault checkpoint')
        method._save_checkpoint = crash
    with pytest.raises((OSError, RuntimeError)):
        method.run()
    for _ in range(2):
        resumed = make_method(tmp_path, FakeLLM(), budget=1)
        with pytest.raises(RuntimeError, match='infrastructure failed'):
            resumed.run()
        assert resumed.evaluations_used == 0
        summary = json.loads(resumed.storage.summary_path.read_text())
        assert summary['search_evaluations'] == summary['budget_reserved'] == 0
        assert summary['infrastructure_failures'] == summary['evaluation_attempts'] == 1
        assert summary['evaluation_calls_with_receipts'] == 1
        assert summary['status'] == 'error'
    assert len(calls) == 1
    assert len(read_journal(resumed.storage.evaluations_path)) == 1
    assert len(read_journal(resumed.storage.events_path)) == 1


@pytest.mark.parametrize('kind', ['exec_error', 'runtime_error', 'timeout', 'invalid_result',
                                 'nonfinite_fitness'])
def test_candidate_failures_still_cost_one_evaluation(tmp_path, kind):
    from core.evaluate import EvaluationOutcome
    method = make_method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    results = iter([EvaluationOutcome(1), EvaluationOutcome(None, failure_kind=kind)])
    method.evaluator.evaluate_program_with_details = lambda code: next(results)
    method.run()
    summary = json.loads(method.storage.summary_path.read_text())
    assert summary['budget_used'] == summary['search_evaluations'] == 2
    assert summary['budget_reserved'] == summary['infrastructure_failures'] == 0


@pytest.mark.parametrize('crash_after_resolution', [False, True])
def test_unclassified_exception_needs_evidence_and_durable_confirmation(tmp_path, crash_after_resolution):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    def unavailable(code):
        raise RuntimeError('unclassified evaluator exception')
    method.evaluator.evaluate_program_with_details = unavailable
    with pytest.raises(RuntimeError, match='infrastructure failed'):
        method.run()
    summary = json.loads(method.storage.summary_path.read_text())
    assert summary['budget_used'] == summary['budget_reserved'] == 1
    assert summary['search_evaluations'] == summary['infrastructure_failures'] == 0
    with pytest.raises(ValueError, match='evidence'):
        method.confirm_infrastructure_failure(1, evidence=' ')
    before = method.storage.evaluations_path.read_bytes()
    if crash_after_resolution:
        original = method._save_checkpoint
        def crash():
            raise OSError('resolution journal persisted before checkpoint')
        method._save_checkpoint = crash
        with pytest.raises(OSError):
            method.confirm_infrastructure_failure(1, evidence='diagnosis: worker service unavailable')
        method._save_checkpoint = original
    method.confirm_infrastructure_failure(1, evidence='diagnosis: worker service unavailable')
    # Repeating the same adjudication is idempotent, even after a torn checkpoint transition.
    method.confirm_infrastructure_failure(1, evidence='diagnosis: worker service unavailable')
    resumed = make_method(tmp_path, FakeLLM(), budget=1)
    with pytest.raises(RuntimeError, match='infrastructure failed'):
        resumed.run()
    assert resumed.evaluations_used == 0 and resumed.evaluation_attempts == 1
    assert resumed.storage.evaluations_path.read_bytes() == before
    assert len(read_journal(resumed.storage.evaluation_resolutions_path)) == 1
    assert resumed.llm.calls == []


def test_successful_evaluation_cannot_be_exempted_as_infrastructure(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    method.run()
    with pytest.raises(ValueError, match='failed evaluation'):
        method.confirm_infrastructure_failure(1, evidence='not a fault')
    assert method.evaluations_used == 1


def test_evaluation_ids_do_not_reuse_released_budget(tmp_path):
    from core.evaluate import EvaluationOutcome
    method = make_method(tmp_path, FakeLLM(), budget=2)
    candidate = method._schedule_candidate()
    parsed, error = method._parse_completion({'response': response(1), 'finish_reason': 'stop'}, candidate)
    assert error is None
    method.pending = {'stage': 'parsed'}
    method.evaluator.evaluate_program_with_details = lambda code: EvaluationOutcome(
        None, failure_kind='prepare_error')
    outcome, _ = method._evaluate_and_add(parsed, candidate)
    assert outcome['evaluation_id'] == 1 and method.evaluations_used == 0
    # Exercise dispatch identity independently of the operational fault block.
    method.evaluator.evaluate_program_with_details = lambda code: EvaluationOutcome(1)
    method.pending = {'stage': 'parsed'}
    outcome, _ = method._evaluate_and_add(parsed, candidate)
    assert outcome['evaluation_id'] == 2 and method.evaluations_used == 1
    assert [r['evaluation_id'] for r in read_journal(method.storage.evaluations_path)] == [1, 2]


def test_secure_evaluator_preparation_fault_is_not_a_candidate_cost(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)), budget=1)
    def broken_template():
        raise RuntimeError('test evaluator setup fault')
    method.evaluator._target_function_name = broken_template
    with pytest.raises(RuntimeError, match='infrastructure failed'):
        method.run()
    assert method.evaluations_used == 0
    assert read_journal(method.storage.evaluations_path)[0]['reason'] == 'prepare_error'


def test_unclassified_last_evaluation_remains_blocked_on_resume_at_budget_limit(tmp_path):
    from core.evaluate import EvaluationOutcome
    method = make_method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    calls = []
    def evaluate(code):
        calls.append(code)
        if len(calls) == 1:
            return EvaluationOutcome(1)
        raise RuntimeError('unclassified exception')
    method.evaluator.evaluate_program_with_details = evaluate
    with pytest.raises(RuntimeError, match='infrastructure failed'):
        method.run()
    resumed = make_method(tmp_path, FakeLLM(), budget=2)
    with pytest.raises(RuntimeError, match='infrastructure failed'):
        resumed.run()
    assert json.loads(resumed.storage.summary_path.read_text())['status'] == 'error'
    assert len(calls) == 2 and resumed.llm.calls == []


def test_runtime_failure_diagnosed_as_infrastructure_does_not_trigger_candidate_repair(tmp_path):
    from core.evaluate import EvaluationOutcome
    method = make_method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    results = iter([EvaluationOutcome(1), EvaluationOutcome(None, failure_kind='runtime_error')])
    method.evaluator.evaluate_program_with_details = lambda code: next(results)
    method.run()
    method.confirm_infrastructure_failure(2, evidence='diagnosis: task data service failed')
    resumed = make_method(tmp_path, FakeLLM(), budget=2)
    with pytest.raises(RuntimeError, match='infrastructure failed'):
        resumed.run()
    assert resumed.evaluations_used == 1 and resumed.llm.calls == []
