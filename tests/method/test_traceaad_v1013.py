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
    assert "Host Algorithm" in text and "Optional reference ideas" in text
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


def test_shortlist_covers_sources_without_assuming_semantic_categories():
    from traceaad.v10_13.selection import reference_shortlist
    nodes = [Node(i, f'def score(x):\n    return x + {i}', f'mechanism {i}', float(i)) for i in range(30)]
    selected, stats = reference_shortlist(nodes, nodes[0], random.Random(2))
    assert len(selected) == len({n.id for n in selected}) == 3
    assert set(stats['reference_sources'].values()) == {'quality', 'uniform', 'underexposed'}
    assert all(n.id != 0 for n in selected)
    observed = {n.id for seed in range(100) for n in
                reference_shortlist(nodes, nodes[0], random.Random(seed))[0]}
    assert min(observed) < 5 and max(observed) == 29


def test_local_trials_are_optional_compact_and_exact_parent_specific(tmp_path):
    method = make_method(tmp_path)
    parent = add_node(method, 3)
    good = add_node(method, 4, parent_id=parent.id, idea='raise offset')
    bad = add_node(method, 2, parent_id=parent.id, idea='lower offset')
    outsider = add_node(method, 9, idea='unrelated')
    add_node(method, 10, parent_id=outsider.id, idea='do not show')
    normal = method.prompts.build_development(parent, 'Tune')
    read = method.prompts.build_development(parent, 'Tune', include_trials=True, allow_context=False)
    assert not normal.trial_ids
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


def test_context_read_and_edit_cost_one_parent_opportunity(tmp_path):
    import re
    def edit(prompt):
        assert 'Submit a full/edit proposal now' in prompt
        base_hash = re.search(r'base_hash: ([a-f0-9]{64})', prompt).group(1)
        return json.dumps({'mode': 'edit', 'base_hash': base_hash,
                          'edits': [{'search': 'return 1', 'replacement': 'return 2'}]})
    llm = ScriptedLLM(response(1), json.dumps({'mode': 'context', 'trials': True}), edit)
    method = make_method(tmp_path, llm, budget=2)
    method.run()
    events = read_journal(method.storage.events_path)
    assert len(events) == 2 and events[-1]['context_reads'] == 1
    assert events[-1]['output_mode'] == 'edit'
    assert method.tree.parent_selections == 1 and method.evaluations_used == 2
    assert len(llm.calls) == 3
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
    assert json.loads(resumed.storage.summary_path.read_text())['status'] == 'uncertain_evaluation'


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


def test_duplicate_implementation_does_not_reset_reference_exposure():
    class InspectRandom:
        def __init__(self):
            self.weights = []
        def choices(self, options, weights):
            self.weights.append(list(weights))
            return [0]
        def shuffle(self, values):
            pass
    nodes = [Node(i, f'def score(x):\n    return {i}', 'idea', 1) for i in range(5)]
    nodes[3].reference_uses = 20
    nodes.append(Node(5, nodes[3].code, 'same program', 1))
    rng = InspectRandom()
    reference_shortlist(nodes, nodes[0], rng)
    assert rng.weights[-1] == [1 / 21, 1]


def test_initial_and_repair_prompts_only_offer_full_output(tmp_path):
    from traceaad.v10_13.parsing import build_repair_prompt
    method = make_method(tmp_path)
    assert 'Local edit:' not in method.prompts.build_initial()
    repair = build_repair_prompt('task', json.dumps({
        'mode': 'full', 'idea': 'irrelevant ' * 10000,
        'code': 'def score(x):\n    return broken',
    }), {'error': 'undefined name'}, base_code='large unrelated parent')
    assert 'Local edit:' not in repair and 'base_hash' not in repair
    assert 'irrelevant' not in repair and 'large unrelated parent' not in repair
    assert 'return broken' in repair


def test_context_material_matches_operator_and_reports_absent_trials(tmp_path):
    method = make_method(tmp_path)
    parent = add_node(method, 1)
    local = method.prompts.build_development(parent, 'Refine')
    assert '"trials":true' in local.prompt and '"reference_id":ID' not in local.prompt
    empty = method.prompts.build_development(parent, 'Refine', include_trials=True, allow_context=False)
    assert 'No evaluated child records' in empty.prompt
    donor = add_node(method, 2)
    cross = method.prompts.build_development(parent, 'Fuse', references=[donor])
    assert '"reference_id":ID' in cross.prompt and '"trials":true' not in cross.prompt


@pytest.mark.parametrize('point', ['none', 'second_call_journal', 'second_call_scheduled'])
def test_selected_donor_read_and_resume_preserve_one_parent_opportunity(tmp_path, point):
    import re
    evaluation = CountingEvaluation()
    selected = []
    def request(prompt):
        cards = prompt.split('# Optional reference ideas')[1]
        donor_id = int(re.search(r'Node (\d+)', cards).group(1))
        selected.append(donor_id)
        return json.dumps({'mode': 'context', 'reference_id': donor_id})
    def propose(prompt):
        assert '# Requested Reference Implementation' in prompt
        assert f'Node {selected[0]} |' in prompt.split('# Requested Reference Implementation')[1]
        assert 'Local trials' not in prompt
        return json.dumps({'mode': 'full', 'donor_id': selected[0],
                           'code': 'def score(x):\n    return 3'})
    llm = ScriptedLLM(response(1), response(2), request, propose)
    def create():
        # Seed 0 chooses Fuse on the first development call.
        return TraceAADV1013(evaluation=evaluation, llm=llm, run_dir=tmp_path,
                            budget=3, n_roots=2, seed=0)
    method = create()
    if point == 'second_call_journal':
        original = method.storage.record_call
        def crash(record):
            original(record)
            if record['call_id'] == '3:2':
                raise OSError('after second-round response')
        method.storage.record_call = crash
    elif point == 'second_call_scheduled':
        original = method._save_checkpoint
        def crash():
            original()
            pending = method.pending
            if pending and pending['stage'] == 'scheduled' and pending['candidate']['context_reads'] == 1:
                raise OSError('before second-round call')
        method._save_checkpoint = crash
    if point != 'none':
        with pytest.raises(OSError):
            method.run()
        method = create()
    method.run()
    events = read_journal(method.storage.events_path)
    event = events[-1]
    assert event['operator'] == 'Fuse'
    assert event['donor_id'] == event['loaded_reference_id'] == selected[0]
    assert event['context_reads'] == 1 and event['llm_calls'] == 2
    assert method.tree.parent_selections == 1 and evaluation.calls == 3 and len(llm.calls) == 4
    calls = read_journal(method.storage.llm_calls_path)[-2:]
    assert event['llm_seconds'] == sum(call['seconds'] for call in calls)


def test_repeated_context_request_gets_one_repair_without_another_parent(tmp_path):
    request = json.dumps({'mode': 'context', 'trials': True})
    method = make_method(tmp_path, ScriptedLLM(response(1), request, request, response(2)), budget=2)
    method.run()
    events = read_journal(method.storage.events_path)
    assert [e['status'] for e in events] == ['ok', 'invalid_output', 'ok']
    assert events[1]['reason'].startswith('context_error')
    assert events[-1]['repair_of'] == 2 and not events[-1]['parent_selected']
    assert method.tree.parent_selections == 1 and method.evaluations_used == 2


def test_failed_edit_execution_repairs_reconstructed_code_without_reattributing_parent(tmp_path):
    import re
    def edit(prompt):
        return json.dumps({'mode': 'edit', 'base_hash': re.search(
            r'base_hash: ([a-f0-9]{64})', prompt).group(1),
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
    baseline = method.prompts.build_development(parent, 'Fuse', references=[donor], allow_context=False)
    method.prompts.max_tokens = method.prompts.count(baseline.prompt) + 30
    read = method.prompts.build_development(parent, 'Fuse', references=[donor],
                                           read_reference=donor, allow_context=False)
    assert read.reference_program is None
    assert read.fallback_reason == 'requested_implementation_exceeds_context'
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
        donor_id = int(re.search(r'Node (\d+)', prompt.split('# Optional reference ideas')[1]).group(1))
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
