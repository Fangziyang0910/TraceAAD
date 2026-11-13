"""Regressions from the first live run: facts must reach the next LLM call."""
import re

from evotrace_v1.models import Edge, Node, TreePopulation
from evotrace_v1.context import EvoTracePromptBuilder, build_evolution_context
from evotrace_v1.reflection import build_selection_prompt, compact_diff
from evotrace_v1.topology import BranchConsolidation
from evotrace_v1 import Config, Task
from evotrace_v1.models import Evaluation, Program


def _removed_subtree():
    tree = TreePopulation()
    for nid, parent, fit, body in [
        (1, 0, 10., 'return x'),
        (2, 1, 12., 'return x * 2'),
        (3, 2, 11., 'return x * 3'),
        (4, 3, None, 'return x[wrong_axis]'),
    ]:
        tree.add(Node(nid, f'def solve(x):\n    {body}\n', f'Computation {nid}', fit,
                      failure='IndexError: index 20 is out of bounds for axis 1 with size 20' if fit is None else None),
                 Edge(parent, nid, None if parent == 0 else 'Refine'), evaluations=1)
    tree.nodes[2].selection_experience = '[Eliminated] ' + 'Earlier scaling; ' * 22 + 'TAIL_FACT: division raised ZeroDivisionError.'
    tree.nodes[3].selection_experience = '[Eliminated] Logarithmic scaling scored 17.2.'
    tree.edges[2].variation_experience = 'An intermediate greedy simulation scored -11.8538.'
    tree.edges[3].variation_experience = 'Squared distances replaced linear distances; score 11.'
    return tree


def test_pruned_subtree_passes_all_prior_node_and_edge_experience_to_reflection():
    tree = _removed_subtree()
    prompt = build_selection_prompt(tree, BranchConsolidation(1, (2, 3, 4)),
                                    task_description='Minimize the objective.')
    for nid in (2, 3):
        assert tree.nodes[nid].selection_experience in prompt
        assert tree.edges[nid].variation_experience in prompt


def test_failure_inside_mixed_subtree_keeps_its_computation_parent_baseline_and_exception():
    tree = _removed_subtree()
    prompt = build_selection_prompt(tree, BranchConsolidation(1, (2, 3, 4)),
                                    task_description='Minimize the objective.')
    assert 'Retained Parent Baseline' in prompt
    assert '    return x' in prompt
    assert 'IndexError' in prompt
    assert 'Removed Trial 3 [Refine]' in prompt
    assert 'wrong_axis' in prompt


def test_diff_keeps_the_core_computation_in_the_middle_of_a_large_change():
    old = 'def solve(x):\n    return x\n'
    lines = [f'    value_{i} = x + {i}' for i in range(40)]
    lines.insert(20, '    critical_inflation = 0.8 * x')
    new = 'def solve(x):\n' + '\n'.join(lines) + '\n    return x + critical_inflation\n'
    diff = compact_diff(old, new)
    assert '+    critical_inflation = 0.8 * x' in diff
    assert 'diff lines omitted' not in diff


def test_refine_failure_card_explains_the_attempted_change():
    tree = _removed_subtree()
    context, *_ = build_evolution_context(tree, 3, 'Refine')
    assert 'IndexError' in context
    assert 'Computation 4' in context
    assert '+    return x[wrong_axis]' in context


def test_initialization_and_repairs_align_schema_order_and_show_only_executable_code():
    task = Task(task_description='Minimize the objective.',
                template_program='def solve(x):\n    pass\n',
                evaluate_program=lambda code, fn: 0., time_limit=1.)
    prompts = EvoTracePromptBuilder(task, Config())
    code = 'def solve(x):\n    # INTERNAL_PLANNING_NOTE\n    return x[0]\n'
    p = Program(1, code, 'Read the first element.', 'Init', 0, None,
                Evaluation('success', 1.), analysis='UNVERIFIED_PLANNING_RATIONALE')
    initial = prompts.initialization({1: p})
    p.evaluation = Evaluation('error', failure='IndexError: index 0 is out of bounds')
    repair = prompts.repair_program(p)
    formatted = prompts.repair_format('{"analysis": "Read first element"}', 'Missing code and design.')
    for prompt in (initial, repair, formatted):
        assert re.findall(r'^- "(analysis|code|design)"', prompt, re.M) == ['analysis', 'code', 'design']
    for prompt in (initial, repair):
        assert 'INTERNAL_PLANNING_NOTE' not in prompt
        assert 'UNVERIFIED_PLANNING_RATIONALE' not in prompt


def test_reflection_reads_computations_without_generated_design_claims():
    tree = _removed_subtree()
    tree.nodes[2].design = 'UNVERIFIED_DESIGN: this formula is always optimal.'
    prompt = build_selection_prompt(tree, BranchConsolidation(1, (2, 3, 4)),
                                    task_description='Minimize the objective.')
    assert 'UNVERIFIED_DESIGN' not in prompt
    assert '    return x * 2' in prompt
