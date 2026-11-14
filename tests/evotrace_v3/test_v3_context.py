from copy import deepcopy

from evotrace_v3 import Config, Task
from evotrace_v3.context import ContextConfig, EvoTracePromptBuilder, build_evolution_context
from evotrace_v3.models import Edge, Node, TreePopulation


def add(tree, parent, fit, op='Refine'):
    i = tree.latest_node_id + 1
    tree.add(Node(i, f'def solve(x): return x + {i}', f'Compute offset {i}', fit),
             Edge(parent, i, op if parent else None), evaluations=1)
    return i


def test_paths_and_lca_follow_actual_ancestry_across_explore():
    tree = TreePopulation()
    a = add(tree, 0, 9)
    b = add(tree, a, 8, 'Explore')
    c = add(tree, b, 7, 'Explore')
    current = add(tree, c, 6)
    reference = add(tree, b, 5, 'Explore')
    tree.nodes[a].selection_experience = 'EARLY_EVIDENCE'
    tree.nodes[b].selection_experience = 'FORK_EVIDENCE'
    tree.edges[c].variation_experience = 'SHIFT_EVIDENCE'
    tree.root_selection_experience = 'INITIAL_EVIDENCE'
    before = deepcopy(tree)
    text, instruction, _, view = build_evolution_context(tree, current, 'Refine')
    assert view['path'] == [a, b, c, current]
    for evidence in ['EARLY_EVIDENCE', 'FORK_EVIDENCE', 'SHIFT_EVIDENCE', 'INITIAL_EVIDENCE']:
        assert evidence in text
    assert 'initial alternative computation' in instruction
    text, _, _, view = build_evolution_context(tree, current, 'Crossover', reference)
    assert view['lca_id'] == b
    assert view['shown_current_branch'] == [c, current]
    assert view['shown_reference_branch'] == [reference]
    assert 'FORK_EVIDENCE' in text and 'SHIFT_EVIDENCE' in text
    assert 'EARLY_EVIDENCE' not in text  # The shared ancestor is the relevant crossover baseline.
    muted, _, _, _ = build_evolution_context(tree, current, 'Crossover', reference,
                                           ContextConfig(include_experiences=False))
    assert 'FORK_EVIDENCE' not in muted and 'SHIFT_EVIDENCE' not in muted
    assert 'return x + 4' in muted and 'return x + 5' in muted
    assert tree == before


def test_explore_reads_individuals_and_fits_complete_entries_without_grouping():
    tree = TreePopulation()
    parent = 0
    for i in range(16):
        parent = add(tree, parent, 20-i, 'Explore' if i % 3 else 'Refine')
        tree.nodes[parent].selection_experience = f'NOTE_{parent} ' + 'evidence '*30 + 'END_NOTE'
    task = Task('Minimize', 'def solve(x): pass', lambda src, fn: fn(1), 2)
    builder = EvoTracePromptBuilder(task, Config())
    full, view = builder.evolution_from_tree(tree, parent, 'Explore', count_tokens=len,
                                            context_config=ContextConfig(max_input_tokens=100000))
    assert {item['node_id'] for item in view['population']} == set(tree.nodes)
    assert not view['input_window']['reductions']
    prompt, view = builder.evolution_from_tree(tree, parent, 'Explore', count_tokens=len,
                                              context_config=ContextConfig(max_input_tokens=len(full)//2))
    assert len(prompt) <= len(full)//2 and view['input_window']['reductions']
    assert prompt.count('NOTE_') == prompt.count('END_NOTE')
    assert 'return x + 16' in prompt


def test_path_context_fits_input_window_and_keeps_current_implementation():
    tree = TreePopulation()
    for i in range(24):
        add(tree, i, 24-i)
    builder = EvoTracePromptBuilder(Task('Minimize', 'def solve(x): pass', lambda s, f: f(1), 2), Config())
    full, _ = builder.evolution_from_tree(tree, 24, 'Refine', count_tokens=len,
                                         context_config=ContextConfig(max_input_tokens=100000))
    prompt, view = builder.evolution_from_tree(tree, 24, 'Refine', count_tokens=len,
                                              context_config=ContextConfig(max_input_tokens=len(full)//2))
    assert len(prompt) <= len(full)//2
    assert view['input_window']['reductions']
    assert 'return x + 24' in prompt and 'Baseline for the following changes' in prompt
