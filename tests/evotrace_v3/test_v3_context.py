from copy import deepcopy

from evotrace_v3 import Config, Task
from evotrace_v3.context import ContextConfig, EvoTracePromptBuilder, build_evolution_context
from evotrace_v3.models import Edge, Node, TreePopulation


def add(tree, parent, fit, op='Refine'):
    i = tree.latest_node_id + 1
    tree.add(Node(i, f'def solve(x): return x + {i}', f'Compute offset {i}', fit),
             Edge(parent, i, op if parent else None), evaluations=1)
    return i


def test_crossover_reads_the_current_lineage_and_one_reference_card():
    tree = TreePopulation()
    a = add(tree, 0, 9)
    b = add(tree, a, 8, 'Explore')
    c = add(tree, b, 7, 'Explore')
    current = add(tree, c, 6)
    reference = add(tree, b, 5, 'Explore')
    tree.nodes[a].experience = 'EARLY_EVIDENCE'
    tree.nodes[b].experience = 'FORK_EVIDENCE'
    tree.edges[c].experience = 'SHIFT_EVIDENCE'
    tree.edges[reference].experience = 'REF_STAGE'
    tree.nodes[reference].experience = 'REF_ATTEMPTS'
    tree.root_experience = 'INITIAL_EVIDENCE'
    before = deepcopy(tree)
    text, instruction, _, view = build_evolution_context(tree, current, 'Refine')
    assert view['path'] == [a, b, c, current]
    for evidence in ['EARLY_EVIDENCE', 'FORK_EVIDENCE', 'SHIFT_EVIDENCE', 'INITIAL_EVIDENCE']:
        assert evidence in text
    assert 'Explore' not in instruction  # Edges are development relations; operators need no explanation.

    text, _, _, view = build_evolution_context(tree, current, 'Crossover', reference)
    assert view['path'] == [a, b, c, current] and view['attempts'] == [] and view['reference_id'] == reference
    assert text.index('[Evolution Path]') < text.index('[Reference Algorithm]') < text.index('[Current Algorithm]')
    assert 'Common Ancestor' not in text and text.count('```python') == 2 and text.rstrip().endswith('```')
    for evidence in ['INITIAL_EVIDENCE', 'EARLY_EVIDENCE', 'FORK_EVIDENCE', 'SHIFT_EVIDENCE', 'REF_STAGE', 'REF_ATTEMPTS']:
        assert text.count(evidence) == 1  # shared ancestry is shown once, on the current lineage
    assert text.count(f'Compute offset {b}') == 1 and text.count(f'Compute offset {reference}') == 1

    # A reference that is a direct child of the current algorithm appears only as the reference.
    text, _, _, view = build_evolution_context(tree, b, 'Crossover', c)
    assert view['attempts'] == [reference] and text.count(f'Compute offset {c}') == 1

    muted, _, _, _ = build_evolution_context(tree, current, 'Crossover', reference,
                                           ContextConfig(include_experiences=False))
    assert 'FORK_EVIDENCE' not in muted and 'REF_STAGE' not in muted
    assert f'return x + {current}' in muted and f'return x + {reference}' in muted
    assert tree == before


def test_explore_shows_cards_without_code_and_each_experience_once():
    tree = TreePopulation()
    a = add(tree, 0, 9)
    parent = add(tree, a, 8)
    other = add(tree, parent, 7, 'Explore')
    tree.root_experience = 'ROOT_NOTE'
    tree.nodes[a].experience = 'ANCESTOR_NOTE'
    tree.nodes[parent].experience = 'PARENT_NOTE'
    tree.edges[other].experience = 'OTHER_EDGE_NOTE'
    tree.nodes[other].experience = 'OTHER_NOTE'
    builder = EvoTracePromptBuilder(Task('Minimize', 'def solve(x): pass', lambda s, f: f(1), 2), Config())
    prompt, view = builder.evolution_from_tree(tree, parent, 'Explore', explore_ids=(other,))
    assert view['explore_ids'] == [other] and view['shown'] == [other, parent]  # sorted by fitness
    for note in ['ROOT_NOTE', 'ANCESTOR_NOTE', 'PARENT_NOTE', 'OTHER_EDGE_NOTE', 'OTHER_NOTE']:
        assert prompt.count(note) == 1  # each experience appears once, at its first relevant algorithm
    assert f'Compute offset {other}' in prompt and f'Compute offset {parent}' in prompt
    assert f'return x + {parent}' not in prompt and f'return x + {other}' not in prompt  # no code in Explore
    bare, _, _, view = build_evolution_context(tree, parent, 'Explore')
    assert view['shown'] == [parent] and 'ANCESTOR_NOTE' in bare


def test_refine_gives_designs_and_experience_and_code_only_for_the_current_algorithm():
    tree = TreePopulation()
    a = add(tree, 0, 9)
    b = add(tree, a, 8)
    parent = add(tree, b, 7)
    tried = add(tree, parent, 6)
    failed = tree.latest_node_id + 1
    tree.add(Node(failed, 'def solve(x): return x / 0', 'Divide by zero', None, failure='ZeroDivisionError'),
             Edge(parent, failed, 'Refine'), evaluations=1)
    tree.edges[b].contracted, tree.edges[b].operator = True, None
    tree.edges[b].experience = 'STAGE_NOTE'
    tree.nodes[b].experience = 'B_NOTE'
    tree.nodes[parent].experience = 'PARENT_NOTE'
    builder = EvoTracePromptBuilder(Task('Minimize', 'def solve(x): pass', lambda s, f: f(1), 2), Config())
    prompt, view = builder.evolution_from_tree(tree, parent, 'Refine')
    assert view == {'operator': 'Refine', 'parent_id': parent, 'path': [a, b, parent], 'attempts': [tried, failed]}
    text = prompt[prompt.index('[Context]'):prompt.index('[Instruction]')]
    assert text.index('[Evolution Path]') < text.index('[Attempts from Current') < text.index('[Current Algorithm]')
    assert text.rstrip().endswith('```') and text.count('```python') == 1 and '```diff' not in text
    assert f'return x + {a}' not in text and f'return x + {tried}' not in text and 'x / 0' not in text
    assert 'STAGE_NOTE' in text and 'B_NOTE' in text and text.count('PARENT_NOTE') == 1
    assert 'ZeroDivisionError' in text
    for design in [f'Compute offset {a}', f'Compute offset {b}', f'Compute offset {tried}', 'Divide by zero']:
        assert design in text
