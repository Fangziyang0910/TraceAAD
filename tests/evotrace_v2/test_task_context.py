import pytest

from evotrace_v2 import Config, Task
from evotrace_v2.context import ContextConfig, EvoTracePromptBuilder, build_evolution_context
from evotrace_v2.models import TreePopulation, Node, Edge
from evotrace_v2.allocation import eligible_reference_pool


def chain(count):
    tree = TreePopulation()
    for i in range(1, count + 1):
        tree.add(Node(i, f'def solve(x):\n    return x + {i}\n', f'Design {i}', float(count-i)),
                 Edge(i-1, i, None if i==1 else 'Refine'), evaluations=1)
    return tree


def test_all_retained_path_steps_are_read_and_entry_is_not_called_initial():
    tree=chain(16)
    context,_,_,view=build_evolution_context(tree,16,'Refine')
    assert len(view['shown_steps'])==15
    assert 'earlier step(s) omitted' not in context
    assert 'Retained Path Entry 1' in context


def test_all_children_and_failures_are_read_before_window_fitting():
    tree=chain(1)
    for i in range(2,12):
        tree.add(Node(i,f'def solve(x):\n    return x * {i}\n',f'Idea {i}',float(i) if i<8 else None,failure='ValueError'),Edge(1,i,'Refine'),evaluations=1)
    _,_,_,view=build_evolution_context(tree,1,'Refine')
    assert len(view['shown_valid_children'])==6
    assert len(view['shown_failed_children'])==4


def test_explore_reads_internal_condensed_experience_and_all_distinct_entries():
    tree=chain(3)
    tree.edges[3].contracted=True
    tree.edges[3].variation_experience='INTERNAL_DEVELOPMENT'
    text,_,_,_=build_evolution_context(tree,3,'Explore')
    assert 'INTERNAL_DEVELOPMENT' in text


def test_actual_input_budget_is_the_only_default_reduction_trigger():
    tree=chain(24)
    task=Task('Minimize','def solve(x):\n    pass\n',lambda s,f:f(1),2)
    builder=EvoTracePromptBuilder(task,Config())
    full,view=builder.evolution_from_tree(tree,24,'Refine',count_tokens=len,
                                        context_config=ContextConfig(max_input_tokens=100000))
    assert view['input_window']['reductions']==[]
    prompt,view=builder.evolution_from_tree(tree,24,'Refine',count_tokens=len,
                                         context_config=ContextConfig(max_input_tokens=len(full)//2))
    assert len(prompt)<=len(full)//2
    assert view['input_window']['reductions']
    assert '[Current Algorithm]' in prompt and 'Baseline for the following changes' in prompt
    assert 'return x + 24' in prompt


def test_one_large_direction_can_be_fitted_without_cutting_experience_text():
    tree=chain(16)
    for i,n in tree.nodes.items():n.selection_experience=f'NOTE_{i} '+('evidence '*60)+'END_NOTE'
    task=Task('Minimize','def solve(x):\n    pass\n',lambda s,f:f(1),2)
    b=EvoTracePromptBuilder(task,Config())
    full,_=b.evolution_from_tree(tree,16,'Explore',count_tokens=len,context_config=ContextConfig(max_input_tokens=100000))
    text,view=b.evolution_from_tree(tree,16,'Explore',count_tokens=len,context_config=ContextConfig(max_input_tokens=len(full)//2))
    assert len(text)<=len(full)//2 and text.count('NOTE_')==text.count('END_NOTE')
    assert any(x['field']=='max_experiences' for x in view['input_window']['reductions'])


def test_small_reference_pool_uses_same_quality_rule_and_excludes_self_first():
    tree=chain(3)
    assert eligible_reference_pool(tree,3)==[2]
    tree=chain(2)
    assert eligible_reference_pool(tree,2)==[1]
