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


def test_descendant_reads_branch_experience_at_each_shown_path_stage_once():
    tree=chain(4)
    tree.nodes[1].selection_experience='ENTRY_TRIED_A_ROLLOUT'
    tree.nodes[2].selection_experience='INTERMEDIATE_TRIED_AN_INSERTION'
    tree.nodes[4].selection_experience='CURRENT_TRIED_A_PARAMETER_CHANGE'
    text,_,_,_=build_evolution_context(tree,4,'Refine')
    for note in ['ENTRY_TRIED_A_ROLLOUT','INTERMEDIATE_TRIED_AN_INSERTION',
                 'CURRENT_TRIED_A_PARAMETER_CHANGE']:
        assert text.count(note)==1
    without,_,_,_=build_evolution_context(tree,4,'Refine',config=ContextConfig(include_experiences=False))
    assert 'ENTRY_TRIED_A_ROLLOUT' not in without
    assert 'INTERMEDIATE_TRIED_AN_INSERTION' not in without


def test_crossover_reads_experience_along_both_developments():
    tree=chain(3)
    tree.add(Node(4,'def solve(x):\n    return x * 2\n','Other branch',2.),Edge(1,4,'Explore'),evaluations=1)
    tree.nodes[1].selection_experience='SHARED_EXPERIMENTS'
    tree.nodes[2].selection_experience='CURRENT_LINE_EXPERIMENTS'
    tree.nodes[3].selection_experience='CURRENT_ENDPOINT_EXPERIMENTS'
    tree.nodes[4].selection_experience='REFERENCE_EXPERIMENTS'
    text,_,_,_=build_evolution_context(tree,3,'Crossover',4)
    for note in ['SHARED_EXPERIMENTS','CURRENT_LINE_EXPERIMENTS',
                 'CURRENT_ENDPOINT_EXPERIMENTS','REFERENCE_EXPERIMENTS']:
        assert text.count(note)==1


def test_origin_reflection_does_not_turn_an_independent_champion_into_a_parent():
    from evotrace_v2.reflection import build_selection_prompt
    from evotrace_v2.topology import BranchConsolidation
    tree=chain(1)
    tree.nodes[1].fitness=-9.5
    tree.add(Node(2,'def solve(x):\n    return x * 2\n','Independent idea',-14.4),
             Edge(0,2,None),evaluations=1)
    text=build_selection_prompt(tree,BranchConsolidation(0,(1,)),task_description='Minimize')
    assert 'no shared parent implementation' in text
    assert 'return x * 2' not in text
    assert '-14.4' not in text
    assert 'Removed Trial 1 [Init] from initial origin' in text
    assert 'evaluated (no algorithm parent)' in text


def test_path_reflection_states_net_tie_when_an_intermediate_regresses():
    from evotrace_v2.reflection import build_variation_prompt
    from evotrace_v2.topology import PathContraction
    tree=chain(3)
    tree.nodes[1].fitness=7.
    tree.nodes[2].fitness=8.
    tree.nodes[3].fitness=7.
    text=build_variation_prompt(tree,PathContraction(1,3,(2,),'Refine',None),task_description='Minimize')
    assert 'Overall measured outcome: tied' in text
    assert 'Step 1 [Refine] (Removed Intermediate): regressed' in text
    assert 'Step 2 [Refine] (Retained Endpoint): improved' in text


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
