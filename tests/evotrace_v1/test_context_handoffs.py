from evotrace_v1.context import ContextConfig, EvoTracePromptBuilder, build_evolution_context
from evotrace_v1.models import Edge, Node, Program, Evaluation, TreePopulation
from evotrace_v1.task import Task
from evotrace_v1.config import Config


def test_windowed_changes_include_their_actual_baseline():
    tree = TreePopulation()
    for i in range(1, 6):
        tree.add(Node(i, f'def solve(x):\n    return x + {i}\n', f'Offset {i}', float(i)),
                 Edge(i-1, i, None if i == 1 else 'Refine'), evaluations=1)
    context, *_ = build_evolution_context(tree, 5, 'Refine', config=ContextConfig(max_path_steps=2))
    assert 'earlier step(s) omitted' in context
    assert 'Baseline for the following changes (fitness = 4)' in context
    assert 'return x + 4' in context
    assert 'return x + 5' in context


def test_initialization_shares_ideas_and_keeps_failed_implementation():
    task = Task('Minimize the result.', 'def solve(x):\n    pass\n', lambda *args: 1, 5)
    archive = {
        1: Program(1, 'def solve(x):\n    return x + 12345', 'Offset idea', 'Init', 0, None, Evaluation('success', 2)),
        2: Program(2, 'def solve(x):\n    return x[999]', 'Index lookup', 'Init', 0, None, Evaluation('error', failure='IndexError: bad index')),
    }
    prompt = EvoTracePromptBuilder(task, Config()).initialization(archive)
    assert 'Offset idea' in prompt
    assert '12345' not in prompt
    assert 'x[999]' in prompt
    assert 'IndexError' in prompt


def test_timeout_feedback_uses_shared_total_deadline():
    task = Task('Minimize the result.', 'def solve(x):\n    pass\n', lambda *args: 1, 160,
                metadata={'seconds_per_instance': 10})
    program = Program(1, 'def solve(x):\n    return x', 'Identity', 'Refine', 0, None,
                      Evaluation('timeout', failure='Evaluation exceeded the total time limit of 160 seconds.'))
    prompt = EvoTracePromptBuilder(task, Config()).repair_program(program)
    assert 'total time limit of 160 seconds' in prompt
    assert '10 seconds per problem instance' not in prompt
