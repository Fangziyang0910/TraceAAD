import numpy as np
import pytest

from baselines.funsearch.config import ProgramsDatabaseConfig
from baselines.funsearch.funsearch import FunSearch, chat_instruction
from baselines.funsearch.programs_database import ProgramsDatabase
from core import Evaluation, LLM, TextFunctionProgramConverter

TEMPLATE = '''
import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin."""
    return -bins
'''


class FakeEvaluation(Evaluation):
    """Score = value returned by the program on a fixed input; lower is better."""

    def __init__(self):
        super().__init__(template_program=TEMPLATE, task_description='Online bin packing.',
                         safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return float(np.sum(callable_func(1.0, np.array([1.0, 2.0]))))


class ScriptedLLM(LLM):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.prompts = []

    def draw_sample(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else 'no code here'


def make_method(responses, budget=8, **config):
    np.random.seed(0)
    method = FunSearch(ScriptedLLM(responses), FakeEvaluation(), profiler=None, max_sample_nums=budget,
                       database_config=ProgramsDatabaseConfig(**config))
    return method


def test_prompt_lists_versions_from_worse_to_better_and_ends_with_empty_header():
    method = make_method([])
    template = method._template_program.functions[0]
    worse = TextFunctionProgramConverter.text_to_function(str(template).replace('return -bins', 'return bins'))
    method._database.register_program(template, 0, 5.0)
    method._database.register_program(worse, 0, 1.0)
    island = method._database.islands[0]
    code = island._generate_prompt([worse, template])

    assert code.startswith('"""Online bin packing."""')
    assert 'import numpy as np' in code
    assert code.index('def priority_v0') < code.index('def priority_v1') < code.index('def priority_v2')
    assert 'return bins' in code.split('def priority_v1')[0]
    assert 'Improved version of `priority_v0`.' in code
    assert code.rstrip().endswith('"""Improved version of `priority_v1`."""')
    assert 'def priority(' not in code


def test_island_prompt_version_matches_the_header():
    method = make_method([])
    seed = method._template_program.functions[0]
    method._database.register_program(seed, None, 0.0)
    prompt = method._database.get_prompt()
    assert prompt.version_generated == 1
    assert prompt.code.rstrip().endswith('"""Improved version of `priority_v0`."""')
    assert 'def priority_v1(' in prompt.code


def test_chat_response_is_parsed_into_the_evolved_function():
    method = make_method([])
    response = ('Here is my improvement.\n```python\nimport numpy as np\n\n'
                'def priority_v2(item: float, bins: np.ndarray) -> np.ndarray:\n'
                '    """Better."""\n    scores = bins - item\n    return -scores\n```\nIt prefers tight bins.')
    program = method.sample_to_program(response, 2)
    function = program.functions[0]

    assert function.name == 'priority'
    assert 'scores = bins - item' in function.body
    assert 'Better' not in str(program)
    assert function.docstring == method._function_to_evolve.docstring


def test_recursive_calls_are_renamed_and_ancestor_calls_rejected():
    method = make_method([])
    recursive = ('def priority_v1(item, bins):\n    if item > 1:\n        return priority_v1(item - 1, bins)\n'
                 '    return -bins\n')
    program = method.sample_to_program(recursive, 1)
    assert 'priority(item - 1, bins)' in str(program)
    assert not method.calls_ancestor(program)

    ancestor = 'def priority_v1(item, bins):\n    return priority_v0(item, bins) * 2\n'
    assert method.calls_ancestor(method.sample_to_program(ancestor, 1))


def test_completion_style_body_is_accepted():
    method = make_method([])
    program = method.sample_to_program('    return bins * 0.5\n', 1)
    assert 'return bins * 0.5' in program.functions[0].body


def test_cluster_sampling_temperature_and_cluster_keys():
    db = ProgramsDatabase(ProgramsDatabaseConfig(num_islands=2), make_method([])._template_program, 'priority')
    seed = make_method([])._function_to_evolve
    db.register_program(seed, 0, -3.0)
    db.register_program(seed, 0, -3.0)
    db.register_program(seed, 0, -1.0)
    island = db.islands[0]
    assert set(island.clusters) == {(-3.0,), (-1.0,)}
    assert len(island.clusters[(-3.0,)].programs) == 2
    assert db.best_score_per_island == [-1.0, -float('inf')]


def test_reset_replaces_weaker_half_with_founders_from_kept_islands():
    now = [0.0]
    template = make_method([])._template_program
    seed = make_method([])._function_to_evolve
    db = ProgramsDatabase(ProgramsDatabaseConfig(num_islands=4, reset_period=10), template, 'priority',
                          clock=lambda: now[0])
    for island_id, score in enumerate([-4.0, -1.0, -3.0, -2.0]):
        db.register_program(seed, island_id, score)
    for island_id, score in enumerate([-5.0, -6.0]):
        db.register_program(seed, island_id, score)
    assert db.num_resets == 0
    now[0] = 11
    db.register_program(seed, 1, -9.0)

    assert db.num_resets == 1
    best = db.best_score_per_island
    assert best[1] == -1.0 and best[3] == -2.0
    assert best[0] in (-1.0, -2.0) and best[2] in (-1.0, -2.0)
    assert db.islands[0].num_programs == 1 and db.islands[2].num_programs == 1


def test_run_counts_every_sample_and_registers_valid_programs_on_their_island():
    good = 'def priority_v1(item, bins):\n    return bins * 0 - 10\n'
    invalid = 'def priority_v1(item, bins):\n    return undefined_name\n'
    ancestor = 'def priority_v1(item, bins):\n    return priority_v0(item, bins)\n'
    responses = [good, 'I cannot help.', invalid, ancestor]
    method = make_method(responses, budget=4)
    method.run()

    llm = method._llm
    assert method._tot_sample_nums == 4
    assert len(llm.prompts) == 4
    assert len(set(llm.prompts)) == 1  # four samples per prompt
    assert llm.prompts[0].endswith(chat_instruction('priority', 1))
    counts = [island.num_programs for island in method._database.islands]
    assert sum(counts) == 10 + 1  # the seed on every island, then only the good program
    assert max(method._database.best_score_per_island) == pytest.approx(20.0)


def test_run_stops_when_template_is_invalid():
    class Broken(FakeEvaluation):
        def evaluate_program(self, program_str, callable_func, **kwargs):
            return None

    method = FunSearch(ScriptedLLM([]), Broken(), max_sample_nums=4)
    with pytest.raises(RuntimeError):
        method.run()
