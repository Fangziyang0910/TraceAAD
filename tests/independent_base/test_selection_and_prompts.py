import random
import math

import pytest

from traceaad_base import Config, Evaluation, Program
from traceaad_base.models import Attempt
from traceaad_base.prompts import PromptBuilder
from traceaad_base.search import TraceAAD
from traceaad_base.selection import boltzmann_weights, choose, reference_pool, sources, weights


def program(pid, score, parent=0, *, status="success"):
    evaluation = (Evaluation("success", score) if status == "success"
                  else Evaluation(status, failure=f"{status}: diagnostic {pid}"))
    return Program(pid, f"def solve(x):\n    return 'code_{pid}'\n", f"design_{pid}",
                   "Init" if parent == 0 else "Refine", parent, None, evaluation,
                   analysis=f"Computational change for candidate [{pid}]")


def attempt(aid, parent, child, calls, *, reference=None):
    return Attempt(aid, parent, "Crossover" if reference else "Refine", reference,
                   child, [{} for _ in range(calls)])


def test_parent_selection_ignores_call_history_and_children_earn_no_credit(task):
    archive = {p.id: p for p in [program(1, 1), program(2, 2), program(3, 9, 1),
                                program(4, 12, 2), program(5, None, 1, status="error")]}
    attempts = [attempt(1, 1, 3, 2, reference=2), attempt(2, 1, 5, 1), attempt(3, 2, 4, 1)]
    search = TraceAAD(task, None)
    search.archive = archive
    search.rng = random.Random(21)
    before = [search.select(i)[:3] for i in range(30)]
    search.attempts = attempts
    search.rng = random.Random(21)
    assert [search.select(i)[:3] for i in range(30)] == before
    actual = weights(archive, 1.0)
    assert actual[1] / actual[2] == pytest.approx(math.e)
    archive[6] = program(6, 0, 2)
    attempts.append(attempt(4, 2, 6, 1))
    changed = weights(archive, 1.0)
    assert changed[1] / changed[2] == pytest.approx(actual[1] / actual[2])


def test_near_scores_and_temperature_use_fitness_gaps_not_rank():
    actual = boltzmann_weights({1: 5.69, 2: 5.70}, 0.1)
    assert actual[1] / actual[2] == pytest.approx(math.exp(0.01 / 0.1))
    assert boltzmann_weights({1: 5.69, 2: 5.70}, 0.2)[2] > actual[2]
    # A better newcomer changes both weights by the same factor.
    newcomer = boltzmann_weights({1: 5.69, 2: 5.70, 3: 5.60}, 0.1)
    assert newcomer[1] / newcomer[2] == pytest.approx(actual[1] / actual[2])


def test_boltzmann_handles_negative_large_and_extreme_scores_without_overflow():
    original = boltzmann_weights({1: -5.0, 2: -4.0}, 2.0)
    assert boltzmann_weights({1: 999995.0, 2: 999996.0}, 2.0) == pytest.approx(original)
    assert boltzmann_weights({1: -50.0, 2: -40.0}, 20.0) == pytest.approx(original)
    assert boltzmann_weights({1: -1e308, 2: 1e308}, 1e308) == pytest.approx(
        {1: 1.0, 2: math.exp(-2.0)}
    )
    assert boltzmann_weights({1: -1e308, 2: 1e308}, 1.0) == {1: 1.0, 2: 0.0}
    assert boltzmann_weights({}, 1.0) == {}


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan])
def test_selection_temperature_must_be_finite_and_positive(temperature):
    with pytest.raises(ValueError, match="selection_temperature"):
        Config(selection_temperature=temperature)


def test_parent_candidates_exclude_nonfinite_fitness():
    archive = {p.id: p for p in [program(1, 1.0), program(2, math.nan),
                                program(3, math.inf), program(4, None)]}
    assert weights(archive, 1.0) == {1: 1.0}


def test_identical_source_is_one_parent_candidate_kept_at_its_earliest_program():
    archive = {p.id: p for p in [program(1, 1), program(2, 1, 1), program(3, 5)]}
    archive[2].code = archive[1].code + "\n# same\n"
    assert set(weights(archive, 1.0)) == {1, 3}
    assert [p.id for p in sources(archive)] == [1, 3]
    assert reference_pool(archive, 2) == []


def test_tied_scores_get_equal_weights_and_median_references_stay_unchanged():
    archive = {p.id: p for p in [program(1, -4), program(2, -2), program(3, -2), program(4, 5)]}
    archive[0] = Program(0, "seed", None, None, None, None, None)
    assert weights(archive, 1.0) == pytest.approx(
        {1: 1, 2: math.exp(-2), 3: math.exp(-2), 4: math.exp(-9)}
    )
    assert [p.id for p in reference_pool(archive, 2)] == [1, 3]
    assert [p.id for p in reference_pool(archive, 4)] == [1, 2, 3]


def test_operator_weights_and_reference_fallback(task, tmp_path):
    rng = random.Random(4)
    archive = {1: program(1, 2)}
    parent, op = choose(archive, Config(), rng)
    assert parent.id == 1 and op in {"Refine", "Explore", "Crossover"}
    assert reference_pool(archive, 1) == []

    class Always(random.Random):
        def choices(self, population, weights, **kwargs):
            return ["Crossover"] if population[0] == "Refine" else super().choices(population, weights, **kwargs)

    search = TraceAAD(task, None, config=Config(), run_dir=tmp_path)
    search.archive, search.rng = archive, Always(1)
    assert search.select(1)[:3] == (archive[1], "Explore", None)
    archive[2] = program(2, 3)
    _, op, reference, _ = search.select(2)
    assert op == "Crossover" and reference.id in {1, 2}


@pytest.mark.parametrize("operator", ["Refine", "Explore", "Crossover"])
def test_evolution_ignores_trajectory_and_preserves_static_inputs(task, operator):
    parent, reference = program(12, 8, 11), program(30, 1)
    archive = {p.id: p for p in [program(11, 9), parent, program(13, 7, 12),
                                program(14, None, 12, status="error"), reference]}
    builder = PromptBuilder(task, Config())
    selected_reference = reference if operator == "Crossover" else None
    prompt = builder.evolution(archive, parent, operator, selected_reference)
    assert parent.code in prompt and parent.design in prompt and "Score: 8" in prompt
    assert parent.analysis not in prompt
    assert all(label not in prompt for label in
               ("Parent:", "Operator:", "Reference:", "Analysis:", "[Ancestor History]",
                "[Evaluated Offspring]", "[Failed Offspring]"))
    assert all(p.design not in prompt and p.code not in prompt
               for p in (archive[11], archive[13], archive[14]))
    assert (reference.design in prompt) == (operator == "Crossover")
    assert reference.code not in prompt
    parent.parent_id, parent.operator, parent.reference_id, parent.analysis = 999, "Explore", 888, "hidden change"
    changed_archive = {parent.id: parent, 999: program(999, -100)}
    assert builder.evolution(changed_archive, parent, operator, selected_reference) == prompt


def test_initial_context_contains_seed_and_valid_code_but_interface_hides_body(task):
    task = type(task)(task.task_description, "def solve(x):\n    return 'seed_marker'\n",
                      task.evaluate_program, task.time_limit)
    builder = PromptBuilder(task, Config())
    assert "seed_marker" not in task.function_template
    assert "[Context]" not in builder.initialization({})
    archive = {0: Program(0, task.template_program, None, None, None, None, None),
               1: program(1, 9), 2: program(2, 3), 3: program(3, None, status="error")}
    prompt = builder.initialization(archive)
    assert all(archive[i].code in prompt for i in [0, 1, 2])
    assert archive[3].code not in prompt and archive[3].design in prompt
    assert all(p.analysis in prompt for p in (archive[1], archive[2], archive[3]))
    assert "[Failed Attempts]" in prompt
    assert prompt.index("Algorithm 2") < prompt.index("Algorithm 1")
    evolution = builder.evolution(archive, archive[2], "Explore")
    assert "seed_marker" not in evolution
    assert "[Ancestor History]" not in evolution and "[Evaluated Offspring]" not in evolution


def test_failure_text_hides_evaluator_frames_sizes_and_limits(task):
    failure = ('Traceback (most recent call last):\n'
               '  File "/srv/runs/traceaad_base/runtime.py", line 37, in _worker\n    value = function(*args)\n'
               '  File "<candidate>", line 4, in solve\n    return values[50]\n'
               '  File "/srv/runs/benchmarks/tsp/evaluation.py", line 19, in check\n    raise X\n'
               'core.evaluate.IndexProblem: index 50 is out of bounds for axis 0 with size 50')
    parent = program(1, 3)
    failed = program(2, None, 1, status="error")
    failed.evaluation = Evaluation("error", failure=failure)
    builder = PromptBuilder(task, Config())
    prompt = builder.evolution({1: parent, 2: failed}, parent, "Refine")
    repair = builder.repair_program(failed)
    assert "IndexProblem" not in prompt and failed.design not in prompt
    assert failed.analysis in repair
    assert "Parent: Algorithm 1" in repair and "Operator: Refine" in repair
    assert "IndexProblem: index N is out of bounds for axis N with size N" in repair
    assert 'File "<candidate>", line 4' in repair
    assert "/srv/runs" not in repair and "core.evaluate" not in repair and "Traceback (" not in repair
    assert "return values[50]" in repair and "return values[50]" not in prompt


def test_timeout_text_states_a_limit_only_in_repair(task):
    task.metadata["seconds_per_instance"] = 10
    parent = program(1, 3)
    failed = program(2, None, 1, status="timeout")
    failed.evaluation = Evaluation("timeout", failure="Evaluation exceeded the total time limit of 160 seconds.")
    builder = PromptBuilder(task, Config())
    prompt = builder.evolution({1: parent, 2: failed}, parent, "Refine")
    assert "did not finish within the allowed time" not in prompt and "160" not in prompt
    repair = builder.repair_program(failed)
    assert "10 seconds per problem instance" in repair and "160" not in repair
