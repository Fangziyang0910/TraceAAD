import random

import pytest

from traceaad_base import Config, Evaluation, Program
from traceaad_base.models import Attempt
from traceaad_base.prompts import PromptBuilder
from traceaad_base.search import TraceAAD
from traceaad_base.selection import choose, qualities, reference_pool, sources, weights


def program(pid, score, parent=0, *, status="success"):
    evaluation = (Evaluation("success", score) if status == "success"
                  else Evaluation(status, failure=f"{status}: diagnostic {pid}"))
    return Program(pid, f"def solve(x):\n    return 'code_{pid}'\n", f"design_{pid}",
                   "Init" if parent == 0 else "Refine", parent, None, evaluation,
                   analysis=f"Computational change for candidate [{pid}]")


def attempt(aid, parent, child, calls, *, reference=None):
    return Attempt(aid, parent, "Crossover" if reference else "Refine", reference,
                   child, [{} for _ in range(calls)])


def test_weight_falls_with_rank_and_calls_and_children_earn_no_credit():
    archive = {p.id: p for p in [program(1, 1), program(2, 2), program(3, 9, 1),
                                program(4, 12, 2), program(5, None, 1, status="error")]}
    attempts = [attempt(1, 1, 3, 2, reference=2), attempt(2, 1, 5, 1), attempt(3, 2, 4, 1)]
    actual = weights(archive, attempts)
    assert actual[1] == pytest.approx(3 / 6)
    assert actual[2] == pytest.approx(2 ** -1.5 * 3 / 4)
    assert actual[3] == pytest.approx(3 ** -1.5)
    archive[6] = program(6, 0, 2)
    attempts.append(attempt(4, 2, 6, 1))
    assert weights(archive, attempts)[2] == pytest.approx(3 ** -1.5 * 3 / 5)


def test_identical_source_is_one_parent_candidate_kept_at_its_earliest_program():
    archive = {p.id: p for p in [program(1, 1), program(2, 1, 1), program(3, 5)]}
    archive[2].code = archive[1].code + "\n# same\n"
    assert set(weights(archive, [])) == {1, 3}
    assert [p.id for p in sources(archive)] == [1, 3]
    assert reference_pool(archive, 2) == []


def test_average_tied_ranks_and_median_reference_candidates():
    archive = {p.id: p for p in [program(1, -4), program(2, -2), program(3, -2), program(4, 5)]}
    archive[0] = Program(0, "seed", None, None, None, None, None)
    assert qualities(archive) == {1: 1, 2: 2.5 ** -1.5, 3: 2.5 ** -1.5, 4: 4 ** -1.5}
    assert [p.id for p in reference_pool(archive, 2)] == [1, 3]
    assert [p.id for p in reference_pool(archive, 4)] == [1, 2, 3]


def test_operator_weights_and_reference_fallback(task, tmp_path):
    rng = random.Random(4)
    archive = {1: program(1, 2)}
    parent, op = choose(archive, [], Config(), rng)
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


def test_history_windows_and_evolution_code_boundary(task):
    task.example_code = "def solve(x):\n    return 'seed_body'\n"
    archive = {p.id: p for p in [program(i, 20 - i, i - 1) for i in range(1, 13)]}
    archive.update({p.id: p for p in [program(i, score, 12) for i, score in
                                    [(13, 5), (14, 3), (15, 3), (16, 9), (17, 10), (18, 11)]]})
    archive.update({i: program(i, None, 12, status="error") for i in range(19, 23)})
    archive[30] = program(30, 1)
    builder = PromptBuilder(task, Config())
    ancestors, children, failures = builder.histories(archive, archive[12])
    assert [p.id for p in ancestors] == list(range(4, 12))
    assert [p.id for p in children] == [15, 17, 18]
    assert [p.id for p in failures] == [21, 22]
    prompt = builder.evolution(archive, archive[12], "Crossover", archive[30])
    assert archive[12].code in prompt
    assert all(p.code not in prompt for p in archive.values() if p.id != 12)
    assert all(p.design in prompt for p in [*ancestors, *children, *failures, archive[30]])
    selected = [*ancestors, *children, *failures, archive[12]]
    assert all(p.analysis in prompt for p in selected)
    assert all(p.analysis not in prompt for p in archive.values() if p not in selected)
    for p in [*ancestors, *children, *failures, archive[12]]:
        assert f"Algorithm {p.id}\nParent: Algorithm {p.parent_id}" in prompt
    labels = ["[Ancestor History]", "[Evaluated Offspring]", "[Failed Offspring]",
              "[Reference Design]", "[Current Algorithm]"]
    assert [prompt.index(label) for label in labels] == sorted(prompt.index(label) for label in labels)


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
    assert prompt.index("Algorithm 2") < prompt.index("Algorithm 1")
    evolution = builder.evolution(archive, archive[2], "Explore")
    assert "seed_marker" not in evolution
    assert "[Ancestor History]" not in evolution and "[Evaluated Offspring]" not in evolution


def test_crossed_offspring_analysis_keeps_both_source_ids(task):
    parent = program(1, 5)
    child = program(2, 4, 1)
    child.operator, child.reference_id = "Crossover", 9
    prompt = PromptBuilder(task, Config()).evolution({1: parent, 2: child}, parent, "Refine")
    assert "Algorithm 2\nParent: Algorithm 1\nOperator: Crossover\nReference: Algorithm 9" in prompt
    assert child.analysis in prompt and child.code not in prompt


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
    for text in (prompt, repair):
        assert "IndexProblem: index N is out of bounds for axis N with size N" in text
        assert 'File "<candidate>", line 4' in text
        assert "/srv/runs" not in text and "core.evaluate" not in text and "Traceback (" not in text
    assert "return values[50]" in repair and "return values[50]" not in prompt


def test_timeout_text_states_a_limit_only_in_repair(task):
    task.metadata["seconds_per_instance"] = 10
    parent = program(1, 3)
    failed = program(2, None, 1, status="timeout")
    failed.evaluation = Evaluation("timeout", failure="Evaluation exceeded the total time limit of 160 seconds.")
    builder = PromptBuilder(task, Config())
    prompt = builder.evolution({1: parent, 2: failed}, parent, "Refine")
    assert "did not finish within the allowed time" in prompt and "seconds" not in prompt
    repair = builder.repair_program(failed)
    assert "10 seconds per problem instance" in repair and "160" not in repair
