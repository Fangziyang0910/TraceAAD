import os
from pathlib import Path
import random
import time

import numpy as np
import pytest

from traceaad_base import Candidate, Config, Evaluator, FormatError, Task, TraceAAD


def test_syntax_and_signature_errors_reach_evaluation_not_format_repair(task, reply):
    candidate = Candidate.parse(reply(body="return ("))
    outcome = Evaluator(task).evaluate(candidate.code)
    assert outcome.status == "error" and "SyntaxError" in outcome.failure
    wrong = Evaluator(task).evaluate("def solve(other):\n    return 1")
    assert wrong.status == "error" and "parameters and defaults" in wrong.failure
    failure = Evaluator(task).evaluate("def solve(x):\n    return 1 / 0")
    assert 'File "<candidate>", line 2' in failure.failure and "ZeroDivisionError" in failure.failure
    with pytest.raises(FormatError):
        Candidate.parse("```json\n" + reply(1) + "\n```")


@pytest.mark.parametrize("expression", ["None", "float('nan')", "float('inf')", "True", "[1.0]"])
def test_missing_nonfinite_and_nonscalar_scores_are_program_errors(task, expression):
    result = Evaluator(task).evaluate(f"def solve(x):\n    return {expression}")
    assert result.status == "error" and result.fitness is None
    assert "finite" in result.failure


def test_same_random_seed_and_fresh_task_data_for_every_evaluation():
    data = [1]
    def score(source, function):
        data.append(2)
        return function(len(data))

    task = Task("Minimize score.", "def solve(x):\n    pass", score, 3, evaluation_seed=1729)
    code = "import random\nimport numpy as np\ndef solve(x):\n    return x + random.random() + np.random.random()"
    py_state, np_state = random.getstate(), np.random.get_state()
    first, second = Evaluator(task).evaluate(code), Evaluator(task).evaluate(code)
    assert first.status == "success" and first.fitness == second.fitness
    assert data == [1] and random.getstate() == py_state
    current = np.random.get_state()
    assert current[0] == np_state[0] and current[2:] == np_state[2:]
    np.testing.assert_array_equal(current[1], np_state[1])


def test_task_memory_limit_applies_before_generated_module_executes(task):
    task.memory_limit_bytes = 16 << 20
    source = "allocated = bytearray(64 * 1024 * 1024)\ndef solve(x): return 1"
    result = Evaluator(task).evaluate(source)
    assert result.status == "error" and "MemoryError" in result.failure


def test_timeout_spans_all_instances_and_repair_updates_same_individual(reply, scripted_model):
    task = Task("Minimize the mean value.", "def solve(x):\n    pass",
                lambda source, function: sum(function(i) for i in range(3)) / 3, .3)
    slow = reply(body="__import__('time').sleep(.14)\n    return 1")
    model = scripted_model([slow, reply(2)])
    search = TraceAAD(task, model, config=Config(evaluation_budget=2))
    result = search.run()
    first = search.attempts[0].evaluations[0]["evaluation"]
    assert first["status"] == "timeout" and first["fitness"] is None
    assert "total time limit of 0.3 seconds" in first["failure"]
    assert result.best.id == 1 and result.best.evaluation.fitness == 2
    assert len(search.archive) == 1 and result.model_calls == 2
    assert "0.3 seconds in total" in model.prompts[1]


@pytest.mark.skipif(os.name != "posix", reason="Process group cleanup is POSIX-specific")
def test_timeout_terminates_nested_processes(task, tmp_path):
    path = tmp_path / "child.pid"
    task.time_limit = .5
    source = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "def solve(x):\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"    Path({str(path)!r}).write_text(str(child.pid))\n"
        "    time.sleep(60)\n    return 1\n"
    )
    started = time.monotonic()
    result = Evaluator(task).evaluate(source)
    assert result.status == "timeout" and time.monotonic() - started < 3
    pid = int(path.read_text())
    for _ in range(50):
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(.02)
    else:
        pytest.fail(f"Evaluation left a running child process: {pid}")
