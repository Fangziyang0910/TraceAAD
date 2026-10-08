"""Instance deadlines, bounded parallelism, deterministic seeds and cleanup."""

import json
import multiprocessing
import random
from dataclasses import asdict

import pytest

from tests.support import TinyEvaluation, TokenLLM, response, small_task
from benchmarks.tasks import CO_TASKS, FIXED_TASKS, CLASSES, FUNCTION_SECONDS
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from traceaad.v10_21 import Config, TraceAADV1021


class IndexedEvaluation(TinyEvaluation):
    def __init__(self, count=4):
        super().__init__()
        self.n_instance = count
        self.problem_size = 1
        self._datasets = list(range(count))

    def evaluate_program(self, source, function):
        return sum(function(x) for x in self._datasets) / len(self._datasets)


def test_per_call_workers_and_deadlines(tmp_path):
    code = f'''import time
def score(x):
    start = time.monotonic()
    time.sleep(0.3)
    end = time.monotonic()
    with open({str(tmp_path)!r} + '/' + str(x), 'w') as f:
        f.write(str(start) + ',' + str(end))
    return x
'''
    evaluator = InstanceProgramEvaluator(IndexedEvaluation(), timeout_seconds=0.1)
    assert evaluator.evaluate(code, 'short')['failure']['kind'] == 'timeout'
    serial = evaluator.evaluate(code, 'serial', timeout_seconds=1, n_workers=1)
    assert serial['fitness'] == 1.5 and serial['seconds'] > 1
    parallel = evaluator.evaluate(code, 'parallel', timeout_seconds=1, n_workers=2)
    assert parallel['fitness'] == serial['fitness']
    assert parallel['calls'] == 4
    assert parallel['instance_seconds'] > parallel['seconds']
    intervals = [tuple(map(float, (tmp_path / str(i)).read_text().split(','))) for i in range(4)]
    concurrency = [sum(start <= t < end for start, end in intervals) for t, _ in intervals]
    assert max(concurrency) == 2
    assert evaluator.timeout_seconds == 0.1 and evaluator.n_workers == 1
    assert serial['evaluations'][0]['protocol'] == parallel['evaluations'][0]['protocol']
    assert evaluator.evaluate(code, 'default')['failure']['kind'] == 'timeout'


def test_seed_and_globals_do_not_depend_on_workers():
    code = '''import random
count = 0
def score(x):
    global count
    count += 1
    assert count == 1
    return random.random()
'''
    evaluator = InstanceProgramEvaluator(IndexedEvaluation(), seeds=(30, 40))
    serial = evaluator.evaluate(code, 'x', n_workers=1)
    parallel = evaluator.evaluate(code, 'x', n_workers=4)
    assert serial['fitness'] == parallel['fitness']
    for record in parallel['evaluations']:
        assert [r['score'] for r in record['instances']] == [random.Random(record['seed'] + i).random() for i in range(4)]


@pytest.mark.parametrize('body,kind', [('while True: pass', 'timeout'), ('raise ValueError("bad")', 'runtime_error'),
                                    ('return float("nan")', 'invalid_output'), ('os._exit(7)', 'runtime_error')])
def test_failure_cancels_other_instances_and_releases_workers(body, kind):
    before = {p.pid for p in multiprocessing.active_children()}
    code = f'import os, time\ndef score(x):\n    if x == 0:\n        {body}\n    time.sleep(10)\n    return 1'
    result = InstanceProgramEvaluator(IndexedEvaluation()).evaluate(code, 'bad', timeout_seconds=0.5, n_workers=2)
    assert result['fitness'] is None and result['failure']['kind'] == kind
    assert result['seconds'] < 3
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_spawn_and_no_measurement():
    task = IndexedEvaluation(2)
    task.fork_proc = False
    result = InstanceProgramEvaluator(task, measure_calls=False, function_seconds=None).evaluate(
        'def score(x): return x', 'x', n_workers=2)
    assert result['fitness'] == 0.5 and result['calls'] is None


class SlowSolver(IndexedEvaluation):
    def evaluate_program(self, source, function):
        import time
        time.sleep(0.4)  # the fixed solver's own time
        return super().evaluate_program(source, function)


BUSY = 'import time\ndef score(x):\n    end = time.process_time() + {}\n    while time.process_time() < end:\n        pass\n    return x'
SLEEP = 'import time\ndef score(x):\n    time.sleep({})\n    return x'


def test_the_budget_counts_cpu_inside_the_function_only():
    evaluator = InstanceProgramEvaluator(SlowSolver(2), timeout_seconds=5, function_seconds=0.2)
    fast = evaluator.evaluate('def score(x): return x', 'fast', n_workers=2)
    assert fast['failure'] is None and min(r['seconds'] for r in fast['evaluations'][0]['instances']) > 0.4
    waiting = evaluator.evaluate(SLEEP.format(0.5), 'waiting', n_workers=2)
    assert waiting['failure'] is None  # waiting uses no CPU
    assert waiting['function_seconds'] > 0.9 and waiting['function_cpu_seconds'] < 0.2
    within = evaluator.evaluate(BUSY.format(0.1), 'within', n_workers=2)
    assert within['failure'] is None and 0.15 < within['function_cpu_seconds'] < 0.4
    over = evaluator.evaluate(BUSY.format(0.6), 'over', n_workers=2)
    assert over['failure']['kind'] == 'timeout' and over['failure']['error_type'] == 'FunctionBudgetExceeded'
    assert over['seconds'] < 1.4  # stopped near the budget, not at the end of the computation
    hang = InstanceProgramEvaluator(SlowSolver(2), timeout_seconds=0.3, function_seconds=1).evaluate(
        'def score(x): return x', 'hang', n_workers=2)
    assert hang['failure']['kind'] == 'timeout' and hang['failure']['error_type'] == 'TimeoutError'


@pytest.mark.parametrize('task', FIXED_TASKS)
def test_fixed_task_initializes_the_program_once_per_instance(task, tmp_path):
    evaluation = CLASSES[task](limit=1)
    evaluation.outer_settings = {name: 1 for name in evaluation.outer_settings}
    marker = tmp_path / 'initializations'
    code = f'with open({str(marker)!r}, "a") as output:\n    output.write("init\\n")\n' + str(evaluation.template_program)
    result = InstanceProgramEvaluator(evaluation).evaluate(code, 'initialization')
    assert result['failure'] is None
    assert marker.read_text().splitlines() == ['init']


@pytest.mark.parametrize('timeout,workers', [(0, 1), (-1, 1), (float('nan'), 1), (float('inf'), 1),
                                            (True, 1), (10, 0), (10, 1.5), (10, True)])
def test_invalid_execution_parameters(timeout, workers):
    with pytest.raises(ValueError):
        Config(eval_timeout_seconds=timeout, eval_workers=workers)
    with pytest.raises(ValueError):
        InstanceProgramEvaluator(IndexedEvaluation()).evaluate('def score(x): return x', 'x',
                                                              timeout_seconds=timeout, n_workers=workers)


@pytest.mark.parametrize('task', CO_TASKS)
def test_six_task_scores_match_serial_fixed_solvers(task):
    if task in FIXED_TASKS:
        evaluation = CLASSES[task](limit=2)
        evaluation.outer_settings = {name: 1 for name in evaluation.outer_settings}
    else:
        evaluation = small_task(task)
    code = str(evaluation.template_program)
    old = ProgramEvaluator(evaluation).evaluate(code, 'template')
    evaluator = InstanceProgramEvaluator(evaluation)
    serial = evaluator.evaluate(code, 'template', n_workers=1)
    parallel = evaluator.evaluate(code, 'template', n_workers=2)
    assert old['failure'] is None and serial['failure'] is None and parallel['failure'] is None
    assert serial['fitness'] == pytest.approx(old['fitness'])
    assert parallel['fitness'] == serial['fitness']
    assert parallel['calls'] == serial['calls'] == old['calls']


def test_search_and_heldout_keep_execution_parameters(tmp_path, monkeypatch):
    from experiments.infra import search_heldout, evaluate
    config = Config(budget=1, roots=1, eval_timeout_seconds=2.5, eval_workers=2)
    method = TraceAADV1021(evaluation=IndexedEvaluation(2), llm=TokenLLM(response(2)),
                           run_dir=tmp_path, config=config, task='tsp_construct')
    method.run()
    assert method.config.eval_timeout_seconds == 2.5
    assert method.facts.evaluations[0]['n_workers'] == 2
    node = method.programs[1]
    assert node['eval_seconds'] == method.facts.evaluations[0]['instance_seconds']
    assert node['eval_wall_seconds'] == method.facts.evaluations[0]['seconds']
    (tmp_path / 'run_config.json').write_text(json.dumps({'task': 'tsp_construct', 'method': 'v1021',
                                                        'method_params': asdict(config)}))
    for module in (search_heldout, evaluate):
        monkeypatch.setattr(module, 'heldout_task', lambda *args: IndexedEvaluation(2))
    frozen = search_heldout.evaluate_run(tmp_path)
    generic = evaluate.evaluate_run(tmp_path, ['eval'], condition='traceaad')[0]
    for result in (frozen, generic):
        assert result['timeout_scope'] == 'instance' and result['timeout_seconds'] == 2.5
        assert result['function_seconds'] == FUNCTION_SECONDS
        assert result['workers'] == 2 and result['fitness'] == 2
        assert result['evaluations'][0]['calls'] == 2
    assert generic['protocol'] == frozen['protocol']
    from experiments.infra.diagnose_search import diagnose
    assert diagnose(tmp_path)['computation']['time_limit'] == 5


def test_resume_continues_under_changed_execution(tmp_path):
    method = TraceAADV1021(evaluation=IndexedEvaluation(2), llm=TokenLLM(response(1)),
                           run_dir=tmp_path, config=Config(budget=4, roots=2))
    method._roots()
    resumed = TraceAADV1021(evaluation=IndexedEvaluation(2), llm=TokenLLM(), run_dir=tmp_path,
                            config=Config(budget=4, roots=2, eval_timeout_seconds=20))
    assert resumed.attempts == 1 and resumed.config.eval_timeout_seconds == 20


def test_cli_and_batch_forward_execution_parameters(capsys):
    from experiments.traceaad_v10_21.run import main
    from experiments.traceaad_v10_21.launch_local import main as launch
    main(['--task', 'tsp_construct', '--dry-run', '--eval-timeout-seconds', '3.5', '--eval-workers', '4'])
    payload = json.loads(capsys.readouterr().out)
    assert payload['config']['eval_workers'] == 4 and payload['search_timeout'] == 3.5
    assert payload['evaluation_execution']['timeout_scope'] == 'instance'
    launch(['--batch', 'test', '--eval-timeout-seconds', '3.5', '--eval-workers', '4'])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload['plan']) == 18
    assert all('--eval-workers=4' in row['command'] and '--eval-timeout-seconds=3.5' in row['command'] for row in payload['plan'])


def test_freeze_keeps_the_best_program_valid_under_the_current_evaluator(tmp_path):
    slow = 'import time\ndef score(x):\n    time.sleep(0.5)\n    return -100\n'
    answers = [response(1), response(2), response(3)]
    method = TraceAADV1021(evaluation=IndexedEvaluation(2), llm=TokenLLM(*answers), run_dir=tmp_path,
                           config=Config(budget=3, roots=3, eval_timeout_seconds=0.3))
    for _ in range(3):
        method._roots()
    # A program scored earlier under a looser limit ranks first but now times out.
    best = method.archive[1]
    best.update(code=slow, key='slow', fitness=-100, score=-100)
    method._freeze()
    assert method.phase == 'finished'
    assert method.progress.selected_id == 2
    assert [r['node_id'] for r in method.progress.selection_results][:1] == [1]
    assert method.progress.selection_results[0]['failure']['kind'] == 'timeout'
