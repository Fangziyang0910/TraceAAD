"""Offline checks for the paper's delta loop, bounded selection, and recovery."""

import copy
import random
import re

import numpy as np
import pytest

from baselines.deltaevolve import DeltaEvolve
from baselines.deltaevolve.population import Population
from baselines.deltaevolve.prompt import build_prompt, parse_response
from baselines.shared.profiler import ProfilerBase
from core import Evaluation
from core.code import parse_program
from experiments.infra.resume_baseline import recover
from tests.support import TokenLLM
from traceaad.storage import read_json, rows, write_json


TEMPLATE = "def solve(x):\n    return 1\n"
LOG = """
#DELTA-SUMMARY-START
FROM: A constant objective estimate.
TO: A smaller constant objective estimate.
#DELTA-SUMMARY-END
#DELTA-PLAN-DETAILS-START
[Modification 1]
COMPONENT: Constant estimate
OLD LOGIC: Return the previous constant.
NEW LOGIC: Decrease the returned constant by one.
HYPOTHESIS: A smaller return reduces the objective.
#DELTA-PLAN-DETAILS-END
"""


class ToyTask(Evaluation):
    def __init__(self):
        super().__init__(TEMPLATE, "Minimize the returned value.", safe_evaluate=False)
        self.evaluated = []

    def evaluate_program(self, code, function, **kwargs):
        self.evaluated.append(code)
        return float(function(0))


class Mutations(TokenLLM):
    def __init__(self, *, stop_after=None, malformed_first=False, fail_first=False):
        super().__init__()
        self.stop_after, self.malformed_first, self.fail_first = stop_after, malformed_first, fail_first

    def draw_sample_with_details(self, prompt, **kwargs):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise KeyboardInterrupt
        self.calls.append(prompt)
        if self.fail_first and len(self.calls) == 1:
            raise ValueError("Offline transport failure")
        code = prompt[1]["content"].split("--- Parent Program ---", 1)[-1]
        current = re.findall(r"return (-?\d+)", code)[0]
        response = f"<<<<<<< SEARCH\n    return {current}\n=======\n    return {int(current)-1}\n>>>>>>> REPLACE\n" + LOG
        if self.malformed_first and len(self.calls) == 1:
            response = response.replace("#DELTA-PLAN-DETAILS-END", "")
        return {"content": response, "model": "offline",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def method(directory, llm, *, budget=6, embedding=None):
    task = ToyTask()
    write_json(directory / "run_config.json", {"method": "deltaevolve", "task": "toy", "budget": budget,
                                                "method_params": {}})
    instance = DeltaEvolve(llm, task, ProfilerBase(directory), max_sample_nums=budget,
        embedding_fn=embedding or (lambda code: [len(code), code.count("-"), 1.0]), random_seed=9,
        population_size=3, archive_size=3, migration_interval=2)
    return instance, task


def node(identity, fitness, cell, *, parent_id=1):
    return {"id": identity, "fitness": fitness, "cell": cell, "parent_id": parent_id,
            "code": f"PRIVATE_CODE_{identity}_END", "iteration": identity,
            "summary": f"summary_{identity}", "plan": f"plan_{identity}", "change": "improved"}


def test_patch_keeps_helpers_and_rejects_ambiguous_invalid_or_incomplete_output():
    code = "OFFSET = 1\ndef helper(x):\n    return x + OFFSET\ndef solve(x):\n    return helper(x)\n"
    patch = "<<<<<<< SEARCH\nOFFSET = 1\n=======\nOFFSET = 2\n>>>>>>> REPLACE\n" + LOG
    program, summary, plan = parse_response(patch, code, parse_program(TEMPLATE))
    namespace = {}
    exec(str(program), namespace)
    assert namespace["solve"](3) == 5 and "def helper" in str(program)
    assert summary.startswith("FROM:") and "HYPOTHESIS:" in plan
    decorated = patch.replace('#DELTA-', '### DELTA-').replace('\n', '\r\n')
    assert str(parse_response(decorated, code, parse_program(TEMPLATE))[0]) == str(program)
    for bad in (patch.replace("#DELTA-SUMMARY-END", ""),
                patch.replace("HYPOTHESIS:", "REASON:"),
                patch.replace("OLD LOGIC: Return the previous constant.", "OLD LOGIC:"),
                patch.replace("FROM: A constant objective estimate.", "FROM:"),
                patch + "#DELTA-SUMMARY-START\nFROM: Incomplete duplicate\n",
                patch.replace("OFFSET = 2", "OFFSET = ("),
                patch.replace("OFFSET = 1", "return"),
                patch.replace("OFFSET = 1", "def solve(x):").replace("OFFSET = 2", "def solve(y):")):
        with pytest.raises((ValueError, SyntaxError)):
            parse_response(bad, code, parse_program(TEMPLATE))


def test_patch_preserves_module_initialization_and_docstring_fences():
    code = '''if True:
    OFFSET = 3
def solve(x):
    """Example:
```python
    return 99
```
    """
    return x + OFFSET
OFFSET += 1
'''
    patch = "<<<<<<< SEARCH\n    OFFSET = 3\n=======\n    OFFSET = 7\n>>>>>>> REPLACE\n" + LOG
    program, _, _ = parse_response(patch, code, parse_program(TEMPLATE))
    namespace = {}
    exec(str(program), namespace)
    assert namespace['solve'](2) == 10
    assert 'return 99' in namespace['solve'].__doc__


def test_long_program_embedding_preserves_all_utf8_and_sums_usage(monkeypatch):
    from experiments.deltaevolve.embedding import CHUNK_BYTES, CodeEmbedding, ServiceEmbedding
    chunks = []

    def embed(self, text):
        chunks.append(text)
        self.last_usage = {'total_tokens': len(text), 'prompt_tokens': len(text)}
        return [len(chunks), 1.0]

    monkeypatch.setattr(ServiceEmbedding, '__call__', embed)
    service = CodeEmbedding()
    code = '算法🌍 = 1\n' * 2000
    vector = service(code)
    weights = [len(chunk.encode('utf-8')) for chunk in chunks]
    assert ''.join(chunks) == code and max(weights) <= CHUNK_BYTES
    assert vector == pytest.approx(np.average([[i+1, 1] for i in range(len(chunks))], axis=0, weights=weights))
    assert service.last_usage['total_tokens'] == len(code)
    chunks.clear()
    assert service('short code') == [1.0, 1.0]
    assert service.last_usage['total_tokens'] == len('short code')


def test_local_embedding_uses_its_own_endpoint_model_and_credentials(monkeypatch):
    from experiments.shinka_evo import embedding
    from types import SimpleNamespace
    monkeypatch.setenv('SHINKA_EMBEDDING_BASE_URL', 'http://127.0.0.1:8002/v1')
    monkeypatch.setenv('SHINKA_EMBEDDING_MODEL', 'qwen3-embedding-4b')
    monkeypatch.setenv('SHINKA_EMBEDDING_API_KEY_ENV', 'LOCAL_EMBEDDING_KEY')
    monkeypatch.setenv('LOCAL_EMBEDDING_KEY', 'local-test-key')
    calls = {}

    def create(**kwargs):
        calls['request'] = kwargs
        return SimpleNamespace(usage=None, data=[SimpleNamespace(embedding=[1., 2., 3.])])

    def client(**kwargs):
        calls['client'] = kwargs
        return SimpleNamespace(embeddings=SimpleNamespace(create=create), close=lambda: None)

    monkeypatch.setattr(embedding, 'OpenAI', client)
    service = embedding.CodeEmbedding()
    assert service('code') == [1., 2., 3.]
    assert calls['client']['base_url'] == 'http://127.0.0.1:8002/v1'
    assert calls['client']['api_key'] == 'local-test-key'
    assert calls['request']['model'] == 'qwen3-embedding-4b'
    service.close()


def test_history_uses_age_not_score_and_only_parent_has_code():
    parent, old, recent = node(25, -9, (0, 0)), node(2, -100, (1, 1)), node(24, 100, (2, 2))
    prompt, rendered = build_prompt("task", parent, [old], [recent], 25, 10)
    user = prompt[1]["content"]
    assert [r["level"] for r in rendered] == [1, 2]
    assert "summary_2" in user and "plan_24" in user and "plan_2\n" not in user
    assert parent["code"] in user and old["code"] not in user and recent["code"] not in user
    assert "maximize: 9" in user and "-100" not in user


def test_bounded_map_elites_selection_migration_and_embedding_validation():
    pop = Population(random.Random(4), population_size=2, archive_size=3,
                     migration_interval=2, migration_rate=0.5)
    seed = node(1, 9, (0, 0), parent_id=None)
    pop.add(seed, 0)
    for island in (1, 2):
        pop._insert(1, island)
    for i in range(2, 8):
        pop.add(node(i, -i, (i, i)), (i-2) % 3)
    pop.add(node(8, 99, (2, 2)), 0)
    assert 8 not in pop.islands[0].values()
    before = copy.deepcopy(pop.islands)
    transfers = pop.migrate(2)
    assert {t["source_island"] for t in transfers} == {0, 1, 2}
    assert all(t["node_id"] in before[t["source_island"]].values() for t in transfers)
    assert all(len(cells) <= 2 for cells in pop.islands)
    assert pop.archive_ids == [7, 6, 5]
    for _ in range(30):
        _, parent, top, diverse = pop.select(3, 2)
        ids = [n["id"] for n in top + diverse]
        assert len(set(ids)) == len(ids) and parent["id"] not in ids and 1 not in ids
    assert all(0 <= c <= 9 for c in pop.cell([1.0, 2.0, 3.0]))
    for invalid in ([0, 0, 0], [np.nan, 1, 0], [1, 2]):
        with pytest.raises(ValueError):
            pop.cell(invalid)


def test_returned_invalid_output_counts_but_transport_failure_does_not(tmp_path):
    for mode in ("malformed_first", "fail_first"):
        llm = Mutations(**{mode: True})
        instance, task = method(tmp_path / mode, llm, budget=3)
        instance.run()
        summary = read_json(tmp_path / mode / "summary.json")
        assert summary["status"] == "finished" and summary["budget_used"] == 3
        assert len(llm.calls) == (2 if mode == "malformed_first" else 3)
        assert len(task.evaluated) == (2 if mode == "malformed_first" else 3)
        assert summary['evaluation_calls'] == len(task.evaluated)
        candidates = [r for r in rows(tmp_path / mode / "events.jsonl") if r["kind"] == "candidate"]
        assert [r["budget_used"] for r in candidates] == [1, 2, 3]
        assert candidates[-1]["program"]["parent_id"] is not None
        assert summary["token_usage"]["total_tokens"] == 60


def test_resume_preserves_selection_rng_and_does_not_reevaluate_history(tmp_path):
    full, _ = method(tmp_path / "full", Mutations())
    full.run()
    first, _ = method(tmp_path / "split", Mutations(stop_after=2))
    first.run()
    resumed, task = method(tmp_path / "split", Mutations())
    report = recover(resumed, tmp_path / "split")
    assert report["budget_used"] == 3 and task.evaluated == []
    resumed.run()
    assert len(task.evaluated) == 3
    assert resumed._population.nodes == full._population.nodes
    assert resumed._population.islands == full._population.islands
    assert resumed._rng.getstate() == full._rng.getstate()
    assert read_json(tmp_path / 'split' / 'summary.json')['evaluation_calls'] == 6


def test_embedding_failure_resumes_evaluated_candidate_even_at_budget_limit(tmp_path, monkeypatch):
    from experiments.infra import resume_baseline
    count = 0

    def embedding(code):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("Offline embedding outage")
        return [1.0, 0.0, 1.0]

    first, _ = method(tmp_path, Mutations(), budget=2, embedding=embedding)
    first.run()
    assert read_json(tmp_path / "summary.json")["status"] == "aborted"
    resumed, task = method(tmp_path, Mutations(stop_after=0), budget=2)
    monkeypatch.setattr(resume_baseline, "build", lambda *a: resumed)
    monkeypatch.setattr("benchmarks.tasks.validate_task_interface", lambda *a: None)
    monkeypatch.setattr("sys.argv", ["resume", "--run-dir", str(tmp_path), "--execute"])
    monkeypatch.setenv("TRACEAAD_EVALUATION_LOG", "")
    resume_baseline.main()
    assert task.evaluated == [] and resumed.best_function.score == 0
    assert read_json(tmp_path / "summary.json")["status"] == "finished"


@pytest.mark.parametrize('stop_at', [1, 2])
def test_interrupted_evaluation_resumes_without_spending_another_attempt(tmp_path, monkeypatch, stop_at):
    from experiments.infra import resume_baseline
    first, _ = method(tmp_path, Mutations(), budget=2)
    evaluate = first._evaluator.evaluate_program_record_time_with_details
    count = 0

    def interrupted(program):
        nonlocal count
        count += 1
        if count == stop_at:
            raise KeyboardInterrupt
        return evaluate(program)

    monkeypatch.setattr(first._evaluator, 'evaluate_program_record_time_with_details', interrupted)
    first.run()
    assert read_json(tmp_path / 'summary.json')['status'] == 'interrupted'
    llm = Mutations()
    resumed, task = method(tmp_path, llm, budget=2)
    monkeypatch.setattr(resume_baseline, 'build', lambda *a: resumed)
    monkeypatch.setattr('benchmarks.tasks.validate_task_interface', lambda *a: None)
    monkeypatch.setattr('sys.argv', ['resume', '--run-dir', str(tmp_path), '--execute'])
    monkeypatch.setenv('TRACEAAD_EVALUATION_LOG', '')
    resume_baseline.main()
    assert read_json(tmp_path / 'summary.json')['status'] == 'finished'
    assert len(llm.calls) == 2 - stop_at
    assert len(task.evaluated) == 3 - stop_at
    assert resumed.best_function.score == 0
    assert resumed._iteration == 1 and resumed._tot_sample_nums == 2


@pytest.mark.parametrize('task_name', [
    'tsp_construct', 'cvrp_aco', 'fssp_gls', 'graph_colouring', 'jssp_construct', 'op_aco'])
def test_real_solver_entry_final_selection_monitor_and_config_rebuild(tmp_path, monkeypatch, task_name):
    from benchmarks.tasks import training_task
    from traceaad.instance_evaluation import _instance_view
    from baselines.deltaevolve.prompt import program_from_code
    from experiments.deltaevolve import run
    from experiments.infra import base
    from experiments.infra.resume_baseline import build, close_recovery_method
    from experiments.visualization import ResultsMonitor

    monkeypatch.delenv("TRACEAAD_SCHEDULER_SOCKET", raising=False)
    monkeypatch.setenv("TRACEAAD_EVALUATION_LOG", "")
    original, settings = training_task(task_name)
    task = _instance_view(original, 0, original.timeout_seconds)
    monkeypatch.setattr(base, "training_task", lambda *a, **k: (task, settings))
    code = str(program_from_code(task.template_program, parse_program(task.template_program))).rstrip('\n')
    reply = f"<<<<<<< SEARCH\n{code}\n=======\n{code}\nDELTA_SMOKE = 1\n>>>>>>> REPLACE\n" + LOG
    clients = []

    def client(**kwargs):
        llm = TokenLLM(reply)
        clients.append(llm)
        return llm

    monkeypatch.setattr(run, "build_llm_client", client)
    monkeypatch.setattr(run, "CodeEmbedding", lambda: lambda code: [1.0, 2.0, 3.0])
    spec = run.RunSpec(task=task_name, budget=2, eval_workers=1,
                       run_name="integration", experiments_root=tmp_path)
    directory = run.run_experiment(spec)
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "finished" and summary["selection_evaluations"] == 0
    candidates = [r for r in rows(directory / 'events.jsonl') if r['kind'] == 'candidate']
    assert all(r['valid'] for r in candidates)
    assert candidates[0]['fitness'] == candidates[1]['fitness'] == summary['best']['fitness']
    detail = ResultsMonitor(tmp_path).run_detail("deltaevolve", task_name, "integration")
    assert detail["best"]["code"] == (directory / "best_program.py").read_text()
    rebuilt = build(read_json(directory / "run_config.json"), directory)
    try:
        assert clients[-1].calls == []
        report = recover(rebuilt, directory)
        assert report["recovery"] == "search_checkpoint" and rebuilt._tot_sample_nums == 2
    finally:
        close_recovery_method(rebuilt)
