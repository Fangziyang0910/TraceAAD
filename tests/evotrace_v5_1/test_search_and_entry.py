"""EvoTrace V5.1 端到端搜索、同 run 随机屏蔽、崩溃续跑与统一实验入口测试。"""

import json
import threading
import types
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import evotrace_v5_1
from evotrace_v5_1 import Config, EvoTrace, OpenAIClient, Task, cli
from evotrace_v5_1.models import CANDIDATE_RESPONSE_FORMAT
from experiments.traceaad import launch_host, run

ROUTE = [
    "--task",
    "tsp_construct",
    "--run-name",
    "v51_run",
    "--batch",
    "v51",
    "--scheduler-socket",
    "/fake",
]


def make_task(**_kwargs) -> Task:
    return Task(
        "Minimize the returned number.",
        "def solve(x):\n    pass\n",
        lambda _src, fn: fn(1),
        5.0,
        metadata={"n_instance": 2, "settings": {"timeout_seconds": 5.0}},
    )


class DeterministicFakeModel:
    """Offline model returning alternating better and worse programs to exercise both upgrades and evictions."""

    model = "offline-v51"

    def __init__(self, path=None, label=None):
        self.path = path
        self.scheduler = {"endpoints": []}
        self.request_options = {}
        self._state = {
            "generations": 0,
            "reflections": 0,
            "reflection_prompts": [],
            "generation_prompts": [],
        }

    @property
    def generations(self) -> int:
        return self._state["generations"]

    @generations.setter
    def generations(self, value: int) -> None:
        self._state["generations"] = value

    @property
    def reflections(self) -> int:
        return self._state["reflections"]

    @reflections.setter
    def reflections(self, value: int) -> None:
        self._state["reflections"] = value

    @property
    def reflection_prompts(self) -> list[str]:
        return self._state["reflection_prompts"]

    @property
    def generation_prompts(self) -> list[str]:
        return self._state["generation_prompts"]

    def complete(self, prompt, **settings):
        schema = self.request_options.get("response_format") or CANDIDATE_RESPONSE_FORMAT
        required = schema["json_schema"]["schema"]["required"]
        if "code" not in required:
            self._state["reflections"] += 1
            self._state["reflection_prompts"].append(prompt)
            return json.dumps(
                {"experience": f"Unrealized idea {self._state['reflections']}: try decoupled scaling."}
            )
        self._state["generations"] += 1
        self._state["generation_prompts"].append(prompt)
        n = self._state["generations"]
        # Produce diverse scores (some improving, many worse so capacity eviction & leaf reflection trigger)
        score = 50 - n if n % 5 == 0 else 60 + (n % 17)
        return json.dumps(
            {
                "analysis": f"Thoughtful analysis for attempt {n}",
                "code": f"def solve(x):\n    offset_{n} = {n}\n    return {score} + (offset_{n} - offset_{n})\n",
                "design": f"Strategy variant {n} targeting {score}",
            }
        )


def test_end_to_end_capacity_bound_leaf_reflection_and_within_run_masking(tmp_path):
    task = make_task()
    model = DeterministicFakeModel()
    cfg = Config(
        evaluation_budget=30,
        initial_size=3,
        population_capacity=5,
        selection_lambda=10.0,
        reflection_mask_prob=0.5,
        seed=7,
    )
    search = EvoTrace(task, model, config=cfg, run_dir=tmp_path / "mask_run", reflection=True)
    max_tree_size = 0

    def check_invariant(_attempt):
        nonlocal max_tree_size
        max_tree_size = max(max_tree_size, len(search.tree.parents))
        assert len(search.tree.parents) <= cfg.population_capacity

    result = search.run(on_attempt=check_invariant)
    assert result.status == "finished"
    assert result.evaluations == 30
    assert max_tree_size <= cfg.population_capacity
    # Leaf evictions must have triggered Second-Chance reflections
    assert model.reflections > 0
    # Verify NO backward-looking "how we succeeded" reflections ever ran
    assert all(
        "still-untested ideas" in p or "Candidate Notes to Clean / Merge" in p
        for p in model.reflection_prompts
    )

    # Check within-run randomized masking telemetry on Refine attempts
    rows = [json.loads(line) for line in (tmp_path / "mask_run" / "attempts.jsonl").read_text().splitlines()]
    refine_ctxs = [
        r["attempt"]["record"]["context"]
        for r in rows
        if r["attempt"]["operator"] == "Refine" and "context" in r["attempt"]["record"]
    ]
    available = [c for c in refine_ctxs if c["memory_available"]]
    assert available, "Expected Refine steps with unrealized-idea notes available"
    assert all(c["memory_source"] in {"self", "parent", "both"} for c in available)
    assert any(c["memory_shown"] for c in available)
    assert any(c["memory_masked"] for c in available)


def test_no_reflection_mode_makes_zero_reflection_calls(tmp_path):
    task = make_task()
    model = DeterministicFakeModel()
    cfg = Config(evaluation_budget=18, initial_size=3, population_capacity=4, seed=3)
    search = EvoTrace(task, model, config=cfg, run_dir=tmp_path / "noref", reflection=False)
    result = search.run()
    assert result.status == "finished" and result.evaluations == 18
    assert model.reflections == 0
    assert all("[Unrealized Ideas Worth a Second Chance]" not in p for p in model.generation_prompts)


def test_resume_matches_uninterrupted_run_state_and_enforces_immutable_settings(tmp_path):
    task = make_task()
    # Uninterrupted 14-eval run
    m_full = DeterministicFakeModel()
    cfg_full = Config(evaluation_budget=14, initial_size=3, population_capacity=4, seed=11)
    s_full = EvoTrace(task, m_full, config=cfg_full, run_dir=tmp_path / "full", reflection=True)
    res_full = s_full.run()

    # Split run: 7 evals then resume to 14 evals
    m_part1 = DeterministicFakeModel()
    cfg_p1 = Config(evaluation_budget=7, initial_size=3, population_capacity=4, seed=11)
    s_p1 = EvoTrace(task, m_part1, config=cfg_p1, run_dir=tmp_path / "split", reflection=True)
    s_p1.run()

    m_part2 = DeterministicFakeModel()
    m_part2.generations = m_part1.generations
    m_part2.reflections = m_part1.reflections
    s_p2 = EvoTrace(
        task, m_part2, config=cfg_full, run_dir=tmp_path / "split", resume=True, reflection=True
    )
    res_split = s_p2.run()

    assert res_split.best.fitness == res_full.best.fitness
    assert s_p2.tree.snapshot() == s_full.tree.snapshot()

    # Conflicting config on resume must raise ValueError
    with pytest.raises(ValueError, match="reflection_mask_prob"):
        EvoTrace(
            task,
            DeterministicFakeModel(),
            config=Config(
                evaluation_budget=20,
                initial_size=3,
                population_capacity=4,
                reflection_mask_prob=0.5,
                seed=11,
            ),
            run_dir=tmp_path / "split",
            resume=True,
            reflection=True,
        )


@pytest.fixture
def entry(monkeypatch, tmp_path):
    models = []

    def scheduled(path, label):
        models.append(DeterministicFakeModel(path, label))
        return models[-1]

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_environment", lambda *_: None)
    monkeypatch.setattr(run, "load_evotrace_v5_1_task", make_task)
    monkeypatch.setattr(run, "ScheduledModel", scheduled)
    monkeypatch.setattr(
        run, "ScheduledEvaluator", lambda t, p, l, *, evaluator_factory: evaluator_factory(t)
    )
    monkeypatch.setattr(run, "scheduler_status", lambda _: {"instance_id": "cpu-pool"})
    monkeypatch.setattr(run.signal, "signal", lambda *_: None)
    return types.SimpleNamespace(root=tmp_path, models=models)


@pytest.mark.parametrize("method", ["evotrace.v5.1", "evotrace_v5_1"])
def test_experiment_entry_routes_v5_1_and_forwards_mask_prob(entry, monkeypatch, capsys, method):
    run.main(
        [
            *ROUTE,
            "--method",
            method,
            "--dry-run",
            "--population-capacity",
            "16",
            "--selection-lambda",
            "8.0",
            "--reflection-mask-prob",
            "0.5",
        ]
    )
    meta = json.loads(capsys.readouterr().out)
    assert meta["method"] == meta["experiment"] == "evotrace_v5_1"
    assert meta["method_params"]["population_capacity"] == 16
    assert meta["method_params"]["selection_lambda"] == 8.0
    assert meta["method_params"]["reflection_mask_prob"] == 0.5
    assert meta["reflection"] is True
    assert meta["mechanisms"] == list(EvoTrace.mechanisms_for(True))

    # Reject V3/V4-only and Base-only flags on V5.1
    for bad in (["--rank-half-life", "5"], ["--max-representatives", "16"], ["--selection-temperature", "0.5"]):
        with pytest.raises(SystemExit) as err:
            run.main([*ROUTE, "--method", method, "--dry-run", *bad])
        assert err.value.code == 2

    # Reject V5.1-only flags on V5
    with pytest.raises(SystemExit) as err:
        run.main([*ROUTE, "--method", "evotrace.v5", "--dry-run", "--reflection-mask-prob", "0.5"])
    assert err.value.code == 2

    # Launcher forwards V5.1 parameters cleanly into commands & manifest
    monkeypatch.setattr(launch_host, "ROOT", entry.root)
    monkeypatch.setattr(launch_host, "scheduler_status", lambda _: {"instance_id": "cpu-pool"})
    launch_host.main(
        [
            "--method",
            "evotrace.v5.1",
            "--batch",
            "v51_b",
            "--tasks",
            "tsp_construct",
            "--repeats",
            "1",
            "--scheduler-socket",
            "/fake",
            "--reflection-mask-prob",
            "0.5",
            "--population-capacity",
            "20",
        ]
    )
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["method"] == "evotrace_v5_1"
    assert manifest["v5_1"]["reflection_mask_prob"] == 0.5
    assert manifest["v5_1"]["population_capacity"] == 20
    cmd = manifest["plan"][0]["command"]
    assert cmd[cmd.index("--reflection-mask-prob") + 1] == "0.5"
    assert cmd[cmd.index("--population-capacity") + 1] == "20"


def test_experiment_entry_runs_v5_1_end_to_end_and_resumes(entry, capsys):
    common = [
        *ROUTE,
        "--method",
        "evotrace.v5.1",
        "--initial-size",
        "2",
        "--population-capacity",
        "4",
        "--reflection-mask-prob",
        "0.5",
        "--skip-heldout",
    ]
    run.main([*common, "--budget", "8"])
    out1 = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out1["evaluations"] == 8 and out1["status"] == "finished"

    # Resume extending budget works without re-specifying flags
    run.main([*ROUTE, "--method", "evotrace.v5.1", "--budget", "12", "--resume", "--skip-heldout"])
    out2 = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out2["evaluations"] == 12

    # Resume with conflicting --reflection-mask-prob fails fast
    with pytest.raises(SystemExit) as err:
        run.main(
            [
                *ROUTE,
                "--method",
                "evotrace.v5.1",
                "--budget",
                "16",
                "--resume",
                "--reflection-mask-prob",
                "0.0",
            ]
        )
    assert err.value.code == 2


def test_standalone_cli_runs_against_local_http_endpoint(tmp_path, monkeypatch, capsys):
    (tmp_path / "demo_task.py").write_text(
        "from evotrace_v5_1 import Task\n"
        "def make():\n"
        "    return Task('Min', 'def solve(x):\\n    pass\\n', lambda s, fn: fn(1), 2.0)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    counter = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            required = payload["response_format"]["json_schema"]["schema"]["required"]
            if "code" in required:
                counter["n"] += 1
                n = counter["n"]
                content = json.dumps(
                    {
                        "analysis": f"Idea {n}",
                        "code": f"def solve(x):\n    return {20 - n}\n",
                        "design": f"Const {n}",
                    }
                )
            else:
                content = json.dumps({"experience": "Salvaged idea."})
            body = json.dumps(
                {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        code = cli.main(
            [
                "--task-factory",
                "demo_task:make",
                "--run-dir",
                str(tmp_path / "cli_run"),
                "--base-url",
                url,
                "--model",
                "local-v51",
                "--budget",
                "4",
                "--initial-size",
                "2",
                "--population-capacity",
                "3",
                "--reflection-mask-prob",
                "0.5",
            ]
        )
        assert code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["evaluations"] == 4 and out["status"] == "finished"
        saved = json.loads((tmp_path / "cli_run" / "run_config.json").read_text())
        assert saved["method"] == "evotrace_v5_1"
        assert saved["config"]["population_capacity"] == 3
        assert saved["config"]["reflection_mask_prob"] == 0.5
        assert evotrace_v5_1.OpenAIClient is OpenAIClient
        assert asdict(Config())["population_capacity"] == 24
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
