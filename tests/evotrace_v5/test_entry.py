"""EvoTrace V5 entry points: standalone CLI, experiment routing, resume conditions and held-out scope."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import types
from dataclasses import asdict
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import evotrace_v5
from evotrace_v5 import Config, EvoTrace, OpenAIClient, Task, cli
from evotrace_v5.evaluation import Evaluator
from evotrace_v5.models import CANDIDATE_RESPONSE_FORMAT
from experiments.traceaad import launch_host, run

TREE_PARAMETERS = [
    ["--rank-half-life", "5"],
    ["--explore-references", "4"],
    ["--max-representatives", "16"],
    ["--cover-distance", "0.2"],
    ["--region-size", "3"],
]
CONSTRUCTOR_KEYS = {"config", "include_experiences", "reflection", "run_dir", "resume"}
ROUTE = [
    "--task",
    "tsp_construct",
    "--run-name",
    "v5_run",
    "--batch",
    "v5",
    "--scheduler-socket",
    "/fake",
]


def make_task(**_kwargs):
    return Task(
        "Minimize",
        "def solve(x):\n    pass\n",
        lambda source, fn: fn(1),
        5,
        metadata={"n_instance": 2, "settings": {"timeout_seconds": 5}},
    )


def reply(prompt, schema, counter):
    """A candidate for generation requests; every required text field for any other schema."""
    required = (schema or CANDIDATE_RESPONSE_FORMAT)["json_schema"]["schema"]["required"]
    if "code" in required:
        return json.dumps(
            {
                "analysis": f"Plan {counter}",
                "code": f"def solve(x):\n    return {50 - counter}\n",
                "design": f"Constant {counter}",
            }
        )
    return json.dumps({name: "Condensed note." for name in required})


class FakeModel:
    """Stands in for the scheduler client; counts generation and other requests."""

    model = "offline-test"

    def __init__(self, path=None, label=None):
        self.path, self.scheduler = path, {"endpoints": []}
        self.request_options = {}
        self.generations, self.others = 0, 0

    def complete(self, prompt, **settings):
        assert self.factory is OpenAIClient
        schema = self.request_options.get("response_format")
        if schema is not None and "code" not in schema["json_schema"]["schema"]["required"]:
            self.others += 1
        else:
            self.generations += 1
        return reply(prompt, schema, self.generations)


@pytest.fixture
def entry(monkeypatch, tmp_path):
    """The route with offline task, model and evaluator, rooted in a temporary directory."""
    models = []

    def scheduled(path, label):
        models.append(FakeModel(path, label))
        return models[-1]

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_environment", lambda *_: None)
    monkeypatch.setattr(run, "load_evotrace_v5_task", make_task)
    monkeypatch.setattr(run, "ScheduledModel", scheduled)
    monkeypatch.setattr(
        run, "ScheduledEvaluator", lambda t, p, l, *, evaluator_factory: evaluator_factory(t)
    )
    monkeypatch.setattr(run, "scheduler_status", lambda _: {"instance_id": "cpu-pool"})
    monkeypatch.setattr(run.signal, "signal", lambda *_: None)
    constructed = []

    class Recording(EvoTrace):
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setitem(run.METHODS, "evotrace.v5", ("evotrace_v5", Recording))
    monkeypatch.setitem(run.METHODS, "evotrace_v5", ("evotrace_v5", Recording))
    return types.SimpleNamespace(root=tmp_path, models=models, constructed=constructed)


def run_dir(root, name="v5_run"):
    return root / "experiments_result" / "evotrace_v5" / "tsp_construct" / name


def last_json(capsys):
    return json.loads(capsys.readouterr().out.splitlines()[-1])


# --- routing ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["evotrace.v5", "evotrace_v5"])
def test_both_aliases_route_to_the_v5_package_and_result_folder(entry, capsys, method):
    run.main(
        [
            *ROUTE,
            "--method",
            method,
            "--dry-run",
            "--initial-size",
            "3",
            "--budget",
            "7",
            "--seed",
            "2",
        ]
    )
    metadata = json.loads(capsys.readouterr().out)
    assert run.METHODS[method][0] == metadata["method"] == metadata["experiment"] == "evotrace_v5"
    expected = asdict(Config(evaluation_budget=7, initial_size=3, seed=2))
    assert metadata["method_params"] == expected
    assert metadata["reflection"] is True and metadata["include_experiences"] is True
    assert metadata["mechanisms"] == list(EvoTrace.mechanisms_for(True))
    assert "allocation" not in metadata and "topology" not in metadata
    schema = metadata["llm"]["sampling"]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["analysis", "code", "design"]
    assert metadata["evaluation_execution"]["timeout_seconds"] == 10 * 2
    assert metadata["final_selection"] == "training" and metadata["backend"] == "scheduled"


def test_reflection_switch_sets_mechanisms_and_experience_flag(entry, capsys):
    run.main([*ROUTE, "--method", "evotrace.v5", "--dry-run", "--no-reflection"])
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["reflection"] is False and metadata["include_experiences"] is False
    assert metadata["mechanisms"] == list(EvoTrace.mechanisms_for(False))


@pytest.mark.parametrize(
    "flags", [*TREE_PARAMETERS, ["--selection-temperature", "0.1"], ["--reference-code"]]
)
def test_old_parameters_are_rejected_for_v5_before_runtime_setup(flags):
    with pytest.raises(SystemExit) as error:
        run.main([*ROUTE, "--method", "evotrace.v5", "--dry-run", *flags])
    assert error.value.code == 2


def test_v5_receives_exactly_its_own_constructor_arguments(entry, capsys):
    run.main(
        [
            *ROUTE,
            "--method",
            "evotrace.v5",
            "--budget",
            "4",
            "--initial-size",
            "2",
            "--skip-heldout",
        ]
    )
    ((args, kwargs),) = entry.constructed
    assert len(args) == 2 and set(kwargs) == CONSTRUCTOR_KEYS
    assert (
        kwargs["reflection"] is True
        and kwargs["include_experiences"] is True
        and kwargs["resume"] is False
    )
    assert kwargs["run_dir"] == run_dir(entry.root) / "native"


def test_eco_backend_runs_v5_with_non_thinking_requests(entry, monkeypatch, capsys):
    monkeypatch.setattr(run, "ScheduledModel", lambda *_: pytest.fail("Eco must not reserve a GPU"))
    monkeypatch.setenv("OPENDESIGN_API_KEY", "dummy-eco-test-key")
    run.main([*ROUTE, "--method", "evotrace.v5", "--backend", "opendesign-eco", "--dry-run"])
    text = capsys.readouterr().out
    metadata = json.loads(text)
    assert metadata["backend"] == "opendesign-eco" and metadata["llm"]["model"] == run.ECO_MODEL
    assert metadata["llm"]["sampling"]["reasoning_effort"] == "none"
    assert metadata["llm"]["sampling"]["response_format"] == CANDIDATE_RESPONSE_FORMAT
    assert metadata["model_routing"] == run.eco_routing()
    assert "dummy-eco-test-key" not in text


# --- batch launcher --------------------------------------------------------------------------------------


def test_batch_plan_for_v5_forwards_reflection_and_carries_no_v4_parameters():
    from argparse import Namespace

    args = Namespace(
        tasks=["tsp_construct", "cvrp_aco", "fssp_gls", "op_aco"],
        method="evotrace_v5",
        batch="b",
        experiment="evotrace_v5",
        repeats=3,
        backend="scheduled",
        budget=1000,
        initial_size=5,
        max_tokens=16384,
        model_timeout=600,
        selection_temperature=1.0,
        scheduler_socket="/fake",
        seconds_per_instance=10.0,
        env_file=None,
        skip_heldout=False,
        reflection=False,
    )
    plan = launch_host.build_plan(args)
    assert len(plan) == 12 and {row["seed"] for row in plan} == {0, 1, 2}
    for row in plan:
        command = row["command"]
        assert command[command.index("--method") + 1] == "evotrace_v5"
        assert "--no-reflection" in command and "--selection-temperature" not in command
        assert not {flag for flag in command if flag.startswith("--")} & {
            "--rank-half-life",
            "--explore-references",
            "--max-representatives",
            "--cover-distance",
            "--region-size",
            "--reference-code",
        }
        assert row["run_dir"].endswith(f"evotrace_v5/{row['task']}/{row['run_name']}")
    for name, value in (
        ("rank_half_life", 5.0),
        ("explore_references", 4),
        ("max_representatives", 16),
        ("cover_distance", 0.2),
        ("region_size", 3),
    ):
        with pytest.raises(ValueError, match="V3/V4 only"):
            launch_host.build_plan(Namespace(**{**vars(args), name: value}))


@pytest.mark.parametrize(
    "flags", [["--selection-temperature", "0.5"], ["--max-representatives", "16"]]
)
def test_launcher_rejects_old_parameters_for_v5(monkeypatch, flags):
    monkeypatch.setattr(launch_host, "scheduler_status", lambda _: {"instance_id": "test"})
    with pytest.raises((SystemExit, ValueError)):
        launch_host.main(
            [
                "--method",
                "evotrace.v5",
                "--batch",
                "b",
                "--tasks",
                "tsp_construct",
                "--repeats",
                "1",
                "--scheduler-socket",
                "/fake",
                *flags,
            ]
        )


def test_launcher_manifest_records_v5_conditions_and_eco_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch_host, "ROOT", tmp_path)
    monkeypatch.setattr(launch_host, "scheduler_status", lambda _: {"instance_id": "cpu-pool"})
    monkeypatch.setattr(
        launch_host, "launch_run", lambda _row: pytest.fail("A preview must not start workers")
    )
    launch_host.main(
        [
            "--method",
            "evotrace.v5",
            "--backend",
            "opendesign-eco",
            "--batch",
            "eco",
            "--tasks",
            "fssp_gls",
            "--repeats",
            "1",
            "--scheduler-socket",
            "/cpu",
            "--no-reflection",
        ]
    )
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["method"] == manifest["experiment"] == "evotrace_v5"
    assert manifest["reflection"] is False and manifest["include_experiences"] is False
    assert manifest["mechanisms"] == list(EvoTrace.mechanisms_for(False))
    assert (
        "allocation" not in manifest
        and "topology" not in manifest
        and "selection_temperature" not in manifest
    )
    assert (
        manifest["backend"] == "opendesign-eco" and manifest["model_routing"] == run.eco_routing()
    )
    assert not list(tmp_path.rglob("batch_*.json"))


def test_launcher_command_runs_in_the_route(entry, monkeypatch, capsys):
    monkeypatch.setattr(launch_host, "scheduler_status", lambda _: {"instance_id": "cpu-pool"})
    monkeypatch.setattr(launch_host, "ROOT", entry.root)
    launch_host.main(
        [
            "--method",
            "evotrace_v5",
            "--batch",
            "b",
            "--tasks",
            "tsp_construct",
            "--repeats",
            "1",
            "--scheduler-socket",
            "/fake",
            "--no-reflection",
        ]
    )
    command = json.loads(capsys.readouterr().out)["plan"][0]["command"]
    run.main([*command[command.index("experiments.traceaad.run") + 1 :], "--dry-run"])
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["method"] == "evotrace_v5" and metadata["reflection"] is False


# --- search, resume, held-out ----------------------------------------------------------------------------


def test_run_with_reflection_off_never_asks_for_reflection_and_resumes_with_saved_conditions(
    entry, capsys
):
    common = [*ROUTE, "--method", "evotrace.v5", "--initial-size", "2", "--skip-heldout"]
    run.main([*common, "--budget", "6", "--no-reflection"])
    first = last_json(capsys)
    directory = run_dir(entry.root)
    assert first["evaluations"] == 6 and entry.models[0].others == 0
    public = json.loads((directory / "run_config.json").read_text())
    native = json.loads((directory / "native" / "run_config.json").read_text())
    assert public["reflection"] is native["reflection"] is False
    assert public["include_experiences"] is native["include_experiences"] is False
    assert native["config"]["evaluation_budget"] == 6
    assert json.loads((directory / "summary.json").read_text())["status"] == "finished"

    # Omitting --no-reflection on resume keeps the original condition; a longer budget is the one allowed change.
    run.main([*common, "--budget", "9", "--resume"])
    assert last_json(capsys)["evaluations"] == 9
    (_, kwargs) = entry.constructed[-1]
    assert (
        kwargs["resume"] is True
        and kwargs["reflection"] is False
        and kwargs["include_experiences"] is False
    )
    assert entry.models[-1].others == 0
    public = json.loads((directory / "run_config.json").read_text())
    assert public["budget"] == 9 and public["budget_extended_from"] == 6
    assert public["method_params"] == asdict(Config(evaluation_budget=9, initial_size=2))
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "finished" and summary["budget_used"] == 9
    # A bare resume must not fall back to 1,000 evaluations or the default seed settings.
    run.main([*ROUTE, "--method", "evotrace.v5", "--resume", "--skip-heldout"])
    assert last_json(capsys)["evaluations"] == 9
    assert entry.constructed[-1][1]["config"].initial_size == 2



@pytest.mark.parametrize(
    "flags",
    [
        ["--reflection"],
        ["--initial-size", "3"],
        ["--seed", "1"],
        ["--max-tokens", "100"],
        ["--model-timeout", "5"],
        ["--seconds-per-instance", "5"],
    ],
)
def test_resume_rejects_changed_conditions_before_any_request(entry, capsys, flags):
    common = [*ROUTE, "--method", "evotrace.v5", "--initial-size", "2", "--skip-heldout"]
    run.main([*common, "--budget", "4", "--no-reflection"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        run.main([*common, "--budget", "8", "--resume", *flags])
    assert error.value.code == 2
    assert entry.models[-1].generations == 0 and len(entry.constructed) == 1


def test_resume_without_a_saved_run_is_refused(entry):
    with pytest.raises(SystemExit) as error:
        run.main([*ROUTE, "--method", "evotrace.v5", "--resume"])
    assert error.value.code == 2
    assert not run_dir(entry.root).exists()


def test_existing_run_is_not_overwritten(entry):
    arguments = [
        *ROUTE,
        "--method",
        "evotrace.v5",
        "--initial-size",
        "2",
        "--budget",
        "2",
        "--skip-heldout",
    ]
    run.main(arguments)
    with pytest.raises(FileExistsError):
        run.main(arguments)


def fake_heldout(monkeypatch, evaluators):
    """Offline held-out splits and a recording evaluator around the real V5 evaluator."""

    class Benchmark:
        task_description = "Minimize"
        template_program = "def solve(x):\n    pass\n"
        timeout_seconds = 5
        n_instance = 2

        @staticmethod
        def evaluate_program(source, fn):
            return fn(1)

    class Recording:
        def __init__(self, task, **options):
            self.task, self.options, self.inner = task, options, Evaluator(task, **options)
            self.evaluated, self.accessed = [], []
            evaluators.append(self)

        def evaluate(self, code):
            self.evaluated.append(code)
            return self.inner.evaluate(code)

        def __getattr__(self, name):
            self.accessed.append(name)
            raise AttributeError(name)

    monkeypatch.setattr(run, "PRIMARY_SPLITS", {"tsp_construct": ["split_a", "split_b"]})
    monkeypatch.setattr(run, "heldout_task", lambda name, split, workers=1: Benchmark())
    monkeypatch.setattr(run, "scale_of_split", lambda name, split: split)
    monkeypatch.setattr(run, "EvoTraceV5Evaluator", Recording)


def search_files(directory):
    return {
        str(p.relative_to(directory)): sha256(p.read_bytes()).hexdigest()
        for p in sorted((directory / "native").rglob("*"))
        if p.is_file()
    }


def test_heldout_scores_only_the_frozen_target_and_adds_no_search_evidence(
    entry, monkeypatch, capsys
):
    evaluators = []
    fake_heldout(monkeypatch, evaluators)
    run.main(
        [
            *ROUTE,
            "--method",
            "evotrace.v5",
            "--initial-size",
            "2",
            "--budget",
            "4",
            "--skip-heldout",
        ]
    )
    directory = run_dir(entry.root)
    result = json.loads((directory / "native" / "result.json").read_text())
    best = evotrace_v5.models.Program.from_dict(result["best"])
    before = search_files(directory)
    evaluators.clear()  # the search used the recording evaluator too
    rows = run.evaluate_heldout(
        directory,
        "tsp_construct",
        best,
        None,
        evaluation_seed=730241,
        seconds_per_instance=10.0,
        condition="evotrace_v5",
    )
    assert [row["split"] for row in rows] == ["split_a", "split_b"]
    assert {row["condition"] for row in rows} == {"evotrace_v5"}
    assert {row["node_id"] for row in rows} == {best.id}
    assert {row["source_sha256"] for row in rows} == {sha256(best.code.encode()).hexdigest()}
    assert all(row["status"] == "success" and row["timeout_scope"] == "whole_task" for row in rows)
    assert len(evaluators) == 2
    for evaluator in evaluators:
        assert evaluator.evaluated == [best.code]
        assert evaluator.options == {}
        assert isinstance(evaluator.task, Task)
        assert set(evaluator.accessed) <= {"evaluation_context", "workers", "last_dispatch"}
    assert search_files(directory) == before
    # A second call returns the stored rows without evaluating again.
    assert (
        run.evaluate_heldout(
            directory,
            "tsp_construct",
            best,
            None,
            evaluation_seed=730241,
            seconds_per_instance=10.0,
            condition="evotrace_v5",
        )
        == rows
    )
    assert len(evaluators) == 2
    with pytest.raises(ValueError, match="different program or evaluation scope"):
        run.evaluate_heldout(
            directory,
            "tsp_construct",
            best,
            None,
            evaluation_seed=1,
            seconds_per_instance=10.0,
            condition="evotrace_v5",
        )


def test_full_route_runs_search_then_heldout(entry, monkeypatch, capsys):
    evaluators = []
    fake_heldout(monkeypatch, evaluators)
    run.main([*ROUTE, "--method", "evotrace.v5", "--initial-size", "2", "--budget", "4"])
    directory = run_dir(entry.root)
    rows = json.loads((directory / "heldout.json").read_text())
    assert [row["condition"] for row in rows] == ["evotrace_v5"] * 2
    assert (
        sum(len(e.evaluated) for e in evaluators) == 4 + 2
    )  # four search evaluations, one per split
    assert last_json(capsys)["evaluations"] == 4


# --- standalone command line -----------------------------------------------------------------------------


class Endpoint:
    """A local OpenAI-compatible server for generation and reflection requests."""

    def __init__(self):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                content = reply(
                    payload["messages"][0]["content"],
                    payload.get("response_format"),
                    len(outer.requests),
                )
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"total_tokens": 9},
                    }
                ).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


@pytest.fixture
def sample_task(tmp_path, monkeypatch):
    (tmp_path / "sample_task.py").write_text("""from evotrace_v5 import Task

def make_task(scale=1):
    return Task("Minimize", "def solve(x):\\n    pass\\n", lambda src, fn: scale * fn(1), 2,
                example_code="def solve(x):\\n    return 100\\n", example_design="Constant 100")

class Benchmark:
    task_description = "Minimize"
    template_program = "def solve(x):\\n    pass\\n"
    timeout_seconds = 2

    @staticmethod
    def evaluate_program(source, fn):
        return fn(1)

def make_evaluator():
    return Benchmark()
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def cli_args(endpoint, directory, *extra, factory="sample_task:make_task"):
    return [
        "--task-factory",
        factory,
        "--run-dir",
        str(directory),
        "--base-url",
        endpoint.url,
        "--model",
        "local-test",
        *extra,
    ]


def test_cli_runs_a_custom_task_against_a_local_server_and_persists_its_settings(
    sample_task, capsys
):
    directory = sample_task / "run"
    with Endpoint() as endpoint:
        code = cli.main(
            cli_args(
                endpoint,
                directory,
                "--budget",
                "3",
                "--initial-size",
                "2",
                "--seed",
                "4",
                "--temperature",
                "0.7",
                "--task-options",
                '{"scale": 3}',
            )
        )
        assert code == 0 and endpoint.requests
        result = json.loads(capsys.readouterr().out)
        assert result["evaluations"] == 3 and result["status"] == "finished"
        assert (
            endpoint.requests[0]["temperature"] == 0.7
            and endpoint.requests[0]["model"] == "local-test"
        )
        saved = json.loads((directory / "run_config.json").read_text())
        assert saved["config"]["seed"] == 4 and saved["config"]["evaluation_budget"] == 3
        assert saved["reflection"] is True and saved["model"] == "local-test"
        assert (directory / "result.json").exists()
        # The saved settings carry over; only the budget moves, and later resumes keep the longer one.
        used = len(endpoint.requests)
        assert cli.main(cli_args(endpoint, directory, "--resume", "--budget", "4")) == 0
        assert json.loads(capsys.readouterr().out)["evaluations"] == 4
        assert (
            json.loads((directory / "run_config.json").read_text())["config"]["evaluation_budget"]
            == 4
        )
        assert len(endpoint.requests) > used
        used = len(endpoint.requests)
        assert cli.main(cli_args(endpoint, directory, "--resume")) == 0
        assert (
            json.loads(capsys.readouterr().out)["evaluations"] == 4
            and len(endpoint.requests) == used
        )
        assert {request["temperature"] for request in endpoint.requests} == {0.7}


def test_cli_accepts_a_plain_evaluator_from_the_factory(sample_task, capsys):
    with Endpoint() as endpoint:
        assert (
            cli.main(
                cli_args(
                    endpoint,
                    sample_task / "evaluator",
                    "--budget",
                    "2",
                    "--initial-size",
                    "2",
                    factory="sample_task:make_evaluator",
                )
            )
            == 0
        )
    assert json.loads(capsys.readouterr().out)["evaluations"] == 2


def test_cli_factory_must_name_module_and_function():
    with pytest.raises(ValueError, match="module:function"):
        cli.load_task(factory="sample_task")


def test_cli_passes_only_the_v5_arguments_and_keeps_saved_choices_on_resume(
    sample_task, monkeypatch, capsys
):
    constructed = []

    class Recording(EvoTrace):
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "EvoTrace", Recording)
    directory = sample_task / "run"
    with Endpoint() as endpoint:
        cli.main(
            cli_args(
                endpoint,
                directory,
                "--budget",
                "2",
                "--initial-size",
                "2",
                "--no-reflection",
                "--no-example",
            )
        )
        cli.main(cli_args(endpoint, directory, "--resume"))
        assert {
            tuple(r["response_format"]["json_schema"]["schema"]["required"])
            for r in endpoint.requests
        } == {("analysis", "code", "design")}
    assert [set(kwargs) for _, kwargs in constructed] == [CONSTRUCTOR_KEYS] * 2
    for index, (args, kwargs) in enumerate(constructed):
        task = args[0]
        assert task.example_code is None and task.example_design is None
        assert kwargs["reflection"] is False and kwargs["include_experiences"] is False
        assert kwargs["resume"] is bool(index)
    assert constructed[1][1]["config"] == constructed[0][1]["config"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--seed", "9"],
        ["--initial-size", "3"],
        ["--temperature", "0.5"],
        ["--max-tokens", "10"],
        ["--model-timeout", "9"],
        ["--model", "other"],
        ["--reflection"],
    ],
)
def test_cli_resume_refuses_changed_conditions_without_contacting_the_model(
    sample_task, flags, capsys
):
    directory = sample_task / "run"
    with Endpoint() as endpoint:
        cli.main(
            cli_args(endpoint, directory, "--budget", "2", "--initial-size", "2", "--no-reflection")
        )
        used = len(endpoint.requests)
        arguments = cli_args(endpoint, directory, "--resume", "--budget", "4", *flags)
        if flags == ["--model", "other"]:
            arguments = [a for a in arguments if a not in ("--model", "local-test")]
        with pytest.raises(SystemExit) as error:
            cli.main(arguments)
        assert error.value.code == 2 and len(endpoint.requests) == used
    assert (
        json.loads((directory / "run_config.json").read_text())["config"]["evaluation_budget"] == 2
    )


def test_cli_refuses_to_overwrite_or_resume_nothing(sample_task):
    with Endpoint() as endpoint:
        directory = sample_task / "run"
        with pytest.raises(SystemExit):
            cli.main(cli_args(endpoint, directory, "--resume"))
        cli.main(cli_args(endpoint, directory, "--budget", "2", "--initial-size", "2"))
        with pytest.raises(SystemExit):
            cli.main(cli_args(endpoint, directory))


def test_cli_registry_task_uses_the_experiment_condition_and_score_text(monkeypatch):
    seen = {}

    class Benchmark:
        task_description = "Tour"
        template_program = "def solve(x):\n    pass\n"
        timeout_seconds = 3

        @staticmethod
        def evaluate_program(source, fn):
            return 1.0

    registry = types.SimpleNamespace(
        training_task=lambda name, condition: (
            seen.update(name=name, condition=condition) or (Benchmark(), {"timeout_seconds": 3})
        )
    )
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: registry)
    task = cli.load_task(name="tsp_construct")
    assert seen == {"name": "tsp_construct", "condition": "traceaad"}
    assert task.score_definition == "Mean tour length; lower is better."
    assert task.metadata["name"] == "tsp_construct" and task.metadata["settings"] == {
        "timeout_seconds": 3
    }


def test_package_command_runs_alone_and_resumes(sample_task):
    package = Path(importlib.util.find_spec("evotrace_v5").origin).parent
    shutil.copytree(
        package,
        sample_task / "evotrace_v5",
        ignore=shutil.ignore_patterns("__pycache__"),
        dirs_exist_ok=True,
    )
    environment = {**os.environ, "PYTHONPATH": str(sample_task)}
    with Endpoint() as endpoint:
        command = [
            sys.executable,
            "-S",
            "-m",
            "evotrace_v5",
            *cli_args(endpoint, "cli_run", "--initial-size", "2"),
        ]
        for extra in (["--budget", "2"], ["--resume", "--budget", "3"]):
            done = subprocess.run(
                command + extra,
                cwd=sample_task,
                env=environment,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            assert done.returncode == 0, done.stdout + done.stderr
        assert json.loads(done.stdout)["evaluations"] == 3
    assert endpoint.requests[0]["response_format"]["json_schema"]["schema"]["required"] == [
        "analysis",
        "code",
        "design",
    ]
