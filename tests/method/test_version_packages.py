"""Version packages run and resume without repository implementation dependencies."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ("v14", "v15", "v16", "v17", "v18", "v18_1")


@pytest.mark.parametrize("version", VERSIONS)
def test_every_module_keeps_imports_inside_its_package(version):
    package = ROOT / "traceaad" / version
    repository_packages = {
        path.name for path in ROOT.iterdir() if path.is_dir() and (path / "__init__.py").exists()
    }
    for path in package.rglob("*.py"):
        depth = 1 + len(path.parent.relative_to(package).parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level <= depth, (path, node.lineno, "import leaves the package")
            elif isinstance(node, ast.ImportFrom | ast.Import):
                modules = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else [n.name for n in node.names]
                )
                assert all(
                    module.split(".", 1)[0] not in repository_packages for module in modules
                ), (path, node.lineno, modules)


def run_isolated(script, directory, version):
    result = subprocess.run(
        [sys.executable, "-c", script, version],
        cwd=directory,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(directory), str(ROOT)))},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("version", VERSIONS)
def test_copied_package_searches_and_resumes_with_all_traceaad_imports_blocked(tmp_path, version):
    shutil.copytree(
        ROOT / "traceaad" / version,
        tmp_path / "detached",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    run_isolated(
        r"""
import importlib.abc
import json
from pathlib import Path
import sys

class BlockRepositoryModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"traceaad", "core", "benchmarks", "experiments", "tests"}:
            raise AssertionError(f"detached package imported {fullname}")

sys.meta_path.insert(0, BlockRepositoryModules())
from detached import Config, Evaluation, Program, TraceAAD
from detached.delivery import DeliveryError

class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__("def score(x):\n    pass", "Return a numeric score.", safe_evaluate=False)

    def evaluate_program(self, source, function, **kwargs):
        return -function(1)

class TokenLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 16

    def draw_sample_with_details(self, prompt, **kwargs):
        return {"content": next(self.responses), "finish_reason": "stop", "usage": {}, "model": "test"}

def response(value):
    return json.dumps({"analysis": "", "design": f"return {value}",
                       "code": f"def score(x):\n    return {value}\n"})

version = sys.argv[1]
def replies(values):
    result = []
    for value in values:
        result.append(f"def score(x):\n    return {value}\n" if version == "v18" else response(value))
        if version == "v18_1":
            result.append(json.dumps({
                "relation": "new", "birth_id": None, "mechanism": f"rule {value}",
                "evidence": "Scripted rule.", "code_evidence": f"return {value}",
            }))
    return result

config = Config(budget=3, roots=1)
run_dir = Path("run")
first = TraceAAD(evaluation=TinyEvaluation(), llm=TokenLLM(*replies([1])),
                 run_dir=run_dir, config=config)
first.step() if version == "v18" else first._roots()
assert first.budget_used == 1
resumed = TraceAAD(evaluation=TinyEvaluation(), llm=TokenLLM(*replies([2, 3])),
                   run_dir=run_dir, config=config)
summary = resumed.run()
assert summary["status"] == "finished"
assert summary["budget_used"] == 3 and summary["best"]["fitness"] == -3
assert summary["model_calls"] == (6 if version == "v18_1" else 3)
assert all(isinstance(program, Program) for program in resumed.programs.values())
assert all(cls is object or cls.__module__.startswith("detached.") for cls in TraceAAD.__mro__)
assert Config.__bases__ == (object,)
assert DeliveryError.__module__ == "detached.delivery"
assert "return 3" in (run_dir / "best_program.py").read_text()
finished = TraceAAD(evaluation=TinyEvaluation(), llm=TokenLLM(), run_dir=run_dir, config=config)
assert finished.run()["model_calls"] == summary["model_calls"]

# An injected task can define its own invalid-result exception.
class InvalidEvaluationResult(Exception):
    pass

class RejectingTask(TinyEvaluation):
    def evaluate_program(self, source, function, **kwargs):
        raise InvalidEvaluationResult("task rejected the solution")

from detached.instance_evaluation import InstanceProgramEvaluator
rejected = InstanceProgramEvaluator(RejectingTask()).evaluate("def score(x): return 1", 1)
assert rejected["evaluations"][0]["failure_kind"] == "invalid_result"
assert not any(name.split(".", 1)[0] in {"traceaad", "core", "benchmarks", "experiments", "tests"}
               for name in sys.modules)
""",
        tmp_path,
        version,
    )


@pytest.mark.parametrize("version", VERSIONS)
def test_experiment_previews_load_only_the_selected_package(tmp_path, version):
    run_isolated(
        r"""
import contextlib
import importlib
import importlib.abc
import io
import json
import sys

version = sys.argv[1]
allowed = "traceaad." + version
class SelectedPackageOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "experiments.infra" or fullname.startswith("experiments.infra."):
            raise AssertionError(f"entry imported shared experiment helper {fullname}")
        if fullname.startswith("traceaad.") and not (fullname == allowed or fullname.startswith(allowed + ".")):
            raise AssertionError(f"entry imported unrelated module {fullname}")

sys.meta_path.insert(0, SelectedPackageOnly())
run = importlib.import_module(f"experiments.traceaad_{version}.run")
output = io.StringIO()
with contextlib.redirect_stdout(output):
    run.main(["--task", "tsp_construct", "--budget", "3", "--dry-run"])
assert json.loads(output.getvalue())["method"] == "traceaad_" + version
launch = importlib.import_module(f"experiments.traceaad_{version}.launch_host")
output = io.StringIO()
with contextlib.redirect_stdout(output):
    launch.main(["--batch", "isolated_preview", "--tasks", "tsp_construct", "--repeats", "1"])
manifest = json.loads(output.getvalue())
assert manifest["method"] == "traceaad_" + version and len(manifest["plan"]) == 1
assert all(name == "traceaad" or name == allowed or name.startswith(allowed + ".")
           for name in sys.modules if name == "traceaad" or name.startswith("traceaad."))
""",
        tmp_path,
        version,
    )
