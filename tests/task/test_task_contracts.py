"""Static contract checks for the five frozen main-experiment tasks.

Guards against empty task descriptions, unparseable templates, missing
Evaluation wiring, and obvious signature drift between template and evaluator.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import warnings
from pathlib import Path

import pytest

from llm4ad.base.code import TextFunctionProgramConverter
from llm4ad.base.evaluate import Evaluation

ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "benchmarks"
FROZEN_TASKS = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
    "vrptw_construct",
)


def _all_task_dirs() -> list[Path]:
    return [TASK_ROOT / relative for relative in FROZEN_TASKS]


def _module_path(py_file: Path) -> str:
    return ".".join(py_file.relative_to(ROOT).with_suffix("").parts)


def _eval_str_assign(node: ast.AST) -> str:
    """Evaluate a string assignment value, including ``\"...\".strip()`` forms."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "strip":
        base = _eval_str_assign(node.func.value)
        return base.strip()
    if isinstance(node, ast.JoinedStr):
        # f-strings are unexpected; fall through to literal_eval error
        pass
    return ast.literal_eval(node)


def _load_template_vars(template_path: Path) -> tuple[str, str]:
    src = template_path.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        tree = ast.parse(src, filename=str(template_path))
    task_description = None
    template_program = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "task_description":
                task_description = _eval_str_assign(node.value)
            elif target.id == "template_program":
                template_program = _eval_str_assign(node.value)
    assert isinstance(task_description, str), f"{template_path}: task_description missing"
    assert isinstance(template_program, str), f"{template_path}: template_program missing"
    return task_description, template_program


def _template_function(template_program: str) -> ast.FunctionDef:
    tree = ast.parse(template_program)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(funcs) == 1, f"expected exactly one top-level function, got {len(funcs)}"
    return funcs[0]


@pytest.mark.parametrize("task_dir", _all_task_dirs(), ids=lambda p: str(p.relative_to(TASK_ROOT)))
def test_task_contract_basics(task_dir: Path):
    template_path = task_dir / "template.py"
    evaluation_path = task_dir / "evaluation.py"
    assert evaluation_path.exists(), f"missing evaluation.py for {task_dir}"

    task_description, template_program = _load_template_vars(template_path)
    assert task_description.strip(), f"{task_dir}: empty task_description"
    assert "Input kwargs" not in template_program
    assert "Input (via kwargs)" not in template_program
    assert "passed as keyword arguments" not in template_program

    # Template program must parse and expose exactly one evolvable function.
    func = _template_function(template_program)
    docstring = (ast.get_docstring(func) or "").lower()
    assert "return" in docstring or "output" in docstring, (
        f"{task_dir}: function description does not explain its output"
    )
    for arg in [*func.args.posonlyargs, *func.args.args]:
        if arg.arg != "self":
            assert arg.arg.lower() in docstring, (
                f"{task_dir}: function description omits input {arg.arg}"
            )
    program = TextFunctionProgramConverter.text_to_program(template_program)
    assert program is not None
    assert len(program.functions) == 1
    assert program.functions[0].name == func.name

    # The exact template shown to the LLM must define its function in the
    # repository environment. Parsing alone misses unresolved annotations and
    # unnecessary imports of unavailable packages.
    namespace: dict[str, object] = {}
    exec(
        compile(template_program, str(template_path), "exec", dont_inherit=True),
        namespace,
        namespace,
    )
    assert callable(namespace.get(func.name)), (
        f"{task_dir}: template does not define callable {func.name}"
    )

    # Evaluation must import and pass the same task_description.
    eval_src = evaluation_path.read_text(encoding="utf-8")
    assert "task_description" in eval_src
    assert re.search(r"super\(\).__init__\([\s\S]*?task_description\s*=", eval_src), (
        f"{task_dir}: Evaluation.__init__ does not pass task_description"
    )

    # Import evaluation module and confirm Evaluation subclass wires description.
    mod = importlib.import_module(_module_path(evaluation_path))
    classes = [
        obj
        for _, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, Evaluation) and obj is not Evaluation and obj.__module__ == mod.__name__
    ]
    assert classes, f"{task_dir}: no Evaluation subclass found"
    init_src = inspect.getsource(classes[0].__init__)
    assert "task_description" in init_src

    # Placeholder body must not reference undefined kwargs when signature has named args only.
    has_varkw = func.args.kwarg is not None
    if not has_varkw:
        body_src = ast.unparse(func)
        assert "kwargs[" not in body_src, (
            f"{task_dir}: template body uses kwargs but signature has no **kwargs"
        )


def test_direct_evaluator_calls_match_template_arity():
    """Check every statically visible call from an evaluator to its candidate."""
    mismatches = []

    for task_dir in _all_task_dirs():
        _, template_program = _load_template_vars(task_dir / "template.py")
        func = _template_function(template_program)
        positional = [*func.args.posonlyargs, *func.args.args]
        min_args = len(positional) - len(func.args.defaults)
        max_args = float("inf") if func.args.vararg else len(positional)

        eval_tree = ast.parse((task_dir / "evaluation.py").read_text(encoding="utf-8"))
        calls = []
        for evaluator_func in (
            n
            for n in ast.walk(eval_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            parameter_names = {
                a.arg
                for a in [*evaluator_func.args.posonlyargs, *evaluator_func.args.args]
            }
            for call in ast.walk(evaluator_func):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in parameter_names
                ):
                    calls.append(call)

        relative = str(task_dir.relative_to(TASK_ROOT))
        for call in calls:
            if not min_args <= len(call.args) <= max_args:
                mismatches.append(
                    f"{relative}:{call.lineno} passes {len(call.args)} positional "
                    f"arguments; template accepts {min_args}..{max_args}"
                )

    assert mismatches == []
