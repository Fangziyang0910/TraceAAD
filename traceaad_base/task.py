"""Task interface; benchmark code is supplied by the caller, not imported here."""

import ast
import copy
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Task:
    task_description: str
    template_program: str
    evaluate_program: Callable
    time_limit: float
    entrypoint: str | None = None
    example_code: str | None = None
    example_design: str | None = None
    score_definition: str = "Task objective; lower is better."
    evaluation_seed: int = 730241
    executes_source: bool = False
    metadata: dict = field(default_factory=dict)
    memory_limit_bytes: int | None = None

    def __post_init__(self):
        module = ast.parse(self.template_program)
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        if self.entrypoint is None:
            self.entrypoint = functions[0].name
        self._target = next(node for node in functions if node.name == self.entrypoint)
        interface = copy.deepcopy(self._target)
        docstring = ast.get_docstring(interface, clean=False)
        interface.body = [ast.Expr(ast.Constant(docstring))] if docstring is not None else []
        interface.body.append(ast.Pass())
        skeleton = ast.Module(
            body=[copy.deepcopy(n) for n in module.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            + [interface],
            type_ignores=[],
        )
        self.function_template = ast.unparse(ast.fix_missing_locations(skeleton))

    @classmethod
    def from_evaluator(cls, evaluator, *, metadata=None, use_template_example=True) -> "Task":
        """Adapt a task with template_program, task_description and evaluate_program."""
        info = dict(metadata or {})
        for name in ("split", "seed", "aco_seed", "n_instance", "problem_size", "n_ants",
                     "n_iterations", "n_workers", "outer_settings", "interface"):
            if hasattr(evaluator, name):
                info.setdefault(name, getattr(evaluator, name))
        info.setdefault("evaluator", f"{type(evaluator).__module__}.{type(evaluator).__qualname__}")
        return cls(
            task_description=evaluator.task_description,
            template_program=str(evaluator.template_program),
            evaluate_program=evaluator.evaluate_program,
            time_limit=evaluator.timeout_seconds,
            entrypoint=getattr(evaluator, "FUNCTION_NAME", None),
            example_code=str(evaluator.template_program) if use_template_example else None,
            score_definition=getattr(evaluator, "score_meaning", "Task objective; lower is better."),
            evaluation_seed=getattr(evaluator, "evaluation_seed", 730241),
            executes_source=getattr(evaluator, "executes_source", False),
            metadata=info,
        )

    def validate_source(self, source: str) -> None:
        module = ast.parse(source, filename="<candidate>")
        targets = [n for n in module.body if isinstance(n, ast.FunctionDef)
                   and n.name == self.entrypoint]
        if len(targets) != 1:
            raise ValueError(f"Define the required function {self.entrypoint!r} exactly once")

        def arguments(node):
            args = copy.deepcopy(node.args)
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                arg.annotation = None
                arg.type_comment = None
            for arg in (args.vararg, args.kwarg):
                if arg is not None:
                    arg.annotation = None
                    arg.type_comment = None
            return ast.dump(args, include_attributes=False)

        if arguments(targets[0]) != arguments(self._target):
            raise ValueError(f"Preserve the parameters and defaults of {self.entrypoint!r}")

    def description(self) -> dict:
        return {
            "task_description": self.task_description,
            "template_program": self.template_program,
            "entrypoint": self.entrypoint,
            "example_code": self.example_code,
            "example_design": self.example_design,
            "score_definition": self.score_definition,
            "time_limit": self.time_limit,
            "evaluation_seed": self.evaluation_seed,
            "executes_source": self.executes_source,
            "metadata": self.metadata,
            "memory_limit_bytes": self.memory_limit_bytes,
        }
