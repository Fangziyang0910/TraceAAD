from __future__ import annotations

import copy
import random

from .graph import PathWiseAction, PathWiseNode
from core import Function


class PathWisePrompt:
    POLICY_PHRASES = [
        "Favor unusual parent combinations.",
        "Explore less common heuristic structures.",
        "Prioritize novelty in parent selection.",
        "Choose parents that differ in style.",
        "Invent a directive that departs from usual conventions.",
    ]

    WORLD_MODEL_PHRASES = [
        "Introduce a novel program segment that changes the decision logic.",
        "Modify the structure of the algorithm rather than only tuning constants.",
        "Create a new mechanism that has not appeared in the parent code.",
        "Try a different control flow while keeping the required signature.",
        "Construct a fresh rule that improves the existing methods.",
    ]

    def __init__(self, task_prompt: str, template_function: Function):
        self.task_prompt = task_prompt
        self.template_function = copy.deepcopy(template_function)
        self.func_name = self.template_function.name
        self.func_desc = self._function_description()

    def _function_description(self) -> str:
        docstring = self.template_function.docstring or ""
        docstring = docstring.replace('"""', "").strip()
        if docstring:
            return docstring
        return f"The `{self.func_name}` function must keep the given signature and solve the task."

    def function_signature(self, version: int | str = 2) -> str:
        ret = f" -> {self.template_function.return_type}" if self.template_function.return_type else ""
        return f"def {self.func_name}_v{version}({self.template_function.args}){ret}:"

    def blank_function(self, version: int | str = 2) -> str:
        return f"{self.function_signature(version)}\n    pass"

    @staticmethod
    def perturbation_phrase(phrases: list[str], prob: float) -> str:
        if phrases and random.random() < prob:
            return random.choice(phrases)
        return ""

    @staticmethod
    def _node_code(node: PathWiseNode) -> str:
        func = copy.deepcopy(node.function)
        func.docstring = ""
        return str(func).strip()

    @classmethod
    def format_state(cls, nodes: list[PathWiseNode], *, shuffle: bool = True) -> str:
        state = list(nodes)
        if shuffle:
            random.shuffle(state)
        blocks = []
        for node in state:
            parent_lines = []
            for parent in node.parents:
                parent_lines.append(f"- {parent.node_id} (score: {parent.score:.6f}): {parent.description}")
            parent_text = "\n".join(parent_lines) if parent_lines else "None"
            blocks.append(
                f"ID: {node.node_id}\n"
                f"Description: {node.description}\n"
                f"Derivation rationale: {node.rationale}\n"
                f"Parent metadata:\n{parent_text}\n"
                f"Score: {node.score:.6f}\n"
                f"Code:\n{cls._node_code(node)}"
            )
        return "\n\n".join(blocks) if blocks else "Empty state"

    def initialization_prompt(self, index: int, external_knowledge: str = "") -> str:
        return f"""You are a PathWise initialization agent for automated algorithm design.
Your task is to create one diverse heuristic candidate for the following task.

Task:
{self.task_prompt}

Function description:
{self.func_desc}

Implement this function body using the required signature:
{self.blank_function(2)}

{external_knowledge}

Output exactly:
Description: <under 30 words>
Derivation Rationale: <one concise sentence>
```python
<complete function>
```"""

    def policy_prompt(
            self,
            state: list[PathWiseNode],
            policy_reflection: str,
            perturbation: str = "",
    ) -> str:
        reflection = policy_reflection.strip() if policy_reflection else "None"
        nudge = f"\nExploration nudge: {perturbation}" if perturbation else ""
        available = ", ".join(node.node_id for node in state)
        return f"""You are a PathWise policy agent. Choose parent heuristic ID(s) and write a directive for deriving one better child heuristic.

Task:
{self.task_prompt}

Function description:
{self.func_desc}

Current entailment state:
{self.format_state(state)}

Policy reflection:
{reflection}{nudge}

Available parent IDs: {available}
Use only available IDs. Higher score is better.

Output exactly:
PARENTS: [id_1, id_2]
DIRECTIVE: <how the world model should transform or combine the selected parents>"""

    def world_model_prompt(
            self,
            action: PathWiseAction,
            parent_nodes: list[PathWiseNode],
            world_model_reflection: str,
            perturbation: str = "",
    ) -> str:
        parents = "\n\n".join(
            f"Parent ID: {node.node_id}\n"
            f"Description: {node.description}\n"
            f"Score: {node.score:.6f}\n"
            f"Code:\n{self._node_code(node)}"
            for node in parent_nodes
        )
        reflection = world_model_reflection.strip() if world_model_reflection else "None"
        nudge = f"\nExploration nudge: {perturbation}" if perturbation else ""
        return f"""You are a PathWise world model agent. Execute the policy directive by generating one improved heuristic.

Task:
{self.task_prompt}

Function description:
{self.func_desc}

Selected parents:
{parents}

Directive:
{action.rationale}

World-model reflection:
{reflection}{nudge}

The new function must keep the same inputs and outputs. Higher score is better.
Implement the complete function using this signature:
{self.blank_function(2)}

Output exactly:
Description: <under 30 words>
```python
<complete function>
```"""

    def policy_critic_prompt(
            self,
            state: list[PathWiseNode],
            actions: list[PathWiseAction],
            action_summaries: list[str],
    ) -> str:
        action_text = "\n\n".join(
            f"Action {idx}\nParents: {action.parents}\nDirective: {action.rationale}\n{summary}"
            for idx, (action, summary) in enumerate(zip(actions, action_summaries))
        )
        return f"""You are a PathWise policy critic. Reflect on which parent selections and directives were effective.

Task:
{self.task_prompt}

Current state:
{self.format_state(state, shuffle=False)}

Action rollout results:
{action_text}

Higher score is better. Write concise guidance for the next policy step in under 80 words."""

    def world_model_critic_prompt(
            self,
            best: PathWiseNode,
            worst: PathWiseNode,
    ) -> str:
        return f"""You are a PathWise world model critic. Compare the best and worst generated heuristics and give code-generation guidance.

Task:
{self.task_prompt}

Best rollout:
Description: {best.description}
Score: {best.score:.6f}
Code:
{self._node_code(best)}

Worst rollout:
Description: {worst.description}
Score: {worst.score:.6f}
Code:
{self._node_code(worst)}

Higher score is better. Write concise guidance for the next world-model rollout in under 80 words."""
