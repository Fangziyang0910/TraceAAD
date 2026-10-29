"""Ordinary code-only generation with selected-path and local-retry feedback."""

import ast
import hashlib
import re

from traceaad.delivery import DeliveryError, signature, target
from traceaad.history import code_diff, path
from traceaad.prompts import corrected_description, short_error, SCORE_MEANINGS


PROMPT_REVISION = "v18-context-2"
OUTPUT = (
    "Return only the complete, self-contained Python source, including imports and helpers. "
    "Use the required function signature. Write no Markdown fences, explanations, comments or docstrings."
)


def normalized_source(code):
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_code(text, finish_reason):
    if not isinstance(text, str) or finish_reason != "stop":
        raise DeliveryError(f"incomplete or non-text completion: {finish_reason}")
    code = normalized_source(text)
    opening = re.match(r"\A```(?:python|py)?[ \t]*\n", code)
    if opening:
        code = code[opening.end():]
        closing = re.search(r"\n[ \t]*```\Z", code)
        if closing:
            code = code[:closing.start()]
        # Normally finished replies may omit a closing fence; source/interface checks still apply.
        code = normalized_source(code)
    if not code:
        raise DeliveryError("no usable code")
    return code + "\n"


class PromptTooLarge(ValueError):
    pass


class PromptBuilderV18:
    def __init__(self, llm, task, training, programs, states, config):
        self.llm, self.programs, self.states, self.config = llm, programs, states, config
        evaluation = training.evaluation
        function = target(str(evaluation.template_program))
        returns = f" -> {ast.unparse(function.returns)}" if function.returns else ""
        contract = ast.get_docstring(function) or ""
        description = corrected_description(task, evaluation.task_description.strip())
        for note in (
            "The supplied DSATUR construction is editable starting code.",
            "The supplied Giffler-Thompson/MWKR construction is editable starting code.",
        ):
            description = description.replace(note, "").strip()
        meaning = SCORE_MEANINGS.get(task, getattr(evaluation, "score_meaning", "the task objective"))
        direction = "Lower" if config.objective == "minimize" else "Higher"
        self.common = (
            f"[Task]\n{description}\n\n"
            f"[Function Contract]\ndef {signature(function)}{returns}:\n{contract}\n"
            "Keep the function name, arguments and return contract. Include all imports, constants and helpers.\n"
            f"Score: {meaning}. {direction} is better."
        )

    @staticmethod
    def score(program):
        return program.request.get("raw_score", program.fitness)

    def outcome(self, parent, child):
        if not child.valid:
            return f"{child.status}: {short_error((child.failure or {}).get('error'))}"
        if normalized_source(parent.code) == normalized_source(child.code):
            change = "unchanged source; exact copy of the current program"
        else:
            gain = parent.fitness - child.fitness
            change = "improved" if gain > 0 else "worse" if gain < 0 else "same score, different source"
        return f"score {self.score(child):.12g}; {change}"

    def build(self, action, parent=None, state=None):
        history, attempts, total_attempts = [], [], 0
        if state is not None:
            sequence = path(parent, self.programs)
            for before, after in list(zip(sequence, sequence[1:]))[-self.config.history_depth:]:
                delta = code_diff(ast.unparse(ast.parse(before.code)), ast.unparse(ast.parse(after.code)))
                history.append((after.id, before.id,
                    f"Program {before.id} → {after.id}: {self.score(before):.12g} → {self.outcome(before, after)}\n"
                    + (f"```diff\n{delta}\n```" if delta else "No change after normalizing comments and formatting.")))
            total_attempts = state.attempt_count
            previous = state.retry_from
            while previous is not None and len(attempts) < self.config.attempts_shown:
                prior = self.states[previous]
                result = self.programs[prior.result_id]
                attempts.append((result.id, f"Attempt {prior.attempt_count + 1}: {self.outcome(parent, result)}"))
                previous = prior.retry_from
            attempts.reverse()
        if action == "Init":
            instruction = (
                "Design an algorithm whose decisions address the stated objective and use the available inputs. "
                "Implement a coherent solution procedure that satisfies the full return contract."
            )
        elif action == "Diverge":
            instruction = (
                "Create an alternative starting algorithm for the same task. Use the reference to identify "
                "its main decision computation, then implement a different way to make those decisions. "
                "A parameter adjustment or a renamed copy of the reference is insufficient. "
                "Keep the interface and useful supporting computations."
            )
        else:
            instruction = (
                "Improve the current program below. Identify a weakness in how it makes decisions, "
                "and implement one coherent change that could improve the objective. "
                "Develop its useful computations and keep unrelated parts intact. "
                "Return a changed implementation; copying the current source or changing only its presentation "
                "does not advance this search. "
                f"This is attempt {total_attempts + 1} from this program on the selected continuation path."
            )
        while True:
            sections = [self.common]
            if history:
                sections.append(
                    "[Selected Development Path]\nActual ancestors of the current program, oldest first. "
                    "Diffs normalize comments and formatting. Scores evaluate complete versions; "
                    "a score change does not isolate the effect of each edited line.\n"
                    + "\n\n".join(h[2] for h in history))
            if attempts:
                sections.append(
                    "[Previous Attempts at This Position]\nThese children were generated from the current program "
                    "but were not adopted on this continuation path. Their implementations are not shown. "
                    "Each score or error belongs to that child; the current program remains valid.\n"
                    + "\n".join(a[1] for a in attempts))
            if parent is not None:
                label = "Reference" if action == "Diverge" else "Current"
                sections.append(f"[{label} Program]\nScore: {self.score(parent):.12g}.\n```python\n{parent.code.rstrip()}\n```")
            sections.append("[Request]\n" + instruction)
            sections.append("[Output]\n" + OUTPUT)
            prompt = "\n\n".join(sections)
            tokens = self.llm.count_prompt_tokens(prompt)
            if tokens + self.config.output_tokens <= self.config.context_tokens:
                break
            if attempts:
                attempts.pop(0)
            elif history:
                history.pop(0)
            else:
                raise PromptTooLarge("task, current program and output reserve exceed context_tokens")
        return {
            "action": action,
            "prompt_revision": PROMPT_REVISION,
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": tokens,
            "prompt_program_ids": list(dict.fromkeys(
                ([parent.id] if parent else []) + [i for h in history for i in (h[1], h[0])]
            )),
            "history_edge_ids": [h[0] for h in history],
            "local_attempt_ids": [a[0] for a in attempts],
            "local_attempt_count": total_attempts,
        }
