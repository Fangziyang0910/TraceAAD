"""V21 material selection, with conditional experience used within the operator's task."""

from traceaad_v21.context import CROSSOVER as V21_CROSSOVER, REFINE as V21_REFINE, context as v21_context

from .notes import experience_block

REFINE = (
    "Find one limitation in what the current code computes and change the computation that addresses it, "
    "keeping its main mechanism. The change may add, replace, or remove a computation. "
    "Avoid repeating the same evaluated implementation from the same start under the same key conditions. "
    "An idea may be retried when its limiting conditions, implementation, or supporting computations change. "
    "In analysis, first identify the current computational problem and the change that addresses it. "
    "If using experience, explain how its conditions match this code or the computation you will introduce. "
    "Base changes to the main mechanism on the current code and relevant trajectory, not merely on "
    "comparisons from other computational structures."
)
CROSSOVER = (
    "Adapt one useful computation from the reference into the current method, adjusting representations as needed. "
    "If no coherent adaptation exists, say so and implement your best candidate. "
    "In analysis, first name the reference computation and the current decision it will change. "
    "Use applicable experience to address representation, scale, where the computation acts, or operation order. "
    "If using experience, relate its conditions to the current code or the computation you will introduce."
)


def context(archive, parent, operator, reference, config, notes):
    text, instruction, view = v21_context(archive, parent, operator, reference, config)
    if operator == "Explore" or not notes:
        return text, instruction, view
    original, revised = (V21_REFINE, REFINE) if operator == "Refine" else (V21_CROSSOVER, CROSSOVER)
    instruction = instruction.replace(original, revised)
    text = experience_block(notes) + "\n\n" + text
    return text, instruction, view
