"""Random archive-reference context for the V10.11 trajectory ablation."""

import math

from ..traceaad_v10_11.trajectory import OPERATOR_INSTRUCTIONS, TrajectoryBuilder

REFERENCE_TAU = 8.0


def rank_softmax_sample(nodes, k, rng, *, tau=REFERENCE_TAU):
    """Draw k nodes without replacement, favouring better fitness by rank softmax.

    Probability is proportional to exp(-rank / tau) over the fitness ranking, so
    every archived node has a chance while better programs are drawn more often.
    Ranking instead of raw fitness keeps the distribution scale-free across
    tasks; tau = 8 gives an effective support of about sixteen entries and is a
    transparent default, not claimed optimal.
    """
    ordered = sorted(nodes, key=lambda node: (-node.fitness, node.id))
    weights = [math.exp(-rank / tau) for rank in range(len(ordered))]
    chosen = []
    while len(chosen) < k and ordered:
        pick = rng.random() * sum(weights)
        index, cumulative = 0, 0.0
        while index < len(ordered) - 1:
            cumulative += weights[index]
            if pick < cumulative:
                break
            index += 1
        chosen.append(ordered.pop(index))
        weights.pop(index)
    return chosen


class RandomReferenceBuilder(TrajectoryBuilder):
    """Renders unordered archive reference cards in place of the formation path."""

    def __init__(self, llm, task_contract, *, max_tokens, n_references, lookup, all_nodes):
        super().__init__(llm, task_contract, max_tokens=max_tokens,
                         max_events=n_references, lookup=lookup, all_nodes=all_nodes)
        self.n_references = n_references

    def build(self, parent, operator, donor=None, references=()):
        references = sorted(references, key=lambda node: (-node.fitness, node.id))
        parts = [self.task_contract, "Fitness: higher is better."]
        if parent is not None:
            parts.append(self.program(parent, "Current Algorithm"))
            if references:
                cards = [
                    "# Archive Algorithms",
                    f"{len(references)} independently evaluated algorithms sampled from the search "
                    "archive, listed from best to worst measured fitness. They are separate designs "
                    "with no order or dependency between them.",
                ]
                for index, node in enumerate(references, 1):
                    card = [f"Reference {index} | Fitness: {node.fitness}"]
                    if node.idea:
                        card.append("Idea: " + " ".join(node.idea.split()))
                    cards.append("\n".join(card))
                parts.append("\n\n".join(cards))
        if donor is not None:
            parts.append(self.program(donor, "Reference Algorithm"))
        return self._complete(parts, OPERATOR_INSTRUCTIONS[operator])
