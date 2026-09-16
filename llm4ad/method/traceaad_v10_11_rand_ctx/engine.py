"""TraceAAD V10.11 engine with random archive references instead of formation history."""

from ..traceaad_v10_11.traceaad import (
    OPERATORS,
    OPERATOR_PROBABILITIES,
    REPAIRABLE_FAILURES,
    TraceAADV1011,
)
from .context import RandomReferenceBuilder, rank_softmax_sample


class TraceAADV1011RandCtx(TraceAADV1011):
    """V10.11 search whose generation context swaps trajectory steps for archive cards.

    Selection, operators, repair, budget accounting and persistence follow V10.11
    unchanged. Each search request draws n_references archived programs (excluding
    the current parent) by rank softmax and shows their idea and measured fitness
    as unordered reference cards: no code, no operator, no sequential structure.
    """

    METHOD = "v1011rc"

    def __init__(self, *, n_references=8, **kwargs):
        if n_references < 1:
            raise ValueError("n_references must be positive")
        super().__init__(traj_gens=0, **kwargs)
        self.n_references = n_references
        self.mechanism = {**self.mechanism, "context": "random_references",
                          "n_references": n_references}
        self.builder = RandomReferenceBuilder(
            self.llm, self.task_contract, max_tokens=self.max_input_tokens,
            n_references=n_references, lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes)

    def _schedule(self):
        previous = self._current_event()
        if previous and previous.get("status") == "eval_failed" and previous.get("reason") not in REPAIRABLE_FAILURES:
            raise RuntimeError(f"evaluation infrastructure failed: {previous.get('reason')}")
        if previous and (previous.get("status") == "invalid_output" or
                         previous.get("reason") in REPAIRABLE_FAILURES) and not previous.get("repair_of"):
            return self._schedule_repair(previous)
        requested = operator = "Init"
        parent = donor = None
        selection = {}
        if len(self.tree.roots) >= self.n_roots:
            requested = self.rng.choices(OPERATORS, weights=OPERATOR_PROBABILITIES.values())[0]
            operator = requested
            nodes = self.eligible_nodes()
            probabilities, selection = self.node_distribution(nodes, operator)
            index = self.rng.choices(range(len(nodes)), weights=probabilities)[0]
            parent = nodes[index]
            count = self.parent_selection_counts.get(parent.id, 0)
            self.parent_selection_counts[parent.id] = count + 1
            selection.update(parent_probability=probabilities[index], parent_count_before=count)
            if requested == "Fuse":
                donor = self.select_donor(parent)
                if donor is None:
                    operator = "Refine"
                    selection["fallback_reason"] = "no donor"
        references = [] if parent is None else rank_softmax_sample(
            [node for node in self.tree.all_nodes() if node.id != parent.id],
            self.n_references, self.rng)
        text = self.builder.build_initial() if operator == "Init" else self.builder.build(
            parent, operator, donor, references=references)
        return self._pending(
            text, requested_operator=requested, operator=operator,
            parent_id=parent.id if parent else None, donor_id=donor.id if donor else None,
            parent_fitness=parent.fitness if parent else None,
            donor_fitness=donor.fitness if donor else None,
            selection=selection, reference_ids=[node.id for node in references],
        )
