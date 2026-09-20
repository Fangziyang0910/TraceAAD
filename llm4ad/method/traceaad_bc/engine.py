"""The two missing cells in the V10.11/V11.0 scheduler-context ablation.

The combinations deliberately reuse the reviewed V10.11 and V11.0 engines.
Only one of the two dimensions is switched in each class:

* ``TraceAADV11BudgetV10Context`` keeps V11's code-level percentile scheduler
  and uses V10.11's short trajectory plus Fuse donor context.
* ``TraceAADV10BudgetV11Context`` keeps V10.11's ESS/Softmax parent and donor
  allocation and uses V11's trajectory/reference context split.

The classes have distinct METHOD identities so checkpoints and run manifests
cannot be resumed across arms accidentally.
"""

from __future__ import annotations

import ast
from functools import lru_cache

from ..traceaad_v10_11 import parsing as v1011_parsing
from ..traceaad_v10_11.prompts import TrajectoryBuilder as V1011TrajectoryBuilder
from ..traceaad_v10_11.selection import (
    DONOR_UNIFORM_PROBABILITY,
    OPERATORS as V1011_OPERATORS,
    OPERATOR_PROBABILITIES as V1011_OPERATOR_PROBABILITIES,
    QUALITY_ESS_TARGET,
    calibrate_beta as v1011_calibrate_beta,
    mix_uniform,
    softmax as v1011_softmax,
)
from ..traceaad_v10_11.traceaad import (
    REPAIRABLE_FAILURES as V1011_REPAIRABLE_FAILURES,
    TraceAADV1011,
)
from ..traceaad_v11_0.prompts import (
    ReferenceContextBuilder as V110ReferenceContextBuilder,
    TrajectoryBuilder as V110TrajectoryBuilder,
)
from ..traceaad_v11_0.selection import (
    N_REFERENCES,
    OPERATORS as V110_OPERATORS,
    OPERATOR_PROBABILITIES as V110_OPERATOR_PROBABILITIES,
    code_key as v110_code_key,
    reciprocal_rank_sample,
)
from ..traceaad_v11_0.traceaad import (
    REPAIRABLE_FAILURES as V110_REPAIRABLE_FAILURES,
    TraceAADV110,
)


@lru_cache(maxsize=8192)
def _v1011_code_key(code: str) -> str:
    return ast.dump(ast.parse(code), include_attributes=False)


def _v1011_quality_distribution(nodes):
    """Return V10.11's ESS-calibrated quality distribution for donor choice."""
    scores = [node.fitness for node in nodes]
    beta, target, quality_ess = v1011_calibrate_beta(scores, QUALITY_ESS_TARGET)
    return v1011_softmax(scores, beta), {
        "quality_ess": quality_ess,
        "ess_target": target,
    }


class TraceAADV11BudgetV10Context(TraceAADV110):
    """B: V11 parent scheduling with the V10.11 generic context package."""

    METHOD = "bc_v11budget_v10ctx"
    PROMPT_POLICY = "generic_design_v1"

    def __init__(self, *, n_references=N_REFERENCES, **kwargs):
        # V11's n_references is retained in the constructor/mechanism for a
        # stable common runner, but this arm never samples archive references.
        super().__init__(n_references=n_references, **kwargs)
        self.mechanism = {
            **self.mechanism,
            "scheduler": "v11_percentile_plus_bonus",
            "context": "v1011_generic_trajectory_donor",
            "context_operator_instructions": "v1011",
            "context_reference_weighting": None,
            "context_n_references": 0,
        }
        self.builder = V1011TrajectoryBuilder(
            self.llm,
            self.task_contract,
            max_tokens=self.max_input_tokens,
            max_events=self.traj_gens,
            lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes,
            include_history_code=False,
        )

    def _select_v1011_donor(self, parent):
        parent_key = _v1011_code_key(parent.code)
        nodes = [
            node for node in self.tree.all_nodes()
            if node.id != parent.id and _v1011_code_key(node.code) != parent_key
        ]
        if not nodes:
            return None
        quality, _ = _v1011_quality_distribution(nodes)
        return self.rng.choices(
            nodes,
            weights=mix_uniform(quality, DONOR_UNIFORM_PROBABILITY),
        )[0]

    def _schedule(self):
        previous = self.storage.last_event
        if previous and previous["candidate_id"] != self.completed_attempts:
            previous = None
        if (previous and previous.get("status") == "eval_failed"
                and previous.get("reason") not in V110_REPAIRABLE_FAILURES):
            raise RuntimeError(f"evaluation infrastructure failed: {previous.get('reason')}")
        if (previous and (previous.get("status") == "invalid_output"
                          or previous.get("reason") in V110_REPAIRABLE_FAILURES)
                and not previous.get("repair_of")):
            return self._schedule_repair(previous)

        requested = operator = "Init"
        parent = donor = None
        selection = {}
        if len(self.tree.roots) >= self.n_roots:
            requested = operator = self.rng.choices(
                V110_OPERATORS, weights=V110_OPERATOR_PROBABILITIES.values()
            )[0]
            parent, selection = self.select_parent()
            if requested == "Fuse":
                donor = self._select_v1011_donor(parent)
                if donor is None:
                    operator = "Refine"
                    selection["fallback_reason"] = "no donor"

        text = (self.builder.build_initial() if operator == "Init"
                else self.builder.build(parent, operator, donor))
        return self._candidate(
            text,
            requested_operator=requested,
            operator=operator,
            parent_id=parent.id if parent else None,
            donor_id=donor.id if donor else None,
            parent_fitness=parent.fitness if parent else None,
            donor_fitness=donor.fitness if donor else None,
            reference_ids=[],
            selection=selection,
        )


class TraceAADV10BudgetV11Context(TraceAADV1011):
    """C: V10.11 ESS/Softmax scheduling with V11 reference context."""

    METHOD = "bc_v10budget_v11ctx"
    PROMPT_POLICY = "operator_conditional_v1"

    def __init__(self, *, n_references=N_REFERENCES, **kwargs):
        # The V10.11 base supplies the ESS/Softmax parent and Pivot-uniform
        # allocation. The builders below replace only generation context.
        super().__init__(**kwargs)
        if n_references < 1:
            raise ValueError("n_references must be positive")
        self.n_references = n_references
        self.mechanism = {
            **self.mechanism,
            "scheduler": "v1011_quality_ess_plus_pivot_uniform",
            "context": "v110_trajectory_reference",
            "context_operator_instructions": "v110",
            "context_reference_weighting": "reciprocal_rank",
            "context_n_references": n_references,
            "history_code": False,
        }
        self.builder = V110TrajectoryBuilder(
            self.llm,
            self.task_contract,
            max_tokens=self.max_input_tokens,
            max_events=self.traj_gens,
            lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes,
        )
        self.reference_builder = V110ReferenceContextBuilder(
            self.llm,
            self.task_contract,
            max_tokens=self.max_input_tokens,
            max_events=self.traj_gens,
            lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes,
        )

    def reference_pool(self, parent):
        """V11 archive pool: latest node for each non-current code with an Idea."""
        current = v110_code_key(parent.code)
        pool = {}
        for node in self.tree.all_nodes():
            key = v110_code_key(node.code)
            if key == current or not node.idea or not node.idea.strip():
                continue
            pool[key] = node
        return list(pool.values())

    def _schedule_repair(self, previous):
        prompt = v1011_parsing.repair_prompt(
            self.task_contract,
            self.storage.failed_response(previous["candidate_id"]),
            previous,
        )
        return self._candidate(
            prompt,
            repair_of=previous["candidate_id"],
            operator=previous.get("operator", "Init"),
            requested_operator=previous.get("requested_operator", "Init"),
            parent_id=previous.get("parent_id"),
            donor_id=None,
            parent_fitness=previous.get("parent_fitness"),
            donor_fitness=None,
            reference_ids=[],
            selection=previous.get("selection", {}),
        )

    def _schedule(self):
        previous = self.storage.last_event
        if previous and previous["candidate_id"] != self.completed_attempts:
            previous = None
        if (previous and previous.get("status") == "eval_failed"
                and previous.get("reason") not in V1011_REPAIRABLE_FAILURES):
            raise RuntimeError(f"evaluation infrastructure failed: {previous.get('reason')}")
        if (previous and (previous.get("status") == "invalid_output"
                          or previous.get("reason") in V1011_REPAIRABLE_FAILURES)
                and not previous.get("repair_of")):
            return self._schedule_repair(previous)

        requested = operator = "Init"
        parent = None
        selection = {}
        reference_ids = []
        if len(self.tree.roots) >= self.n_roots:
            requested = operator = self.rng.choices(
                V1011_OPERATORS, weights=V1011_OPERATOR_PROBABILITIES.values()
            )[0]
            nodes = self.eligible_nodes()
            probabilities, selection = self.node_distribution(nodes, requested)
            index = self.rng.choices(range(len(nodes)), weights=probabilities)[0]
            parent = nodes[index]
            count = self.parent_selection_counts.get(parent.id, 0)
            self.parent_selection_counts[parent.id] = count + 1
            selection.update(
                parent_probability=probabilities[index],
                parent_count_before=count,
            )

            if requested in ("Pivot", "Fuse"):
                sampled = reciprocal_rank_sample(
                    self.reference_pool(parent), self.n_references, self.rng
                )
                text, retained = self.reference_builder.build(
                    parent,
                    requested,
                    sampled,
                    require_minimal=requested != "Fuse",
                )
                reference_ids = [node.id for node in retained]
                if requested == "Fuse" and not retained:
                    operator = "Refine"
                    selection["fallback_reason"] = (
                        "reference_pool_empty" if not sampled
                        else "references_trimmed_to_zero"
                    )
                    text = self.builder.build(parent, "Refine")
                    reference_ids = []
            else:
                text = self.builder.build(parent, operator)
        else:
            text = self.builder.build_initial()

        return self._candidate(
            text,
            requested_operator=requested,
            operator=operator,
            parent_id=parent.id if parent else None,
            donor_id=None,
            parent_fitness=parent.fitness if parent else None,
            donor_fitness=None,
            reference_ids=reference_ids,
            selection=selection,
        )
