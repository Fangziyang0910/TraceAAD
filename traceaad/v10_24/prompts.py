"""Decision context keeps failed proposals, repair transitions and sibling trials."""

import json

from traceaad.common.history import code_diff
from traceaad.common.prompts import ContextTooLong
from traceaad.v10_23.prompts import PromptBuilder as PreviousPrompts
from .policy import AXES, stable_key

TASK_CARDS = {
    "tsp_construct": "Choose one supplied unvisited node. The solver repeatedly consumes that node, then completes the last node and return edge. At 50 nodes there are 48 callback calls. Internal lookahead only affects the returned next node.",
    "cvrp_aco": "Return a static edge prior once per instance. The ACO combines it with pheromone and capacity/visit masks during repeated route construction. Relative positive magnitudes affect sampling, not just ordering. Ant position, remaining vehicle capacity and iteration are not callback inputs. Masked entries do not select actions.",
    "op_aco": "Return a static edge prior once per instance. The ACO combines it with pheromone and feasibility masks to sample moves. Positive monotone transformations can preserve ordering while changing probability ratios. Clipping and numerical stabilizers can break ideal scaling invariances. Ant state and iteration are not callback inputs; the dummy terminal is supplied by the solver.",
    "fssp_gls": "Return a perturbed matrix and 1-5 distinct job IDs. After descent on original processing times, the solver takes one best improving swap/insertion involving these jobs under the perturbed matrix. The callback runs once each round (200 rounds in the fixed protocol). The best true objective is retained. Perturbation need not approximate true durations; job selection and permitted randomness can change subsequent search. Round number is not supplied.",
    "graph_colouring": "Rank supplied moves by finite scores; argmax wins and ties select the first row. Construction and conflict repair share this callback, with phase exposed. A common strictly increasing transform preserves ranking in exact arithmetic. The candidate list, tabu rule and acceptance of a valid reduced coloring are external. Call count varies with construction and repair outcomes.",
    "jssp_construct": "Rank only the supplied Giffler-Thompson conflict set; argmax wins and ties select the first row. All instance times/routes and current readiness/progress are visible. A common strictly increasing transform preserves ranking in exact arithmetic. The callback cannot change conflict-set formation or machine assignments. There is one call per scheduled operation (400 at 20x20).",
}

GOALS = {
    "Init": "Create a complete useful starting implementation. Use the supplied solver facts to explain how its output affects the final solution. Avoid repeating the concrete computations shown without a reason.",
    "Refine": "Refine one concrete part of the anchor's computation. A parameter, representation, implementation or output mapping can be the right target. Complete all edits needed for this one change and preserve independent useful computations. Explain the downstream effect, not the edit size.",
    "Explore": "Investigate the supplied solver-effect question. Choose one concrete mechanism and implement its first test on the anchor, keeping mature independent parts. A small change with a meaningful effect is valid. Completion, simulation and local improvement are also allowed. If ranking, normalization, masking or unavailable state cancels the proposed effect, choose a compatible implementation.",
    "Crossover": "Identify one concrete computation in the reference that could help the anchor. Explain its connection point, required inputs/state and downstream effect. Implement the integration and resolve normalization, filtering and return-contract interactions. Paired scores suggest complementarity but do not identify the causal component. A weaker reference can still help.",
    "Develop": "Continue the same concrete request. Identify which requested computations exist in the working implementation, which were removed, and which remain unconnected to its return. Choose a supplied base and complete the next coherent change. Temporary score loss within the block is allowed. Replacing an implementation is not evidence that the original mechanism matured; explain the actual change.",
    "Repair": "Repair the failed implementation using the actual failure evidence. For execution failure inspect repeated work, shared/batched/incremental computations and the placement of internal search. Retain the intended output effect where feasible. State which computations you retain, change or remove. Replacing code can serve the same question; explicitly declare change_request if you abandon that question. Call progress does not estimate an overshoot factor.",
}

OUTPUT = """[Output Format]
Analysis: (about 120 words total)
Base: <one ID from available_bases; explain a return to anchor/champion in Evidence>
Change: <one concrete computation and code location>
Effect: <how the returned value changes the solver's behavior>
Evidence: <relevant observation, base choice and remaining uncertainty>
Status: <continue_request or change_request, followed by a reason>
Design: <at most 60 words describing what the actual code below computes>
Code:
```python
<complete self-contained implementation with the original interface>
```
Write no comments or docstrings in the code, and nothing after the code block.
"""
# Escape marker starts so Git does not treat the prompt as an unresolved merge.
EDIT_OUTPUT = """[Output Format]
Analysis: (about 120 words total)
Change: <one solver-effect change, including all code locations that must change together>
Effect: <how the returned value changes the solver's behavior>
Evidence: <observed results and what remains uncertain>
Status: <continue_request or change_request, with a reason>
Design: <at most 60 words describing the resulting algorithm>
Edits:
\x3c<<<<<< SEARCH
<exact existing text, with indentation and enough context to match uniquely>
\x3d======
<complete replacement computation>
\x3e>>>>>> REPLACE

Default to a batch of structurally complete edits to the host-bound editing base.
Submit every known dependent change in this same response. All SEARCH fragments
must match the ORIGINAL bound source, not the result of another edit in this batch.
Regions must not overlap; combine dependent or overlapping regions into one block.
Use an existing nonempty fragment as an anchor for additions; an empty replacement
deletes that fragment. The host applies the whole batch atomically and evaluates
one complete result. Source outside the edited regions is retained by the host.
Do not optimize for the fewest changed lines. Whole loops, helpers and functions
may be replaced together when that completes one coherent algorithm change.
If the program is short or a broad rewrite is clearer, use Code: followed by one
complete Python code block INSTEAD of Edits. Only for complete Code, an optional
Base: ID in Analysis may select another fully supplied base; explain it in Evidence.
Edits always target the host-bound base: omit Base, and do not copy a hash.
Keep the original interface and include required imports/helpers. Add no comments
or docstrings. Formatting wrappers are optional; return only one delivery mode.
"""
PLAN = """Only the first new-question proposal may optionally add these separate Analysis lines (about 80 words):
Plan: two blocks
Stage 1: <concrete first work>
Stage 2: <dependent second work>
Dependency: <why the second requires the first>
Location: <actual code locations>
Otherwise omit Plan or write Plan: single block. Admission is decided before this candidate is scored.
"""


class PromptBuilder(PreviousPrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        card = TASK_CARDS.get(self.task, "Use only the supplied interface and output consumer.")
        self.common.append("[TaskCard]\n" + card + "\nProgram globals are isolated between instances. "
                           "Only declared inputs and state within the instance are available. "
                           "Do not infer hidden iterations or access evaluation/test data.\n"
                           f"Contract source: benchmarks/{self.task}/template.py and evaluation.py.")

    def code(self, pid, role):
        p = self.programs[pid]
        result = str(p["fitness"]) if p["valid"] else self.failure(p)
        return f"[{role}: code {pid}] result={result}\n```python\n{p['code'].rstrip()}\n```"

    def initial(self, roots):
        shown = list(roots[-2:])
        while True:
            result = self._result(self.common + [self.code(p["id"], "Earlier root") for p in shown]
                + ["[Your Task: Init]\n" + GOALS["Init"], "Requested angle: " + AXES[len(roots) % len(AXES)], OUTPUT], "Init")
            if result["input_tokens"] <= self.config.max_input_tokens:
                return result
            if not shown:
                raise ContextTooLong("initial contract exceeds context")
            shown.pop(0)

    def repair(self, failed):
        # Initialization uses the existing one-repair rule; search repairs use the unit ledger.
        result = self._result(self.common + [self.code(failed["id"], "Failed implementation"),
                f"Host-bound editing base: code {failed['id']}. The failed source is the editing target.",
                "[Your Task: Repair]\n" + GOALS["Repair"], EDIT_OUTPUT], "Repair")
        if result["input_tokens"] > self.config.max_input_tokens:
            raise ContextTooLong("failed initial source exceeds context")
        return result

    def trial(self, a):
        p = self.programs.get(a.get("program_id"))
        before = self.programs.get(a.get("parent_id"))
        diff = a.get("actual_diff")
        if diff is None and p and before:
            diff = code_diff(before["code"], p["code"])
        outcome = self.failure(p) if p and not p["valid"] else p["fitness"] if p else a.get("error")
        return (f"Trial {a['id']}: {a['action']}, base={a.get('parent_id')}, code={a.get('program_id')}, "
                f"repair_of={a.get('repair_of')}, status={a['status']}, result={outcome}\n"
                f"Actual diff:\n{diff or '(no source change available)'}" +
                ("\nUnapplied edit submission (not program state):\n" + a['failed_edit_submission']
                 if a.get('failed_edit_submission') else ''))

    def request(self, unit, block, action, default, linked, paired=None):
        roles = [(default, 'Host-bound editing base'), (unit['anchor_id'], 'Anchor'), (unit['worktip_id'], 'Working implementation'),
                 (unit['champion_id'], 'Best NEW implementation'), (unit['proposal_id'], 'Original proposal'),
                 (unit['pending_failure'], 'Failed implementation'), (unit['donor_id'], 'Reference')]
        ids = list(dict.fromkeys(pid for pid, _ in roles if pid is not None))
        bases = [pid for pid in ids if pid != unit['donor_id'] or pid == default]
        sections = self.common + [
            "[Current Request]\n" + json.dumps({"id": unit['id'], "purpose": block['purpose'],
                "question": unit['question'], "axis": unit['axis'], "block": len(unit['block_ids']),
                "blocks_authorized": unit['blocks_authorized'], "spent_on_request": len(unit['trial_ids']),
                "attempts_remaining": block['limit'] - block['spent']}, ensure_ascii=False),
            "[Committed Dependency Plan]\n" + json.dumps({k: unit.get('plan', {}).get(k)
                for k in ('stage_1', 'stage_2', 'dependency', 'location')}) if unit.get('plan') else '',
            "[Code Roles]\n" + json.dumps({role: pid for pid, role in roles}) +
                f"\navailable_bases={bases}; system_default_base={default}",
            *[self.code(pid, 'HOST-BOUND EDITING BASE' if pid == default else 'Supplied reference snapshot') for pid in ids],
            "[Complete Request Ledger]\n" + "\n\n".join(self.trial(self.attempts[i]) for i in unit['trial_ids']),
            "Use the actual code changes and outcomes to decide what to try next. Prior Design text and "
            "the dependency plan are hypotheses, not proof of implementation or benefit. Avoid repeating "
            "the same tested change under the same conditions without new evidence. When opening a "
            "request from related work, state a concrete next problem rather than renaming the old one.",
        ]
        if paired:
            sections.append("[Paired training outcomes: anchor, donor]\n" + json.dumps(paired))
        extras = list(linked)
        trims = []
        while True:
            history = [f"[Related request {u['id']} / cost {len(u['trial_ids'])} / question {u['question']}]\n" +
                       "\n\n".join(self.trial(self.attempts[i]) for i in u['trial_ids']) for u in extras]
            result = self._result(sections + history + ["[Your Task: " + action + "]\n" + GOALS[action],
                PLAN if unit['purpose'] == 'new_question' and not unit['trial_ids'] else '', EDIT_OUTPUT], action, trims=trims)
            if result['input_tokens'] <= self.config.max_input_tokens:
                result.update(available_bases=bases, system_default_base_id=default,
                    material_ids=ids, attempt_ids=list(unit['trial_ids']),
                    related_trial_ids=[i for u in extras for i in u['trial_ids']],
                    linked_unit_ids=unit['linked_unit_ids'], context_version='v1024-edit-1',
                    prompt_hash=stable_key('prompt', result['prompt']))
                return result
            if not extras:
                raise ContextTooLong('required code roles and complete request ledger exceed context')
            trims.append(f"related_unit:{extras.pop(0)['id']}")
