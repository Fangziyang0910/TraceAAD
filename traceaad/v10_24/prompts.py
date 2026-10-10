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
    "Explore": "Identify and investigate one concrete solver-effect question. Choose one concrete mechanism and implement its first test on the anchor, keeping mature independent parts. A small change with a meaningful effect is valid. Completion, simulation and local improvement are also allowed. If ranking, normalization, masking or unavailable state cancels the proposed effect, choose a compatible implementation.",
    "Crossover": "Identify one concrete computation in the reference that could help the anchor. Explain its connection point, required inputs/state and downstream effect. Implement the integration and resolve normalization, filtering and return-contract interactions. Paired scores suggest complementarity but do not identify the causal component. A weaker reference can still help.",
    "Develop": "Investigate the current solver-effect question using the observed trials. The first Change was a tested hypothesis, not an instruction to repeat. Explain what the results support or contradict, which useful computations remain, and what the next change will test. Temporary score loss is allowed; the host separately retains the best result. A rollback may support a different next test, but returning to a known program produces no new evidence. Do not cycle between tested alternatives. If the question is resolved or you have no justified next test, declare change_request to close this unit; you may return the unchanged bound program. Replacing an implementation does not show that the original mechanism matured.",
    "Repair": "Repair the failed implementation using the actual failure evidence. For execution failure inspect repeated work, shared/batched/incremental computations and the placement of internal search. Retain the intended output effect where feasible. State which computations you retain, change or remove. Replacing code can serve the same question; explicitly declare change_request if you abandon that question. Call progress does not estimate an overshoot factor.",
}

OUTPUT = """[Output Format]
Analysis: (about 120 words total)
Base: <one ID from available_bases; explain a return to anchor/champion in Evidence>
Question: <solver behavior to investigate, separate from this particular implementation>
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
Question: <on the first trial, the solver-effect question; on later trials, retain that question>
Change: <one solver-effect change, including all code locations that must change together>
Effect: <how the returned value changes the solver's behavior>
Evidence: <observed results and what remains uncertain>
Status: <continue_request for another test; change_request to close a resolved or abandoned question, with a reason>
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
Copy SEARCH from that current source, never from a historical diff or failed edit.
Prefer short unique SEARCH fragments to copying long unchanged bodies. Preserve
all source whitespace in SEARCH. Check that the replacement is not already present.
Keep the original interface and include required imports/helpers. Add no comments
or docstrings. Formatting wrappers are optional; return only one delivery mode.
"""
class BaseContextTooLong(ContextTooLong):
    """The task contract and this complete edit base alone cannot fit."""


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
                + ["[Your Task: Init]\n" + GOALS["Init"], "Optional thinking aids (choose only if useful): " + "\n".join(AXES), OUTPUT], "Init")
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

    def trial(self, a, *, detail=True):
        p = self.programs.get(a.get("program_id"))
        before = self.programs.get(a.get("parent_id"))
        outcome = self.failure(p) if p and not p["valid"] else p["fitness"] if p else a.get("error")
        text = (f"Trial {a['id']}: {a['action']}, base={a.get('parent_id')}, code={a.get('program_id')}, "
                f"repair_of={a.get('repair_of')}, status={a['status']}, result={outcome}")
        if not detail:
            return text
        diff = a.get("actual_diff")
        if diff is None and p and before:
            diff = code_diff(before["code"], p["code"])
        return (text + "\nReported reasoning (not verified): " + a.get('control', {}).get('analysis', '') +
                f"\nActual diff:\n{diff or '(no source change available)'}" +
                ("\nUnapplied edit submission (not program state):\n" + a['failed_edit_submission']
                 if a.get('failed_edit_submission') else ''))

    def request(self, unit, block, action, default, linked, paired=None):
        # Archive completeness and prompt completeness have different responsibilities.
        sections = self.common + [self.code(default, 'HOST-BOUND EDITING BASE'),
            f"available_bases={[default]}; system_default_base={default}. "
            "Only this complete source may be selected as a base. Historical diffs are evidence, not editable snapshots."]
        ending = ["[Your Task: " + action + "]\n" + GOALS[action],
                  f"[Delivery target] Edit code {default}, the complete HOST-BOUND EDITING BASE above. "
                  "Historical changes and failed submissions are observations, not current instructions or source.", EDIT_OUTPUT]
        def render(extra=()):
            return self._result(sections + list(extra) + ending, action)
        if self._result(sections + [EDIT_OUTPUT], action)['input_tokens'] > self.config.max_input_tokens:
            raise BaseContextTooLong('task contract and complete editing base exceed context')
        current = "[Current Request]\n" + json.dumps({
            "id": unit['id'], "source": unit['source'], "purpose": block['purpose'],
            "question": unit['question'], "block": len(unit['block_ids']),
            "max_blocks": 2, "spent_on_request": len(unit['trial_ids']),
            "attempts_remaining": block['limit'] - block['spent'],
            "future_block": "unallocated" if len(unit['block_ids']) < 2 else "none",
            "roles": {k: unit[k] for k in ('anchor_id', 'worktip_id', 'champion_id', 'proposal_id', 'pending_failure', 'donor_id')}
        }, ensure_ascii=False)
        latest = unit['trial_ids'][-1:]
        required = [current] + [self.trial(self.attempts[i], detail=False) for i in unit['trial_ids']]
        if render(required)['input_tokens'] > self.config.max_input_tokens:
            raise ContextTooLong('current question and latest feedback exceed local context')
        sections += required
        shown_trials, related_trials = list(unit['trial_ids']), []
        materials, excerpts, omitted = [default], [], []

        def include(label, text):
            if render([text])['input_tokens'] <= self.config.max_input_tokens:
                sections.append(text)
                return True
            omitted.append(label)
            return False

        # A transfer needs actual donor code; if it cannot fit, record a refinement fallback.
        donor = unit['donor_id']
        if action == 'Crossover' and donor is not None:
            if include(f'donor:{donor}', self.code(donor, 'REFERENCE ONLY')):
                materials.append(donor)
                if paired:
                    include('paired_scores', '[Paired training outcomes: anchor, donor]\n' + json.dumps(paired))
            else:
                result = self.request(unit, block, 'Refine', default, linked)
                result['trims'].append('donor_unavailable:refine_fallback')
                return result

        # Recent transitions, the original proposal and repair removals precede other history.
        ids = list(dict.fromkeys(list(reversed(unit['trial_ids'][-2:])) + unit['trial_ids'][:1] +
                   [i for i in reversed(unit['trial_ids']) if self.attempts[i].get('repair_of')] +
                   list(reversed(unit['trial_ids']))))
        for i in ids:
            a = self.attempts[i]
            if include(f'trial_detail:{i}', self.trial(a)):
                if i not in shown_trials:
                    shown_trials.append(i)
            else:
                if i not in shown_trials and include(f'trial_outcome:{i}', self.trial(a, detail=False)):
                    shown_trials.append(i)
                # Keep complete diff hunks rather than cut arbitrary code lines.
                diff = a.get('actual_diff', '')
                for n, hunk in enumerate(diff.split('\n@@')):
                    if hunk and include(f'trial_hunk:{i}:{n}',
                            f'[Trial {i}: partial diff, other hunks may be omitted]\n' +
                            ('@@' if n else '') + hunk):
                        excerpts.append({'trial_id': i, 'hunk': n})

        related_budget = self.config.history_depth
        for u in linked:
            producer = self.programs[unit['anchor_id']]['attempt_id']
            related_ids = sorted(u['trial_ids'], key=lambda i: (i != producer, -i))
            for i in related_ids[:related_budget]:
                related_budget -= 1
                heading = f'[Source-related request {u["id"]}]\n'
                if (i == producer and include(f'producer_detail:{i}', heading + self.trial(self.attempts[i]))
                        or include(f'related:{u["id"]}:{i}', heading + self.trial(self.attempts[i], detail=False))):
                    related_trials.append(i)
        seen = {default}
        for role in ('proposal_id', 'champion_id', 'anchor_id', 'worktip_id', 'donor_id'):
            pid = unit[role]
            if pid is None or pid in seen or pid in materials:
                continue
            seen.add(pid)
            p = self.programs[pid]
            outcome = p['fitness'] if p['valid'] else self.failure(p)
            diff = code_diff(self.programs[default]['code'], p['code'])
            include(f'role_diff:{role}:{pid}',
                    f'[{role}: code {pid}, result={outcome}, diff FROM editing base {default}; not a selectable base]\n' + diff)
        if action == 'Explore':
            include('thinking_aids', '[Optional thinking aids; no required axis or quota]\n' + '\n'.join(AXES))
        result = render()
        result.update(available_bases=[default], system_default_base_id=default,
            material_ids=materials, attempt_ids=shown_trials, related_trial_ids=related_trials,
            evidence_excerpts=excerpts, omitted_materials=omitted, trims=omitted,
            linked_unit_ids=[u['id'] for u in linked], context_version='v1024-context-3',
            prompt_hash=stable_key('prompt', result['prompt']))
        return result
