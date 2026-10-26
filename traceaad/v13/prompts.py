"""V13 prompts contain task semantics and observed program feedback, not evaluator internals."""

import json
from traceaad.history import code_diff
from .knowledge import ancestors, trial_view

FORMATS = {
    'generate': '''Return JSON with analysis, design, code (complete self-contained Python source).
For Init, propose a computationally distinct algorithm from existing roots, not just another
constant setting of the same algorithm. The roots should provide different algorithmic routes.
In analysis state recent progress, failed directions and useful sibling ideas. For an open edit,
avoid already-known library operations. Do not repeat a failed direction without identifying
an actual subsequent change in this parent's computation that makes the retry relevant.
For crossover combine improvements from both branches and respect both failure histories.''',
    'match': '''Return JSON with matches (list). Each match has id and bindings (list of {role, old}).
Match computational roles against CURRENT_PARENT only, never against its siblings or examples.
Each old is an exact unique contiguous block of CURRENT_PARENT. Include all dependency/import
locations that may need editing. Return an empty list if no candidate applies. Bind actual existing
computations, not desired future computations. Match the BEFORE structure required by the
operation, not its desired AFTER structure. An operation is already present only when ALL its
changes are already implemented; sharing the same algorithm family is not enough to reject it.
Also return rejections (object mapping each rejected id to a concrete structural reason).''',
    'edit': '''Return JSON with design, analysis, edits (list of {old, new}). Each old must match
exactly once in the ORIGINAL current parent and lie wholly inside one bound region. Patches
must not overlap. Implement only the operation and permitted parameter adaptations. Preserve
unrelated code. Include necessary dependencies within the bound regions. No full-code field.''',
    'audit': '''Independently inspect the ACTUAL diff and recipient code, not the generator's claim.
Check preconditions, implemented operation, every dependency, parameters and preserved logic.
Return JSON with summary (actual change), operation_ok (boolean), dependencies_ok (boolean),
preservation_ok (boolean), parameters_ok (boolean), reason (string). If any required property is
unsupported or violated, mark its boolean false. A score improvement cannot establish fidelity.''',
    'describe': '''Summarize executable changes and observed outcomes. Generator intent is not evidence.
Return JSON with change (brief actual delta), key (stable semantic change category, reusing an
existing category for an equivalent change), lineage (compressed formation summary, at most 2000
characters), features (list of computational roles), route (existing route id or null), and
route_name (short algorithmic family name if route is null). Merge common ancestry only once.
The change key identifies the EDIT OPERATION, not the algorithm family: candidate filtering,
score reweighting and cache reuse are different edits even within one algorithmic route.
Reuse a key only when the actual edits are equivalent. When parent is null, this is an independent
root: summarize its computations, use key "initialization", and do not invent a history from the
template or other roots. Classify by computations, not score. Do not attribute a whole-edit gain
to each individual change.''',
    'learn': '''Compare evaluated edits and existing library operations. Return JSON with kind:
"none", "attach", or "new", and reason explaining the decision. For none, say precisely why
this edit cannot currently be represented as a reusable operation. Source regions are already
known from the actual diff; do not copy code blocks or provide recipient bindings at this stage.
attach needs id of an equivalent existing operation, not merely a
related algorithm. new needs spec. spec contains nonempty strings name, applicability, operation,
dependencies, parameters, preserve, checks, plus requires (list of existing transformation ids).
applicability describes the BEFORE structure. operation must describe an actual EDIT in imperative
form: replace which old computation with which new computation, or insert/remove what. Do not
summarize the whole resulting algorithm as if it were a transformation. Name every cooperating
change that makes this edit different from the parent, including execution strategy and new terms.
Use computational roles, keep concrete before/after meaning, and separate fixed operations from
allowed parameter adaptations. Existing operations should receive evidence, not duplicate entries.
An isolated unsuccessful edit must not create a new operation. Repeated cross-route failures can
be attached to an existing operation but cannot alone establish a useful new tool. A new operation
needs at least one improving example. The improves flags are computed from the actual scores
(lower is better). ONE improving edit is enough to propose a CANDIDATE conditional operation.
Fundamental novelty, universal usefulness and demonstrated transfer are NOT admission requirements;
later applications will establish its scope. A concrete local edit or parameterized modification
can be reusable. If several changes cooperate, preserve them as a compound operation: do not
claim the whole gain for each part. Return none only when no concrete reusable edit can be
identified from the evidence, and explain that specific obstacle.''',
    'revise': '''Review actual use records, including structural nonmatches and implementation failures.
Do not count implementation failures or nonmatches as evidence of performance harm. Compare positive
and negative recipient structures; observations suggest scope, not causal proof. Return JSON with
kind (keep, revise, split, compose), reason, specs (list), and components (list of existing ids).
Each spec has name, applicability, operation, dependencies, parameters, preserve, checks (strings)
and requires (existing ids). revise supplies one spec and may clarify conditions/dependencies or
parameters but must keep the same operation. split supplies two narrower operations, preserving
old evidence. compose supplies one joint operation and its component ids, including the reviewed
operation; it must explicitly implement the components together. Discover a prerequisite only when
successful recipients already used it and comparable failures lacked it. Otherwise do not invent
prerequisites. keep uses specs=[] and components=[]. Retain negative records and prior versions.''',
}


def node_view(node):
    return dict(id=node.id, code=node.code, score=node.fitness,
                failure=node.failure['kind'] if node.failure else None,
                trace=node.request.get('trace', {}))


def trajectory(parent, programs, attempts, depth):
    ids = sorted(ancestors(parent.id, programs))
    recent = ids[-depth:]
    edges = []
    for pid in recent:
        p = programs[pid]
        before = programs.get(p.parent_id)
        edges.append(dict(id=pid, parent=p.parent_id, reference=p.reference_id,
                          score=p.fitness, failure=p.failure['kind'] if p.failure else None,
                          change=p.request.get('trace', {}).get('change'),
                          diff=code_diff(before.code, p.code) if before else None))
    branches = []
    for p in programs.values():
        if p.parent_id in ids and p.id not in ids:
            branches.append(dict(id=p.id, parent=p.parent_id, score=p.fitness,
                                 failure=p.failure['kind'] if p.failure else None,
                                 change=p.request.get('trace', {}).get('change', p.idea)))
    return dict(lineage=parent.request.get('trace', {}).get('lineage', ''), recent=edges,
                branches=branches[-depth:], trials=trial_view(parent.id, programs, attempts))


def prompt(common, stage, required, optional=()):
    return '\n\n'.join(common) + '\n[V13:' + stage + ']\n' + FORMATS[stage] + '\n' + json.dumps(
        {**required, 'context': list(optional)}, ensure_ascii=False)
