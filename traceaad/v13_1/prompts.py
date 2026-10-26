"""Stage-specific, grounded contracts. Selection and credit never come from the LLM."""
import json
from traceaad.v13.prompts import node_view

SPEC = '''spec must contain nonempty text fields name, applicability (BEFORE structure), operation
(imperative concrete edit), dependencies, parameters (ALL fixed semantic choices), preserve,
checks, binding_policy (computational roles; only variable/interface names may adapt), plus
conditions: at most three {name, kind, predicate, rationale} objects. kind is static or runtime.
Each predicate is a fixed binary proposition evaluable BEFORE editing. State thresholds explicitly.
Prefer semantic static properties visible in source. Runtime claims require actual measurements:
if unavailable the condition is UNKNOWN, never inferred from code or the observed gain.
Keep exact constants, scoring formula, shortlist rule and execution strategy from the actual edit.
Changing a behavioral choice creates a different operation. Multiple inseparable changes form ONE
compound operation; never attribute the whole gain to each component. No generality/novelty proof
is required to propose an operation from one improvement.'''
FORMATS = {
'generate': '''Return JSON {analysis, design, code}. code is complete self-contained Python source.
For Init construct a different computational approach from the other root summaries.
For Open improve CURRENT_PARENT. Use observed path and sibling trials to reason about its current
background. Historical failures are evidence, NOT bans: a later background change may justify a
retry. You may discover a new operation. Do not claim a cached or merely proposed result is new
evidence. Keep analysis and design concise; avoid long code comments and copying history.''',
'match': '''Return JSON {matches: [...], rejections: {id: reason}}. For EVERY candidate, either
return a match or a concrete structural rejection. A match is {id, bindings:[{role,old}],
symbols:{role:identifier}, conditions:[{value:true|false|null,evidence:[exact_source_snippets]}]}.
Match only CURRENT_PARENT, not source examples or desired after-code. Each old is an exact UNIQUE
contiguous parent block. Bind the main computation, helper/import/dependency sites that need editing.
Use the SAME role names in all recipients. symbols maps roles to actual Python identifiers only.
All semantic parameters of the AFTER operation are FROZEN across replays. The operation may
explicitly change a BEFORE constant to the fixed AFTER constant. For example, an operation
"replace candidate cap 20 by 50" MUST change the parent's 20 to 50; freezing does not mean
preserving the BEFORE value. The reference_edit illustrates the complete actual source change.
All semantic parameters are FROZEN. Do NOT choose a parameter value or alternative strategy.
If the complete operation is already present at this site, reject that binding. Novelty and predicted gain are NEVER applicability criteria. A scalar candidate loop and a
vectorized batch computation are different execution strategies even when they implement related
scoring logic. Before rejecting as already present, check EVERY required semantic choice and
execution strategy against actual parent source; identify the exact statements implementing each.
A familiar algorithm family, existing NumPy use, or the same high-level scoring idea is insufficient.
Missing even one specified cooperating change means the complete operation is NOT already present.
Similar algorithm family alone is not a rejection. Do NOT reject because the operation once failed or is predicted
unhelpful. Conditions are not execution preconditions. Evaluate each fixed condition in spec order
using only pre-edit source and supplied measurements. For a static boolean, supply short exact
parent excerpts supporting the entire proposition. Insufficient evidence => null and evidence=[].
Runtime conditions without supplied measurements MUST be null. Do not invent timings, counts or
rank correlations. Never choose which transformation to execute: the search policy chooses.''',
'edit': '''Return JSON {analysis,design,edits:[{old,new}]} implementing EXACTLY the frozen spec.
Every old occurs once in ORIGINAL CURRENT_PARENT and lies inside a supplied bound region; edits
cannot overlap. Include required imports and helpers only inside their bound dependency regions.
Preserve interface, feasibility filtering, and invariants. Adapt names/interfaces via bindings;
do not change numeric values, ranking formula or algorithm choices. Do not add unrelated fixes.
If a previous attempt failed, repair ONLY that implementation under the same spec and boundaries;
all patches still apply to ORIGINAL CURRENT_PARENT, not the rejected child. Keep prose short.''',
'audit': '''Independently check ACTUAL diff against the frozen spec and bindings. Return JSON
{operation_ok:bool,dependencies_ok:bool,preservation_ok:bool,parameters_ok:bool,summary:str,reason:str}.
Check that every required cooperating change occurs, fixed semantic choices are unchanged, all
invariants hold, and no unrelated behavior is added. Do not use performance to judge compliance.
Reject incomplete or extra changes. Be concrete and concise; do not rewrite the code.''',
'learn': '''Extract a reusable operation from the supplied actual improving parent-child edit.
Account for the ENTIRE actual diff. First inspect all changed constants, formulas, candidate
limits, loops and execution strategy. If the edit changes cap 20 to 50, alpha 0.2/0.8 to
0.3/0.7, vectorizes a loop AND adds a rank penalty, the spec must include ALL FOUR changes as one
compound operation. Do not select only the most interesting change and attribute the whole gain
to it. A source audit will reject specs that leave any actual behavioral change unexplained.
Return JSON {kind:"new"|"attach"|"none", reason, spec, id}. attach requires EXACT equivalence
of operation, semantic parameters AND binding policy to a listed frozen spec. Similar purpose is
insufficient; use new for a behavioral variant. For none state the concrete representation obstacle.
Do not reject an observed improvement for lack of novelty or evidence of transfer. Source binding
is computed from the actual diff, so do not return recipient bindings or copy the entire code.
Use this exact shape for new (all fields required):
{"kind":"new","reason":"...","spec":{"name":"...","applicability":"...","operation":"...",
"dependencies":"...","parameters":"all fixed choices as text","preserve":"...","checks":"...",
"binding_policy":"...","conditions":[{"name":"...","kind":"static","predicate":"...","rationale":"..."}]},"id":null}.
Do not put condition predicates in applicability unless execution truly cannot proceed without them.
All frozen specs and conditions remain unchanged. Discovery is not validation. ''' + SPEC,
}


def prompt(common, stage, required, optional=()):
    return '\n\n'.join(common)+'\n[V13.1:'+stage+']\n'+FORMATS[stage]+'\n'+json.dumps(
        {**required,'context':list(optional)},ensure_ascii=False)
