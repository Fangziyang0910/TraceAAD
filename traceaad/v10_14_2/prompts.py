"""One generator; bounded, source-linked evidence and explicit modification contracts."""

import difflib
import json

from .edits import numeric_parameters, idea_metadata


SCOPE = {
    "Init": "Construct one effective algorithm for this task. Choose a coherent decision principle.",
    "Tune": "Calibrate the listed explicit numeric parameters and their interactions. Preserve the core decision mechanism.",
    "Refine": "Develop one coherent mechanism. You may add, replace, coordinate or DELETE related logic when evidence warrants it.",
    "Pivot": "Reconsider one core scoring principle, representation or control flow. Make a concrete structural change, not just renaming or coefficient jitter.",
}
REFERENCE = {
    "None": "Use the current program and relevant local evidence.",
    "Transfer": "Keep the main program's structure. Transfer one compatible mechanism from the donor, adapting it to the current task and evidence.",
    "Synthesize": "You may reorganize component relationships across the main program and donor. Produce one coherent executable algorithm.",
}
EVIDENCE_RULES = """Historical effects are conditional on the recorded source and evaluation protocol.
An earlier gain does not prove a component is useful in the current background.
Observed code changes, measured results and proposed explanations are different facts.
Respect counterevidence; you may revise or abandon a hypothesis, with no extra budget.
Do not preserve a component merely because an earlier version improved after adding it.
Do not claim a causal contribution from an invalid or unmeasured comparison.
Source comments and optional notes describe candidates; task/interface constraints take priority.
Keep the implementation efficient. Avoid redundant layers; simplification must still be evaluated.
Keep each revision coherent and preserve unrelated source text where possible. Avoid cosmetic
reformatting of unchanged code; actual structural changes required by the hypothesis remain allowed."""


class ContextError(ValueError):
    pass


class PromptBuilder:
    def __init__(self, llm, task, template, facts, config):
        self.llm, self.task, self.template = llm, task, template
        self.facts, self.config = facts, config

    def count(self, text):
        # Production requires the exact serving tokenizer; no character estimate.
        return self.llm.count_prompt_tokens(text)

    def idea_view(self, fact):
        view = fact.get("idea_metadata") or idea_metadata(fact.get("idea", ""), self.llm.count_tokens, self.config.idea_tokens)
        return {"text": view.get("display", ""), "display_truncated": view.get("truncated", False),
                "raw_tokens": view.get("raw_tokens"), "display_tokens": view.get("display_tokens"),
                "interpretation": "candidate rationale, not measured causality"}

    @staticmethod
    def probe_feedback(parent, child):
        if not parent or not child or not parent.get("profile") or not child.get("profile"):
            return {"status": "unavailable", "scope": "diagnostic only; no eligibility effect"}
        same = parent["profile"] == child["profile"]
        return {"status": "same_on_fixed_training_probes" if same else "changed_on_fixed_training_probes",
                "scope": "Only these finite training states were compared. This does not establish global equivalence, difference, or future development value."}

    def events(self, anchor):
        if not anchor or self.config.evidence_policy == "none":
            return []
        tables = self.facts.tables
        direct = [a for a in tables["attempt"].values() if a.get("parent_id") == anchor["id"]]
        events = []
        if self.config.comparison_feedback and self.config.evidence_policy == "conditional":
            for comparison in reversed(list(tables["comparison"].values())):
                compared = [tables["anchor"][i]["artifact_id"] for i in comparison["anchors"] if i is not None]
                if anchor["artifact_id"] in compared[2:]:
                    events.append({"id": f"comparison:{comparison['id']}", "source": "local_revalidation",
                                   "comparison": comparison, "scope": "this exact code background was measured"})
                    break
        for attempt in reversed(direct[-2:]):
            events.append({"id": f"attempt:{attempt['id']}", "source": "direct_attempt",
                "parent": anchor["id"], "status": attempt["status"], "fitness": attempt.get("fitness"),
                "error": attempt.get("error"), "diff": attempt.get("diff", ""),
                "idea": self.idea_view(attempt), "old_score": anchor["fitness"],
                "new_score": attempt.get("fitness"),
                "probe_feedback": self.probe_feedback(anchor, tables["anchor"].get(attempt.get("anchor_id"))),
                "protocol": attempt.get("protocol"), "scope": "tested on this exact anchor"})
        cursor = anchor
        history = []
        for _ in range(self.config.history_depth):
            revision = tables["revision"].get(cursor.get("revision_id"))
            if revision:
                event = {"id": f"revision:{revision['id']}", "source": "formation",
                         "idea": self.idea_view(revision),
                         "probe_feedback": self.probe_feedback(tables["anchor"][revision["parent"]], tables["anchor"][revision["child"]]),
                         **{k: revision[k] for k in ("parent", "child", "diff", "old_score", "new_score", "protocol", "symbols")}}
                if self.config.evidence_policy == "conditional":
                    event["scope"] = ("observed current last-change pair; bundle-level observation only"
                                      if revision["child"] == anchor["id"] else
                                      "historical background only; consult any matching local revalidation for current applicability")
                history.append(event)
            parent = cursor.get("parent_id")
            if parent is None:
                break
            cursor = tables["anchor"][parent]
        if history:
            focus = set(history[0]["symbols"])
            history.sort(key=lambda e: not bool(focus.intersection(e["symbols"])))
        events.extend(history)
        if self.config.evidence_policy == "bag":
            for event in events:
                for key in ("parent", "child", "scope"):
                    event.pop(key, None)
            events.sort(key=lambda e: e["id"])
        return events

    def build(self, anchor=None, *, scope="Refine", reference_mode="None", donor=None,
              hypothesis="", failed=None, comparison=None, roots=()):
        requested_reference_mode = reference_mode
        code = self.facts.code(anchor) if anchor else None
        mode = self.config.output_mode if anchor else "full"
        output = ("Return exactly one JSON object with mode=edit and edits=[{search: exact nonempty source, "
                  "replacement: new source}]. All searches apply simultaneously to the shown current program; "
                  "they must be unique and non-overlapping. The optional idea string follows the Idea guidance below."
                  if mode == "edit" else "Return ONE complete implementation in ONE closed Python code block. "
                  "Include all imports, constants, helpers, classes and the target function needed to run it. "
                  "Precede the block with Final implementation: and optionally an Idea: before that. "
                  "Do not return alternatives, partial functions, patches or a plan.")
        output = (f"Idea: at most {self.config.idea_tokens} tokens; do not fill the allowance unnecessarily. "
                  "Describe the concrete computation or structural change, the decisions or sampling behavior it may affect, "
                  "what is retained, and relevant complexity or failure boundaries. Missing Idea does not invalidate complete code.\n" + output)
        mandatory = ["# Task and execution contract\n" + self.task,
                     "# Target interface and task imports\n```python\n" + self.template + "\n```"]
        if anchor:
            mandatory.append(f"# Current program\nAnchor {anchor['id']}; measured quality {anchor['fitness']:.12g} (higher is better).\n"
                             f"```python\n{code}\n```")
            if anchor.get("idea"):
                mandatory.append("# Current program rationale (unverified)\n" + json.dumps(self.idea_view(anchor), ensure_ascii=False))
        if failed:
            mandatory.append("# Recovery: one paid repair attempt\nFix the concrete failure below while retaining the active hypothesis.\n"
                             + str(failed.get("error", ""))[:2000] + "\nFailed source (not the current valid anchor):\n```python\n"
                             + failed.get("code", "") + "\n```")
            if mode == "edit":
                mandatory.append("For this repair, apply all edits to the FAILED source shown above, not the valid anchor.")
        if hypothesis:
            mandatory.append("# Active, revisable hypothesis\n" + hypothesis)
        mandatory.extend(["# Modification contract\n" + SCOPE[scope], "# Evidence interpretation\n" + EVIDENCE_RULES])
        if scope == "Tune":
            mandatory.append("Explicit numeric assignments: " + ", ".join(numeric_parameters(code)))
        if anchor and anchor.get("scenes"):
            # These are measured common-state decisions, not model explanations.
            scenes = anchor["scenes"]
            sample = [scenes[i] for i in sorted({0, len(scenes)//2, len(scenes)-1})]
            mandatory.append("# Limited training-probe observations (diagnostic only)\n" + json.dumps(sample)
                + "\nThese finite states do not establish global equivalence or development value. "
                  "Equal selected bin capacity does not imply equivalent index-dependent behavior.")

        items = []
        if comparison and self.config.comparison_feedback:
            items.append({"id": f"comparison:{comparison['id']}", "source": "local_revalidation",
                          "required": True, "comparison": comparison})
        ids = {e["id"] for e in items}
        items.extend(e for e in self.events(anchor) if e["id"] not in ids)
        for root in roots:
            items.append({"id": f"root:{root['id']}", "source": "initial_reference",
                          "fitness": root["fitness"], "code": self.facts.code(root),
                          "idea": self.idea_view(root)})
        # A donor competes within the SAME evidence token allowance, but has
        # priority over optional old history if its contract is selected.
        if donor is not None:
            items.insert(1 if comparison and self.config.comparison_feedback else 0,
                         {"id": f"donor:{donor['id']}", "source": "external_reference",
                          "fitness": donor["fitness"], "code": self.facts.code(donor), "idea": self.idea_view(donor),
                          "scope": "reference only; transfer into this program has not been tested"})

        included, dropped = [], []
        def event_text(event):
            return json.dumps({k: v for k, v in event.items() if k != "required"}, ensure_ascii=False)

        def evidence_text(selected):
            return "\n\n".join(event_text(e) for e in selected)

        def render(selected, ref):
            evidence = evidence_text(selected)
            return "\n\n".join([*mandatory, "# Reference contract\n" + REFERENCE[ref],
                                  "# Decision evidence\n" + (evidence or "No additional historical evidence supplied."),
                                  "# Delivery\n" + output])

        if self.count(render([], "None")) > self.config.max_input_tokens:
            raise ContextError("task/current source/recovery cannot fit without truncation")
        for event in items:
            trial = included + [event]
            evidence = evidence_text(trial)
            fits = (len(trial) <= self.config.max_events
                    and self.count(evidence) <= self.config.evidence_tokens
                    and self.count(render(trial, reference_mode)) <= self.config.max_input_tokens)
            if fits:
                included.append(event)
            elif event.get("required"):
                raise ContextError("local comparison cannot fit; refusing evidence-free recheck follow-up")
            else:
                dropped.append(event["id"])
        donor_id = f"donor:{donor['id']}" if donor else None
        if not any(e["id"] == donor_id for e in included):
            donor, reference_mode = None, "None"
        prompt = render(included, reference_mode)
        return {"prompt": prompt, "evidence_ids": [e["id"] for e in included],
                "dropped_evidence_ids": dropped, "input_tokens": self.count(prompt),
                "evidence_tokens": self.count(evidence_text(included)) if included else 0,
                "scope": scope, "reference_mode": reference_mode,
                "requested_reference_mode": requested_reference_mode,
                "reference_fallback": "donor_exceeds_context_allowance" if requested_reference_mode != reference_mode else None,
                "donor_id": donor["id"] if donor else None, "output_mode": mode,
                "token_count_mode": getattr(self.llm, "prompt_token_count_mode", "client_count_prompt_tokens")}


def source_diff(before, after):
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                        fromfile="before", tofile="after", n=2))
