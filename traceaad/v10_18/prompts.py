"""Version-specific goals and context; measured history is shared."""

from traceaad.common.prompts import PromptBuilder as MeasuredPrompts, ContextTooLong, idea_view, KNOWN
from traceaad.common.history import path, score_text

DEVELOP = f"""[Your Task: Develop the Proposed Design]
Develop the proposed design further: make the computation it introduces work as intended and write a version that scores better than its current version. {KNOWN}"""

ANALYSIS = {
    "Refine": "what limits the current algorithm, and what change keeps its core idea and makes it score better",
    "Crossover": "what the reference algorithm does well that the current algorithm lacks, and how to bring it in while keeping the current algorithm's core idea",
    "Repair": "what caused the failure, and how to fix it",
    "Explore": "what the current algorithm cannot capture, and what different computation should capture it",
    "Develop": ("whether the code computes what the proposed design intends and how that computation reaches the "
                "decisions, what keeps the design from scoring better, and what change to its implementation or to "
                "what it estimates fixes that"),
    "Init": None,
}

class PromptBuilder(MeasuredPrompts):
    ANALYSIS = ANALYSIS

    def develop(self, current, source, attempts):
        """Develop the design an Explore proposal introduced, from the best version it has reached.

        ``source`` is the program the proposal started from and ``attempts`` are
        the development attempts made on this design so far. The formation shown
        starts at the design's first version (a failed first version included);
        the source is stated with its score and Design, not as a version on the path.
        """
        sequence = path(current, self.programs)
        ids = [p["id"] for p in sequence]
        if source["id"] not in ids:
            raise ValueError("the current version does not descend from the exploration's source")
        sequence = sequence[ids.index(source["id"]) + 1:]
        sequence_ids = {p["id"] for p in sequence}
        best = max((p for p in self.programs.values() if p["valid"]), key=lambda p: p["fitness"])
        origin = (f"[Where the Proposed Design Came From]\nThe current algorithm develops a design that an Explore "
                  f"step proposed from an algorithm scoring {score_text(source['score'])} "
                  f"(Design: {idea_view(source['idea'])}). "
                  f"The best algorithm found so far in this search scores {score_text(best['score'])}.")
        repairs = self._repairs()
        off_path = [a for a in sorted(attempts, key=lambda a: a["id"])
                    if repairs.get(a["id"], a).get("program_id") not in sequence_ids]
        current_section = self._current(current)
        depth = len(self.steps(sequence)) - 1
        count, shown, trims = max(1, min(self.config.history_depth, depth)), len(off_path), []
        while True:
            history, edge_ids = self._formation(sequence, count, title="How the Proposed Design Has Been Developed",
                                                origin="first version of the proposed design")
            listed = off_path[len(off_path) - shown:] if shown else []
            tried = ""
            if listed:
                lines = []
                for attempt in listed:
                    start = self.programs[attempt["parent_id"]]
                    outcome = self._outcome(attempt, start, sequence_ids)
                    repair = repairs.get(attempt["id"])
                    if repair is not None:
                        outcome += ("; the repair returned the same failing program"
                                    if repair.get("program_id") == attempt.get("program_id")
                                    else "; repaired: " + self._outcome(repair, start, sequence_ids))
                    lines.append(f"Attempt {attempt['id']} · Develop from the version scoring "
                                 f"{score_text(start['score'])} · Design: {idea_view(attempt.get('idea')) or '(none)'}"
                                 f"\n  → {outcome}")
                tried = (f"[Other Development Attempts on This Design]\n{len(off_path)} development "
                         f"attempt{'' if len(off_path) == 1 else 's'} did not produce a version on the path above. "
                         + ("All are listed" if len(listed) == len(off_path)
                            else f"The most recent {len(listed)} are listed")
                         + ", oldest first, with their evaluated outcomes.\n\n" + "\n\n".join(lines))
            sections = self.common + [current_section, origin, history, tried, DEVELOP, self.output_format("Develop")]
            result = self._result(sections, "Develop", edge_ids, trims)
            result["attempt_ids"] = [a["id"] for a in listed]
            if result["input_tokens"] <= self.config.max_input_tokens:
                return result
            if count > 1:
                count -= 1
                trims.append("oldest_history")
            elif shown > 0:
                shown -= 1
                trims.append("oldest_attempt")
            else:
                raise ContextTooLong("Develop prompt with the latest development step exceeds context")
