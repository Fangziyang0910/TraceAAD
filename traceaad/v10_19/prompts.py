"""Version-specific goals and context; measured history is shared."""

from traceaad.common.prompts import PromptBuilder as MeasuredPrompts, ContextTooLong, idea_view, KNOWN
from traceaad.common.history import path, score_text, verdict
from .selection import outcome, by_proposal

EXPLORE = """[Your Task: Explore]
Write an algorithm that scores better than the best found so far by changing how the current algorithm makes its decisions, not by tuning it. Keep unchanged the parts of the current algorithm that the change does not replace."""

DEVELOP = f"""[Your Task: Make the Change Work]
Make the change work in the algorithm it was made to: keep the computation the change introduces and write a version that scores better than that algorithm. {KNOWN}"""

ANALYSIS = {
    "Refine": "what limits the current algorithm, and what change keeps its core idea and makes it score better",
    "Crossover": "what the reference algorithm does well that the current algorithm lacks, and how to bring it in while keeping the current algorithm's core idea",
    "Repair": "what caused the failure, and how to fix it",
    "Explore": "what the current algorithm cannot capture, and what different computation should capture it",
    "Develop": ("what the current algorithm computes differently from the algorithm the change was made to, why "
                "it scores worse than that algorithm, and what change keeps the new computation and makes it score better"),
    "Init": None,
}

COMPARED = {"improved": "better than", "worse": "worse than", "same score": "the same score as"}

class PromptBuilder(MeasuredPrompts):
    EXPLORE = EXPLORE
    ANALYSIS = ANALYSIS

    def __init__(self, *args, explorations=None):
        super().__init__(*args)
        self.explorations = explorations if explorations is not None else {}

    def _attempt_lines(self, node, attempts, sequence_ids):
        repairs = self._repairs()
        lines = []
        for attempt in attempts:
            action = attempt["action"]
            if action == "Crossover" and attempt.get("reference_id") in self.programs:
                action += f" with an algorithm scoring {score_text(self.programs[attempt['reference_id']]['score'])}"
            outcome = self._outcome(attempt, node, sequence_ids)
            repair = repairs.get(attempt["id"])
            if repair is not None:
                if repair.get("program_id") is not None and repair["program_id"] == attempt.get("program_id"):
                    outcome += "; the repair returned the same failing program"
                else:
                    outcome += "; repaired: " + self._outcome(repair, node, sequence_ids)
            outcome += self._change_outcome(attempt, node)
            design = idea_view(attempt.get("idea")) or "(none)"
            lines.append(f"Attempt {attempt['id']} · {action} · Design: {design}\n  → {outcome}")
        return lines

    def _change_outcome(self, attempt, node):
        """Where an Explore change ended after its development, relative to ``node``."""
        if attempt["action"] != "Explore" or not attempt.get("exploration"):
            return ""
        record = next((e for e in self.explorations.values() if e["proposal_attempt"] == attempt["id"]), None)
        if record is None:  # the open change: developed unless its first version is already better
            first = outcome(attempt, self.attempts, self.programs)
            return "" if first is None or self._improved(first, node) else "; the change is being developed"
        steps = len(record["development_attempts"])
        best = self.programs.get(record["best_id"]) if record.get("best_id") is not None else None
        if not steps or best is None:
            return ""
        return (f"; developed in {steps} further step{'' if steps == 1 else 's'}: best version score "
                f"{score_text(best['score'])} ({verdict(node['score'], best['score'], self.higher_is_better)})")

    def _attempts_section(self, node, attempts, shown, sequence_ids):
        if not attempts or shown == 0:
            return "", []
        listed = attempts[-shown:]
        repairs = self._repairs()
        closed = by_proposal(self.explorations)
        improved = sum(1 for a in attempts if self._improved(
            outcome(a, self.attempts, self.programs, closed, repairs), node))
        header = (f"[Attempts From the Current Algorithm]\n{len(attempts)} attempt{'' if len(attempts) == 1 else 's'} "
                  f"started from the current algorithm; {improved} produced a new algorithm scoring better than it. ")
        header += ("All are listed, oldest first, with their evaluated outcomes." if len(listed) == len(attempts)
                   else f"The most recent {len(listed)} are listed, oldest first, with their evaluated outcomes.")
        return header + "\n\n" + "\n\n".join(self._attempt_lines(node, listed, sequence_ids)), [a["id"] for a in listed]

    def _improved(self, program, node):
        """Whether an attempt's outcome (``selection.outcome``) is a program better than ``node``."""
        return (program is not None and program["id"] != node["id"] and
                verdict(node["score"], program["score"], self.higher_is_better) == "improved")

    def develop(self, current, source, attempts):
        """Make the change an Explore proposal made work in the algorithm it was made to.

        ``source`` is the algorithm the proposal changed and ``attempts`` are the
        development attempts made on the change so far. The algorithm is shown in
        full; the formation starts at the change's first version (a failed first
        version included), and every attempt that left that path is listed.
        """
        sequence = path(current, self.programs)
        ids = [p["id"] for p in sequence]
        if source["id"] not in ids:
            raise ValueError("the current version does not descend from the exploration's source")
        sequence = sequence[ids.index(source["id"]) + 1:]
        sequence_ids = {p["id"] for p in sequence}
        first = self.steps(sequence)[0][0]
        best = max((p for p in self.programs.values() if p["valid"]), key=lambda p: p["fitness"])
        start = (f"[The Algorithm the Change Was Made To]\nAn Explore step changed this algorithm; the first "
                 f"version of the change scored {score_text(first['score'])} "
                 f"({COMPARED[verdict(source['score'], first['score'], self.higher_is_better)]} this algorithm). "
                 f"The best algorithm found so far in this search scores {score_text(best['score'])}.\n"
                 f"{self.measured(source)}\nDesign: {idea_view(source['idea'])}\n"
                 f"```python\n{source['code'].rstrip()}\n```")
        repairs = self._repairs()
        off_path = [a for a in sorted(attempts, key=lambda a: a["id"])
                    if repairs.get(a["id"], a).get("program_id") not in sequence_ids]
        current_section = self._current(current)
        depth = len(self.steps(sequence)) - 1
        count, shown, trims = max(1, min(self.config.history_depth, depth)), len(off_path), []
        while True:
            history, edge_ids = self._formation(sequence, count, title="How the Change Has Been Developed",
                                                origin="first version of the change")
            listed = off_path[len(off_path) - shown:] if shown else []
            tried = ""
            if listed:
                lines = []
                for attempt in listed:
                    origin = self.programs[attempt["parent_id"]]
                    result = self._outcome(attempt, origin, sequence_ids)
                    repair = repairs.get(attempt["id"])
                    if repair is not None:
                        result += ("; the repair returned the same failing program"
                                   if repair.get("program_id") == attempt.get("program_id")
                                   else "; repaired: " + self._outcome(repair, origin, sequence_ids))
                    if attempt.get("program_id") == source["id"] or (repair or {}).get("program_id") == source["id"]:
                        result += " (the algorithm the change was made to)"
                    lines.append(f"Attempt {attempt['id']} · Develop from the version scoring "
                                 f"{score_text(origin['score'])} · Design: {idea_view(attempt.get('idea')) or '(none)'}"
                                 f"\n  → {result}")
                tried = (f"[Other Attempts on This Change]\n{len(off_path)} development "
                         f"attempt{'' if len(off_path) == 1 else 's'} did not produce a version on the path above. "
                         + ("All are listed" if len(listed) == len(off_path)
                            else f"The most recent {len(listed)} are listed")
                         + ", oldest first, with their evaluated outcomes.\n\n" + "\n\n".join(lines))
            sections = self.common + [current_section, start, history, tried, DEVELOP, self.output_format("Develop")]
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
                raise ContextTooLong("Develop prompt with the algorithm and the latest step exceeds context")
