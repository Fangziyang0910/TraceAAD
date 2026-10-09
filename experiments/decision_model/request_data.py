"""Label the ordinary operator request including its actual automatic Repair."""

import experiments  # noqa: F401
import argparse
import copy
import hashlib
import json
from pathlib import Path

from traceaad.common.state import Facts
from traceaad.common.storage import append_jsonl, write_json
from .data import delivered_request, outcome, questions
from .prepare import reward_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    metadata = json.loads((args.source / "metadata.json").read_text())
    if metadata["horizons"] != [1]:
        parser.error("source must contain individual chronological first-candidate observations")
    rows, report, facts = {}, {}, {}
    for split in ("train", "calibration", "test"):
        rows[split], censored = [], 0
        for original in map(json.loads, (args.source / f"{split}.jsonl").read_text().splitlines()):
            run_id = original["run_id"]
            if run_id not in facts:
                source = args.source / "snapshots" / run_id
                facts[run_id] = (Facts(source), json.loads((source / "run_config.json").read_text()))
            archive, config = facts[run_id]
            aid = original["measurement"]["attempt_id"]
            delivery = delivered_request(archive.attempts[aid], archive.attempts.get(aid + 1),
                                         archive.programs, config["budget"])
            if delivery is None:
                censored += 1
                continue
            row = copy.deepcopy(original)
            state = row["state"]
            state["development_budget_candidates"] = 2
            labels, gain = outcome(state["parent"]["fitness"], state["frontier_score_before"],
                                   delivery["fitnesses"], state["score_scale"])
            row.update(questions=questions(2), gold={"frontier_gain": labels["frontier_gain"],
                       "repair_used": delivery["candidate_cost"] == 2})
            row["measurement"].update(first_candidate_gain=original["measurement"]["normalized_gain"],
                normalized_gain=gain, candidate_cost=delivery["candidate_cost"],
                request_attempt_ids=delivery["attempt_ids"])
            rows[split].append(row)
        if not rows[split]:
            raise ValueError(f"empty {split} after excluding unfinished requests")
        report[split] = {"requests": len(rows[split]), "right_censored_requests": censored}
    args.output.mkdir(parents=True)
    for split, values in rows.items():
        for row in values:
            append_jsonl(args.output / f"{split}.jsonl", row)
    metadata.update(horizons=[2], decision_horizon=2, trained_questions=list(questions(2)),
        decision_unit="ordinary operator request including its automatic Repair",
        selection_objective="expected frontier gain per expected candidate cost",
        rows={split: len(values) for split, values in rows.items()}, request_label_audit=report,
        source_dataset=str(args.source), reward_support={"2": reward_support(rows["train"])},
        sha256={f"{split}.jsonl": hashlib.sha256((args.output / f"{split}.jsonl").read_bytes()).hexdigest()
                for split in rows})
    write_json(args.output / "metadata.json", metadata)
    print(json.dumps({"rows": metadata["rows"], "audit": report, "decision_horizon": 2}, indent=2))


if __name__ == "__main__":
    main()
