"""Recompute two pre-incident train scores through the shared evaluator.

Diagnostic results are separate from every search journal and budget.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    import __main__
    import experiments.traceaad_v10_12.run as entry
    from experiments.infra.evaluate import SecureEvaluator, CVRPACOEvaluation, OPACOEvaluation

    assert Path(entry.__file__).resolve().is_relative_to(args.runtime.resolve())
    # Exercise the same spawn re-import as the real search entry, not just an
    # import of this diagnostic script.
    __main__.__spec__ = entry.__spec__
    records = []
    for task, evaluation_class in [("cvrp_aco", CVRPACOEvaluation), ("op_aco", OPACOEvaluation)]:
        state_path = next((args.stage / task).glob("*rep1/tree_state.json"))
        state = json.loads(state_path.read_text())
        config = json.loads((args.backup / task / state_path.parent.name / "run_config.json").read_text())
        best = max(state["nodes"], key=lambda node: node["fitness"])
        evaluator = SecureEvaluator(evaluation_class(**config["task_eval"]))
        outcome, seconds = evaluator.evaluate_program_record_time_with_details(best["code"])
        row = dict(task=task, run_dir=str(state_path.parent), diagnostic=True,
                   budget_used=state["budget_used"], node_id=best["id"],
                   train_artifact_score=best["fitness"], train_recomputed_score=outcome.result,
                   train_eval_seconds=seconds, task_eval=config["task_eval"],
                   failure_kind=outcome.failure_kind, error_type=outcome.error_type,
                   error=outcome.error)
        records.append(row)
        args.output.write_text(json.dumps({
            "created_at": datetime.now().astimezone().isoformat(),
            "runtime": str(args.runtime), "purpose": "pre-recovery training sanity; not held-out",
            "results": records,
        }, indent=2) + "\n")
        assert outcome.result is not None, row
        assert abs(float(outcome.result) - best["fitness"]) < 1e-9, row
        print(task, "score", outcome.result, "matches archived score; seconds", round(seconds, 3), flush=True)


if __name__ == "__main__":
    main()
