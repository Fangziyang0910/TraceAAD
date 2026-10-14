"""Whether Explore versions are tests of their Designs, and what completing them gives.

A first version is a test when it carries out its proposal's Design and keeps the rest of its start.
Earlier runs (``--replay``, C1): read every Explore first version offline, and record what the search did
with it afterwards. V12.1 runs: summarize the recorded readings and completions; in the randomized arm,
explorations that needed completion are split by whether they got it. ``--examples`` writes cases of each
reading with their code, for reading by hand.
"""

import argparse
import json
from pathlib import Path
import random

from core.llm import generate
from experiments.infra.base import RESULTS_ROOT, build_llm_client
from traceaad.common.history import final_attempt
from traceaad.common.selection import OPERATORS, better
from traceaad.common.state import Facts
from traceaad.common.storage import read_json, rows as saved_rows, write_json
from traceaad.v12_0.linking import normal
from traceaad.v12_1.config import Config
from traceaad.v12_1.verification import contain_prompt, parse_dropped, parse_verdict, verify_prompt

OUT = RESULTS_ROOT / "traceaad_v12_1" / "verification"


def task_text(run_dir):
    """The task section of the run's prompts: the first prompt up to its score line."""
    for name in ("calls.jsonl.gz", "calls.jsonl"):
        for call in saved_rows(Path(run_dir) / name):
            return call["prompt"].split("\n\nScore: ")[0]
    raise ValueError(f"{run_dir}: no saved calls")


def explorations(facts):
    """Each Explore proposal with its first valid version (after its repair, else None) and its start."""
    for attempt in sorted(facts.attempts.values(), key=lambda a: a["id"]):
        if attempt["action"] != "Explore" or attempt.get("repair_of") is not None:
            continue
        final = final_attempt(attempt, facts.attempts)
        first = facts.programs[final["program_id"]] if final["status"] == "valid" else None
        yield attempt, first, facts.programs[attempt["parent_id"]]


def later(facts, program_id):
    """Operator attempts started from a program, and proposals linked to it."""
    started = sum(1 for a in facts.attempts.values() if a["parent_id"] == program_id and a["action"] in OPERATORS)
    linked = sum(1 for a in facts.attempts.values() if (a.get("link") or {}).get("linked_id") == program_id)
    return started, linked


def reading(verdict):
    """test: carries out the Design and keeps the rest of the start; missing: does not carry it out;
    dropped: carries it out but drops a computation of the start; None: not judged."""
    if verdict.get("implemented") is False:
        return "missing"
    if verdict.get("kept") is False:
        return "dropped"
    return "test" if verdict.get("implemented") and verdict.get("kept") else None


def row(facts, attempt, first, start, verdict):
    started, linked = later(facts, first["id"])
    return {"attempt": attempt["id"], "start": start["id"], "first": first["id"],
            "design": normal(attempt.get("idea")), "reading": reading(verdict),
            "missing": verdict.get("missing"), "dropped": verdict.get("dropped"),
            "first_beats_start": better(first["fitness"], start["fitness"]),
            "started_from_first": started, "linked_to_first": linked}


def replay(facts, task, llm):
    """C1: verify every Explore first version offline against its Design and its start, as V12.1 would."""
    result, tokens = [], [0, 0]
    for attempt, first, start in explorations(facts):
        design = normal(attempt.get("idea"))
        if first is None or not design:
            continue
        def ask(prompt):
            details = generate(llm, prompt, max_tokens=Config.verify_output_tokens, temperature=0.0)
            usage = details.get("usage") or {}
            tokens[0] += usage.get("prompt_tokens") or 0
            tokens[1] += usage.get("completion_tokens") or 0
            return details.get("content", "")
        verdict = dict(zip(("implemented", "missing"), parse_verdict(ask(verify_prompt(task, design, first["code"])))))
        if verdict["implemented"] is not False:
            verdict.update(zip(("kept", "dropped"),
                               parse_dropped(ask(contain_prompt(task, design, start["code"], first["code"])))))
        result.append(row(facts, attempt, first, start, verdict))
    return result, {"verify_input_tokens": tokens[0], "verify_output_tokens": tokens[1]}


def recorded(facts):
    """V12.1: the saved verdicts, and each exploration's completions."""
    result = []
    for attempt, first, start in explorations(facts):
        if first is None:
            continue
        r = row(facts, attempt, first, start, facts.verdicts.get(first["id"], {}))
        exploration = facts.explorations.get((attempt.get("exploration") or {}).get("id"))
        if exploration:
            best = facts.programs.get(exploration["best_id"])
            r.update(completed=exploration["developed"], completion_steps=len(exploration["development_attempts"]),
                     completion_draw=exploration["completion_draw"],
                     ends_test=exploration["versions"][-1]["tests"], test_improved=exploration["test_improved"],
                     best_beats_start=bool(best and better(best["fitness"], start["fitness"])),
                     followed=sum(later(facts, v["id"])[0] for v in exploration["versions"]))
        result.append(r)
    return result


def summary(rows):
    def share(subset, test):
        return round(sum(map(test, subset)) / len(subset), 3) if subset else None

    def readings(subset):
        return {kind: share(subset, lambda r: r["reading"] == kind) for kind in ("test", "missing", "dropped")}
    judged = [r for r in rows if r["reading"] is not None]
    worse = [r for r in judged if not r["first_beats_start"]]
    result = {"first_versions": len(rows), "judged": len(judged), "readings": readings(judged),
              "readings_if_worse": readings(worse),
              "readings_if_better": readings([r for r in judged if r["first_beats_start"]]),
              # Whether the search ever started from a first version that did not beat its start.
              "worse_first_followed": {
                  "test": share([r for r in worse if r["reading"] == "test"], lambda r: r["started_from_first"] > 0),
                  "not_test": share([r for r in worse if r["reading"] != "test"],
                                    lambda r: r["started_from_first"] > 0)}}
    needing = [r for r in worse if r["reading"] != "test" and "completed" in r]
    if needing:
        def outcomes(subset):
            return {"explorations": len(subset), "ends_test": share(subset, lambda r: r["ends_test"]),
                    "test_improved": share(subset, lambda r: r["test_improved"]),
                    "best_beats_start": share(subset, lambda r: r["best_beats_start"]),
                    "followed": share(subset, lambda r: r["followed"] > 0)}
        completed = [r for r in needing if r["completed"]]
        result["completion"] = {"completed": outcomes(completed),
                                "not_completed": outcomes([r for r in needing if not r["completed"]]),
                                # Completed in place (missing) or in the start itself (dropped).
                                "completed_by_reading": {kind: outcomes([r for r in completed if r["reading"] == kind])
                                                         for kind in ("missing", "dropped")}}
    return result


def examples(facts, rows, count, seed=0):
    """``count`` cases of each reading, with the start's and the first version's code."""
    blocks, rng = [], random.Random(seed)
    for kind in ("missing", "dropped", "test"):
        cases = [r for r in rows if r["reading"] == kind]
        for r in sorted(rng.sample(cases, min(count, len(cases))), key=lambda r: r["attempt"]):
            start, first = facts.programs[r["start"]], facts.programs[r["first"]]
            label = {"missing": f"missing: {r['missing']}", "dropped": f"dropped: {r['dropped']}",
                     "test": "test"}[kind]
            blocks.append(
                f"## Attempt {r['attempt']} · {label}\n\n"
                f"Design: {r['design']}\n\nStart {start['id']} score {start['fitness']} · first version "
                f"{first['id']} score {first['fitness']}\n\n"
                f"### Start {start['id']}\n```python\n{start['code'].rstrip()}\n```\n\n"
                f"### First version {first['id']}\n```python\n{first['code'].rstrip()}\n```\n")
    return "\n".join(blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", action="store_true", help="C1: verify offline on an earlier run")
    parser.add_argument("--examples", type=int, default=0, help="write this many cases of each reading per run")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report, llm = {}, None
    for run_dir in args.run_dir:
        facts = Facts(run_dir)
        if args.replay:
            if llm is None:
                settings = read_json(run_dir / "run_config.json")["llm"]
                llm = build_llm_client(base_url=settings["base_url"], model=settings["model"],
                                       no_proxy=settings.get("no_proxy", ""), max_tokens=Config.verify_output_tokens,
                                       enable_thinking=settings.get("enable_thinking"))
            rows, cost = replay(facts, task_text(run_dir), llm)
        else:
            rows, cost = recorded(facts), {}
        name = run_dir.name
        report[name] = {"summary": summary(rows), **cost}
        write_json(args.out / f"{name}{'_replay' if args.replay else ''}.json", {"rows": rows, **report[name]})
        if args.examples:
            (args.out / f"{name}_examples.md").write_text(examples(facts, rows, args.examples), encoding="utf-8")
    print(json.dumps({name: r["summary"] for name, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
