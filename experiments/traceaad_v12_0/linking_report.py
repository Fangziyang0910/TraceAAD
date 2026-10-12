"""How Explore proposals link to earlier implementations of the same computation.

V12.0 runs: summarize the recorded links and the outcomes of linked and unlinked proposals.
Base runs (``--replay``, E0): run the match step offline on every Explore proposal, against
the programs evaluated before it. E0 asks whether proposals name computations that already
have implementations, and whether those implementations are further along than the new
proposal's own first version. ``--examples`` writes matched cases for reading the code.
"""

import argparse
import json
from pathlib import Path
import random
import statistics

from core.llm import generate
from experiments.infra.base import RESULTS_ROOT, build_llm_client
from traceaad.common.history import final_attempt
from traceaad.common.selection import better
from traceaad.common.state import Facts
from traceaad.common.storage import read_json, write_json
from traceaad.v12_0.config import Config
from traceaad.v12_0.linking import choose_linked, normal, parse_match, rank_candidates
from traceaad.v12_0.prompts import match_prompt

OUT = RESULTS_ROOT / "traceaad_v12_0" / "linking"


def proposals(facts):
    return [a for _, a in sorted(facts.attempts.items())
            if a["action"] == "Explore" and a.get("repair_of") is None]


def outcome(facts, attempt):
    """The proposal's own result after its repair, and its start."""
    final = final_attempt(attempt, facts.attempts)
    program = facts.programs.get(final.get("program_id"))
    return (program if final["status"] == "valid" and program and program["valid"] else None,
            facts.programs[attempt["parent_id"]])


def record(facts, attempt, link):
    first, parent = outcome(facts, attempt)
    linked = facts.programs.get(link.get("linked_id")) if link else None
    return {"attempt": attempt["id"], "parent": parent["id"], "design": normal(attempt.get("idea")),
            "candidates": (link or {}).get("candidates", []), "matched": (link or {}).get("matched", []),
            "linked": linked["id"] if linked else None, "shown": (link or {}).get("shown", False),
            "first": first["id"] if first else None, "status": final_attempt(attempt, facts.attempts)["status"],
            "first_beats_parent": bool(first and better(first["fitness"], parent["fitness"])),
            "linked_beats_first": bool(linked and (first is None or better(linked["fitness"], first["fitness"]))),
            "first_beats_linked": bool(linked and first and better(first["fitness"], linked["fitness"])),
            "linked_beats_parent": bool(linked and better(linked["fitness"], parent["fitness"])),
            "copied_linked": bool(linked and attempt.get("program_id") == linked["id"])}


def replay(run_dir, k, llm, embed):
    facts, vectors, rows, tokens = Facts(run_dir), {}, [], [0, 0]

    def vector(text):
        if text not in vectors:
            vectors[text] = embed(text)
        return vectors[text]

    for attempt in proposals(facts):
        design = normal(attempt.get("idea"))
        before = {pid: p for pid, p in facts.programs.items() if pid < attempt["id"]}
        ranked = rank_candidates(design, before, attempt["parent_id"], vector, k) if design else []
        link = {"candidates": [p["id"] for p, _ in ranked], "matched": []}
        if ranked:
            details = generate(llm, match_prompt(design, ranked), max_tokens=Config.match_output_tokens, temperature=0.0)
            usage = details.get("usage") or {}
            tokens[0] += usage.get("prompt_tokens") or 0
            tokens[1] += usage.get("completion_tokens") or 0
            link["matched"] = parse_match(details.get("content", ""), link["candidates"])
            if link["matched"]:
                link["linked_id"] = choose_linked([facts.programs[i] for i in link["matched"]])["id"]
        rows.append(record(facts, attempt, link))
    return rows, {"match_input_tokens": tokens[0], "match_output_tokens": tokens[1]}


def summary(rows):
    def share(subset, key):
        return round(sum(r[key] for r in subset) / len(subset), 3) if subset else None
    linked = [r for r in rows if r["linked"] is not None]
    unlinked = [r for r in rows if r["linked"] is None]
    sizes = [len(r["matched"]) for r in linked]
    return {"proposals": len(rows),
            "with_candidates": sum(bool(r["candidates"]) for r in rows),
            "linked": len(linked), "linked_share": round(len(linked) / len(rows), 3) if rows else None,
            "matched_size_median": statistics.median(sizes) if sizes else None,
            "linked_beats_first": share(linked, "linked_beats_first"),
            "linked_beats_parent": share(linked, "linked_beats_parent"),
            # A linked proposal whose new implementation beats the computation's best existing one.
            "first_beats_linked": {"all": share(linked, "first_beats_linked"),
                                   "shown": share([r for r in linked if r["shown"]], "first_beats_linked"),
                                   "hidden": share([r for r in linked if not r["shown"]], "first_beats_linked")},
            "first_beats_parent": {"linked": share(linked, "first_beats_parent"),
                                   "unlinked": share(unlinked, "first_beats_parent"),
                                   "linked_shown": share([r for r in linked if r["shown"]], "first_beats_parent"),
                                   "linked_hidden": share([r for r in linked if not r["shown"]], "first_beats_parent")},
            "copied_linked": share(linked, "copied_linked")}


def examples(facts, rows, count, seed=0):
    picked = random.Random(seed).sample([r for r in rows if r["linked"]], min(count, sum(bool(r["linked"]) for r in rows)))
    blocks = []
    for r in sorted(picked, key=lambda r: r["attempt"]):
        linked, parent = facts.programs[r["linked"]], facts.programs[r["parent"]]
        first = facts.programs.get(r["first"])
        named = "\n".join(f"- [{i}] {normal(facts.programs[i]['idea'])}" for i in r["matched"])
        blocks.append(
            f"## Attempt {r['attempt']}\n\nProposal Design: {r['design']}\n\nMatched:\n{named}\n\n"
            f"Parent {parent['id']} score {parent['fitness']} · first version "
            f"{first['fitness'] if first else r['status']} · linked {linked['id']} score {linked['fitness']}\n\n"
            f"### Linked program {linked['id']}\n```python\n{linked['code'].rstrip()}\n```\n\n"
            + (f"### First version {first['id']}\n```python\n{first['code'].rstrip()}\n```\n" if first else ""))
    return "\n".join(blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", action="store_true", help="E0: match offline on a base run")
    parser.add_argument("--examples", type=int, default=0, help="write this many linked cases per run")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    llm = embed = None
    for run_dir in args.run_dir:
        facts = Facts(run_dir)
        if args.replay:
            if llm is None:
                from experiments.shinka_evo.embedding import CodeEmbedding
                settings = read_json(run_dir / "run_config.json")["llm"]
                llm = build_llm_client(base_url=settings["base_url"], model=settings["model"],
                                       no_proxy=settings.get("no_proxy", ""), max_tokens=Config.match_output_tokens,
                                       enable_thinking=settings.get("enable_thinking"))
                embed = CodeEmbedding()
            rows, cost = replay(run_dir, Config.match_candidates, llm, embed)
        else:
            rows, cost = [record(facts, a, a.get("link")) for a in proposals(facts)], {}
        name = run_dir.name
        report[name] = {"summary": summary(rows), **cost}
        write_json(args.out / f"{name}{'_replay' if args.replay else ''}.json", {"rows": rows, **report[name]})
        if args.examples:
            (args.out / f"{name}_examples.md").write_text(examples(facts, rows, args.examples), encoding="utf-8")
    print(json.dumps({name: r["summary"] for name, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
