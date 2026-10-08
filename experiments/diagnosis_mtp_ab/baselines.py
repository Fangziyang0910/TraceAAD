"""How often does code the model wrote for each related method fail under the current evaluator?

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_mtp_ab.baselines [--per-method 600]

The five methods' TSP runs (formal_tsp_<method>_rep1-3, Qwen3.6-27B, August) record failures
differently or not at all, so every reply is re-read: the last Python block that defines
``select_next_node`` is the program, distinct programs are evaluated once on the training set
under the current protocol, and failures are classified like the TraceAAD replay.
"""

import argparse
import collections
import gzip
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import experiments  # noqa: F401  (evaluation thread limits)
from experiments.diagnosis_mtp_ab.replay import OUT, SHAPE, SOCKET, ROOT
from benchmarks.tasks import INSTANCE_SECONDS, training_task
from traceaad.common.canonical import canonical, key
from traceaad.common.instance_evaluation import InstanceProgramEvaluator

METHODS = ("eoh", "reevo", "mcts_ahd", "pathwise", "calm")
BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
HEADER = "import numpy as np\ndef select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):\n"


def programs(method):
    found = {}
    for calls in sorted((ROOT / "experiments_result" / method / "tsp_construct").glob("formal_tsp_*_rep*/calls.jsonl.gz")):
        for line in gzip.open(calls, "rt"):
            # PathWise names versions select_next_node_v2; ReEvo replies continue the given signature with a body.
            reply = re.sub(r"def select_next_node_v\d+", "def select_next_node", json.loads(line).get("response") or "")
            blocks = [b for b in BLOCK.findall(reply) if "def select_next_node" in b]
            if not blocks and reply.startswith((" ", "\t")) and "def select_next_node" not in reply:
                blocks = [HEADER + reply.split("```")[0]]
            if not blocks:
                continue
            try:
                code = canonical(blocks[-1])
            except (SyntaxError, ValueError):
                found.setdefault("syntax:" + key(blocks[-1]), None)
                continue
            if "import numpy as np" not in code and "np." in code:
                code = "import numpy as np\n" + code  # templates import numpy outside the function
            found.setdefault(key(code), code)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-method", type=int, default=600)
    args = parser.parse_args(argv)
    evaluation = training_task("tsp_construct", condition="traceaad")[0]
    out = OUT / "baselines.jsonl"

    def evaluate(item):
        method, code = item
        if code is None:
            return {"method": method, "status": "syntax_error"}
        result = InstanceProgramEvaluator(evaluation, (730241,), "search", timeout_seconds=INSTANCE_SECONDS,
                                          n_workers=8, scheduler_socket=SOCKET).evaluate(code, key(code))
        failure = result["failure"]
        error = (failure["error"] or "").strip().splitlines()[-1][:200] if failure else None
        return {"method": method, "status": failure["kind"] if failure else "valid", "error": error,
                "shape_error": bool(error and any(s in error for s in SHAPE))}

    jobs = []
    for method in METHODS:
        codes = list(programs(method).values())
        random.Random(0).shuffle(codes)
        jobs += [(method, code) for code in codes[:args.per_method]]
    with ThreadPoolExecutor(12) as pool, out.open("w") as output:
        rows = []
        for row in pool.map(evaluate, jobs):
            rows.append(row)
            output.write(json.dumps(row) + "\n")
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        status = collections.Counter(r["status"] for r in group)
        n = len(group)
        print(f"{method:9s} programs {n} valid {status['valid']/n:.1%} runtime {status['runtime_error']/n:.1%} "
              f"shape/index {sum(r.get('shape_error', False) for r in group)/n:.1%} "
              f"invalid_output {status['invalid_output']/n:.1%} timeout {status['timeout']/n:.1%} "
              f"syntax {status['syntax_error']/n:.1%}")


if __name__ == "__main__":
    main()
