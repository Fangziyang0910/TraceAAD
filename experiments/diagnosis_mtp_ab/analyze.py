"""Summarize the MTP A/B replay: outcome rates per endpoint and paired differences per prompt."""

import collections
import json
import math
import statistics

from experiments.diagnosis_mtp_ab.replay import OUT


def main():
    rows = [json.loads(line) for line in (OUT / "results.jsonl").open()]
    print(f"{len(rows)} generations")
    for arm in ("mtp", "no_mtp"):
        group = [r for r in rows if r["arm"] == arm]
        n = len(group)
        status = collections.Counter(r["status"] for r in group)
        shape = sum(r.get("shape_error", False) for r in group)
        tokens = [r["output_tokens"] for r in group if r.get("output_tokens")]
        print(f"{arm:7s} n={n} valid {status['valid']/n:.1%} runtime {status['runtime_error']/n:.1%} "
              f"(shape/index {shape/n:.1%}) invalid_output {status['invalid_output']/n:.1%} "
              f"timeout {status['timeout']/n:.1%} duplicate {status['duplicate']/n:.1%} "
              f"unusable {(status['delivery_failed'] + status['invalid_source'])/n:.1%} "
              f"median output tokens {statistics.median(tokens):.0f}")
        for action in ("Refine", "Explore", "Crossover"):
            part = [r for r in group if r["action"] == action]
            bad = sum(r["status"] in ("runtime_error", "invalid_output") for r in part)
            print(f"    {action:9s} n={len(part)} runtime/invalid {bad/len(part):.1%} "
                  f"duplicate {sum(r['status'] == 'duplicate' for r in part)/len(part):.1%}")
    # Paired by prompt and sample: does one endpoint fail where the other does not?
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[r["index"], r["sample"]][r["arm"]] = r["status"] in ("runtime_error", "invalid_output")
    only = collections.Counter()
    for pair in pairs.values():
        if len(pair) == 2 and pair["mtp"] != pair["no_mtp"]:
            only["mtp" if pair["mtp"] else "no_mtp"] += 1
    b, c = only["mtp"], only["no_mtp"]
    p = 2 * sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c) if b + c else 1.0
    print(f"discordant pairs: errors only with MTP {b}, only without {c}; exact two-sided p = {min(p, 1):.3f}")


if __name__ == "__main__":
    main()
