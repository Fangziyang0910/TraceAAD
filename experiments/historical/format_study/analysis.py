"""Paired analysis of the Design experiments (part A: Refine arms; part B: Explore card lengths)."""
import collections
import json
import random
import statistics as st
import sys

BOOT = 2000


def load(path):
    try:
        return [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        return []


def arm_of(r):
    return r["order"] if r["order"] == "none" else f"{r['order']}-{r['length']}" if "order" in r else r["length"]


def ranks(records, arm):
    """Rank arms inside each context by score; failures tie at the bottom. Returns {context: {arm: rank}}."""
    by = collections.defaultdict(dict)
    for r in records:
        by[r["context"]][arm(r)] = r.get("score") if r.get("status") == "valid" else None
    out = {}
    for ctx, scores in by.items():
        arms = list(scores)
        ordered = sorted(arms, key=lambda a: (scores[a] is None, -(scores[a] or 0)))
        rank = {}
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and scores[ordered[j + 1]] == scores[ordered[i]]:
                j += 1
            for a in ordered[i:j + 1]:
                rank[a] = (i + j) / 2 + 1
            i = j + 1
        out[ctx] = rank
    return out


def boot_mean(values_by_ctx, rng=random.Random(0)):
    ctxs = list(values_by_ctx)
    if not ctxs:
        return None, None, None
    mean = st.fmean(values_by_ctx[c] for c in ctxs)
    samples = sorted(st.fmean(values_by_ctx[rng.choice(ctxs)] for _ in ctxs) for _ in range(BOOT))
    return mean, samples[int(.025 * BOOT)], samples[int(.975 * BOOT)]


def fmt_ci(t, pct=False):
    m, lo, hi = t
    if m is None:
        return "-"
    f = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.2f}")
    return f"{f(m)} [{f(lo)}, {f(hi)}]"


def summary(records, arm, order):
    rk = ranks(records, arm)
    print(f"{'arm':14s} {'n':>4} {'valid':>6} {'undo/dup':>8} {'improved':>8} {'tokens':>7} {'words':>6}   mean rank within context [95% CI]")
    for a in order:
        rs = [r for r in records if arm(r) == a]
        if not rs:
            continue
        n = len(rs)
        valid = sum(r["status"] == "valid" for r in rs)
        dup = sum(r["status"] in ("undo", "duplicate") for r in rs)
        imp = sum(r["status"] == "valid" and r["score"] > r["parent_fitness"] + 1e-9 for r in rs)
        tok = st.median([r["output_tokens"] for r in rs if r.get("output_tokens")])
        words = st.median([r.get("design_words", 0) for r in rs])
        by_ctx = {c: v[a] for c, v in rk.items() if a in v}
        print(f"{a:14s} {n:4d} {valid/n:6.0%} {dup/n:8.0%} {imp/n:8.0%} {tok:7.0f} {words:6.0f}   {fmt_ci(boot_mean(by_ctx))}")


def paired(records, arm, a, b, label, match=None):
    """Win rate of arm a over arm b in the same context (ties = half), and improvement-rate difference."""
    by = collections.defaultdict(dict)
    for r in records:
        by[r["context"]][arm(r)] = r
    wins, imp = {}, {}
    for ctx, arms in by.items():
        for ka, kb in (match or [(a, b)]):
            if ka in arms and kb in arms:
                ra, rb = arms[ka], arms[kb]
                sa = ra.get("score") if ra["status"] == "valid" else None
                sb = rb.get("score") if rb["status"] == "valid" else None
                w = 0.5 if sa == sb else 1.0 if sb is None or (sa is not None and sa > sb) else 0.0
                ia = ra["status"] == "valid" and ra["score"] > ra["parent_fitness"] + 1e-9
                ib = rb["status"] == "valid" and rb["score"] > rb["parent_fitness"] + 1e-9
                wins[(ctx, ka)] = w
                imp[(ctx, ka)] = ia - ib
    print(f"  {label:34s} win rate {fmt_ci(boot_mean(wins), True):22s} improvement-rate diff {fmt_ci(boot_mean(imp), True)}  (pairs {len(wins)})")


def arm_key(r):
    return r["arm"]


def generic(path, order, by_task=False):
    """Paired tables for the Qwen rounds (records carry an 'arm' and optionally an 'action')."""
    rs = load(path)
    for action in sorted({r.get("action", "Refine") for r in rs}):
        sub = [r for r in rs if r.get("action", "Refine") == action]
        print(f"\n===== {action}: contexts {len({r['context'] for r in sub})}, generations {len(sub)}")
        summary(sub, arm_key, [a for a in order if any(r["arm"] == a for r in sub)])
        if by_task:
            for task in sorted({r["task"] for r in sub}):
                print(f"  -- {task}")
                summary([r for r in sub if r["task"] == task], arm_key, [a for a in order if any(r["arm"] == a for r in sub)])
        reference = "few" if any(r["arm"] == "few" for r in sub) else "concise_design"
        print(f"Paired vs {reference} (same context):")
        for a in order:
            if a != reference and any(r["arm"] == a for r in sub):
                paired(sub, arm_key, a, reference, f"{a} vs {reference}")


def main(root="experiments_result/format_study/deepseek_length_order"):
    a = load(f"{root}/part_a.jsonl")
    if a:
        print(f"=== Part A: Refine, {len({r['context'] for r in a})} contexts, {len(a)} generations")
        order = ["none"] + [f"{o}-{l}" for o in ("first", "after") for l in ("short", "medium", "long")]
        summary(a, arm_of, order)
        for task in sorted({r["task"] for r in a}):
            print(f"  -- {task}")
            summary([r for r in a if r["task"] == task], arm_of, order)
        print("Paired comparisons (same context):")
        paired(a, arm_of, None, None, "first vs after (all lengths)",
               match=[(f"first-{l}", f"after-{l}") for l in ("short", "medium", "long")])
        for l in ("short", "medium", "long"):
            paired(a, arm_of, f"first-{l}", f"after-{l}", f"first vs after ({l})")
        for x, y in (("medium", "short"), ("long", "medium"), ("long", "short")):
            paired(a, arm_of, None, None, f"{x} vs {y} (both orders)",
                   match=[(f"{o}-{x}", f"{o}-{y}") for o in ("first", "after")])
        paired(a, arm_of, "first-medium", "none", "first-medium vs code only")
        paired(a, arm_of, None, None, "any Design vs code only",
               match=[(f"{o}-{l}", "none") for o in ("first", "after") for l in ("short", "medium", "long")])
    b = load(f"{root}/part_b.jsonl")
    if b:
        print(f"\n=== Part B: Explore with reference cards, {len({r['context'] for r in b})} contexts, {len(b)} generations")
        for l in ("short", "medium", "long"):
            rs = [r for r in b if r["length"] == l]
            if rs:
                print(f"  {l}: card words median {st.median(r['card_words'] for r in rs):.0f}")
        summary(b, lambda r: r["length"], ["short", "medium", "long"])
        for x, y in (("medium", "short"), ("long", "medium"), ("long", "short")):
            paired(b, lambda r: r["length"], x, y, f"cards {x} vs {y}")


QWEN_ORDER = ["concise_design", "long_idea", "brief_analysis", "free_analysis", "code_only", "native_thinking",
              "design_only", "one_sentence", "few", "paragraph", "generic", "options", "target_options",
              "few_d25", "few_d150"]

if __name__ == "__main__":
    # analysis.py deepseek [root] | analysis.py qwen FILE.jsonl [bytask]
    if len(sys.argv) > 1 and sys.argv[1] == "qwen":
        generic(sys.argv[2], QWEN_ORDER, by_task="bytask" in sys.argv)
    else:
        main(*sys.argv[2:3])
