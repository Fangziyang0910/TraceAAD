"""Replace selected historical samples with independent repeated observations."""

import experiments  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path

from traceaad.common.storage import write_json
from .prepare import reward_support
from .train import load_splits


def replace_observations(observed, paired):
    keys = {(r["run_id"], r["state_id"].split('/')[0]) for r in paired}
    retained = [r for r in observed if (r["run_id"], r["state_id"].split('/')[0]) not in keys]
    return retained + paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observed", type=Path)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("enriched", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    metadata = json.loads((args.observed / "metadata.json").read_text())
    for dataset in (args.pilot, args.enriched):
        other = json.loads((dataset / "metadata.json").read_text())
        for key in ("revision", "prompt_policy", "prompt_sources_sha256", "score_scales", "decision_horizon", "trained_questions"):
            if other[key] != metadata[key]:
                raise ValueError(f"different {key} in {dataset}")
    observed, pilot = load_splits(args.observed), load_splits(args.pilot)
    enriched = list(map(json.loads, (args.enriched / "train.jsonl").read_text().splitlines()))
    rows = {split: replace_observations(values, pilot[split] + (enriched if split == "train" else []))
            for split, values in observed.items()}
    args.output.mkdir(parents=True)
    for split, values in rows.items():
        (args.output / f"{split}.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values))
    load_splits(args.output)
    metadata.update(rows={s: len(v) for s, v in rows.items()},
        augmentation="independent repeated labels replace historical samples at the same selected state; training enrichment only",
        source_datasets=[str(p) for p in (args.observed, args.pilot, args.enriched)],
        reward_support={"2": reward_support(rows["train"])},
        sha256={f"{s}.jsonl": hashlib.sha256((args.output / f"{s}.jsonl").read_bytes()).hexdigest() for s in rows})
    write_json(args.output / "metadata.json", metadata)
    print(json.dumps({"rows": metadata["rows"]}, indent=2))


if __name__ == "__main__":
    main()
