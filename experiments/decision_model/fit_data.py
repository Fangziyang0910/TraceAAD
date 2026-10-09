"""Check full encoded inputs before training, preserving complete programs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from data import fit_record
from train import load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists() or args.max_seq_length < 1024:
        parser.error("choose a new output directory and context of at least 1024")
    from unsloth.models.clef import encode_record
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoded_length = lambda row: len(encode_record(tokenizer, row, max_length=1000000).input_ids)
    rows = load_splits(args.source)
    fitted, report = {}, {}
    for split, values in rows.items():
        fitted[split], lengths, trimmed = [], [], 0
        for row in values:
            fixed = fit_record(row, encoded_length, args.max_seq_length)
            trimmed += fixed["state"] != row["state"]
            assert fixed["state"]["parent"] == row["state"]["parent"]
            assert fixed["state"]["reference"] == row["state"]["reference"]
            lengths.append(encoded_length(fixed))
            fitted[split].append(fixed)
        report[split] = {"rows": len(values), "history_shortened": trimmed, "max_tokens": max(lengths)}
    args.output.mkdir(parents=True)
    for split, values in fitted.items():
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
    metadata = json.loads((args.source / "metadata.json").read_text())
    metadata.update(max_seq_length=args.max_seq_length, encoding=report,
                    tokenizer=args.tokenizer, source_dataset=str(args.source),
                    sha256={f"{s}.jsonl": hashlib.sha256((args.output / f"{s}.jsonl").read_bytes()).hexdigest()
                            for s in fitted})
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if (args.source / "comparisons.jsonl").exists():
        shutil.copyfile(args.source / "comparisons.jsonl", args.output / "comparisons.jsonl")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
