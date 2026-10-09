"""Train a Qwen3.5-9B decision model on one CUDA GPU with BF16 LoRA.

Use a separate Python 3.12 environment with the pinned Unsloth release source,
Transformers 5, and CUDA PyTorch. Keep these out of TraceAAD's search environment.

Usage, from the repository root:
    CUDA_VISIBLE_DEVICES=0 python experiments/decision_model/train.py DATA OUTPUT
    CUDA_VISIBLE_DEVICES=0 python experiments/decision_model/train.py DATA SMOKE --max-steps 10

DATA contains train.jsonl, calibration.jsonl and test.jsonl. Each row has:
    {"run_id": "task/batch/rep0", "state": "code and past results",
     "questions": {"operator": {"type": "choice", "instructions": "Choose the next step",
        "criteria": {"Refine": "Keep the core computation and improve it",
                     "Explore": "Change the core computation",
                     "Crossover": "Borrow computation from the supplied reference"}}},
     "gold": {"operator": "Refine"}}

This is a format example, not a labeled observation. Gold labels must come from
measured comparisons. Split entire search runs before writing the three files.
Short smoke runs validate execution, not decision quality. Inputs that would be
truncated are rejected; shorten the state deliberately or raise --max-seq-length.
"""

import argparse
import hashlib
import json
from pathlib import Path


def load_splits(directory):
    splits, seen_runs = {}, set()
    for name in ("train", "calibration", "test"):
        path = directory / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if not rows:
            raise ValueError(f"{path}: empty split")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("run_id"), str) or not row["run_id"].strip():
                raise ValueError(f"{path}: each row needs a nonempty run_id")
            if not isinstance(row.get("state"), (str, dict)) or not row["state"]:
                raise ValueError(f"{path}: each row needs a nonempty state")
            if not isinstance(row.get("questions"), dict) or not row["questions"]:
                raise ValueError(f"{path}: each row needs questions")
            if not isinstance(row.get("gold"), dict) or set(row["questions"]) != set(row["gold"]):
                raise ValueError(f"{path}: gold must answer every question")
        runs = {row["run_id"] for row in rows}
        if runs & seen_runs:
            raise ValueError(f"{path}: search runs occur in multiple splits: {sorted(runs & seen_runs)}")
        splits[name] = rows
        seen_runs.update(runs)
    return splits


def check_encoding(name, items, report):
    if not items or report["skipped"] or report["truncated"]:
        raise ValueError(f"{name}: refusing empty, skipped or truncated data: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="Cloudflare/clef-flash")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.max_seq_length <= 0 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-seq-length must be positive; max-steps must be positive or -1")
    if not 0 < args.learning_rate < 1:
        parser.error("learning-rate must be between zero and one")
    if args.output_dir.exists() and args.resume is None and not args.check_data:
        parser.error("output_dir already exists; choose a new run directory")
    rows = load_splits(args.data_dir)
    if args.check_data:
        print(json.dumps({name: {"rows": len(split), "runs": len({row['run_id'] for row in split})}
                          for name, split in rows.items()}, indent=2))
        return

    from unsloth import FastDecisionModel, DecisionTrainer
    import torch
    from transformers import TrainingArguments

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use CUDA PyTorch and select one GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires BF16 support")
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16, load_in_4bit=args.load_in_4bit, full_finetuning=False,
    )
    if hasattr(model.encoder, "peft_config"):
        FastDecisionModel.for_training(model)
    else:
        model = FastDecisionModel.get_peft_model(
            model, r=16, lora_alpha=16, lora_dropout=0,
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )
    items = {}
    for name, split in rows.items():
        items[name], report = FastDecisionModel.build_dataset(split, tokenizer, model)
        check_encoding(name, items[name], report)
        print(name, report, flush=True)

    trainer = DecisionTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=items["train"], eval_dataset=items["calibration"],
        head_learning_rate=1e-4,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"),
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=16, num_train_epochs=2,
            max_steps=args.max_steps, learning_rate=args.learning_rate,
            lr_scheduler_type="cosine", warmup_steps=0.05, weight_decay=0.01,
            bf16=True, fp16=False, optim="adamw_8bit", eval_strategy="epoch",
            logging_steps=1, save_strategy="steps", save_steps=50, save_total_limit=2,
            report_to="none", seed=args.seed,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "data_sha256": {f"{name}.jsonl": hashlib.sha256((args.data_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                                  for name in rows},
                  "model_config": model.encoder.config.to_dict()}
    config_path = args.output_dir / "run_config.json"
    if args.resume and config_path.exists():
        old = json.loads(config_path.read_text())
        if old["data_sha256"] != run_config["data_sha256"]:
            raise ValueError("cannot resume with changed data")
    else:
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str))
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    calibration = FastDecisionModel.calibrate(model, tokenizer, items["calibration"], batch_size=1)
    model.save_pretrained(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))
    metrics = {"calibration": calibration,
               "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
               "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
    if args.evaluate_test:
        metrics["test"] = FastDecisionModel.evaluate(model, tokenizer, items["test"], batch_size=1)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
