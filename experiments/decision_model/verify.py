"""Check decision APIs and a BF16 backward/8-bit optimizer step on one visible GPU."""

import argparse
import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from unsloth import FastDecisionModel, DecisionTrainer
    import torch
    import bitsandbytes as bnb

    assert torch.cuda.is_available(), "CUDA PyTorch cannot see a GPU"
    assert torch.cuda.is_bf16_supported(), "The selected GPU does not support BF16"
    assert callable(FastDecisionModel.build_dataset)
    assert callable(FastDecisionModel.predict)
    assert callable(DecisionTrainer.train)
    parameter = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
    optimizer = bnb.optim.AdamW8bit([parameter], lr=1e-4)
    loss = (parameter @ parameter.T).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(parameter.grad).all()
    optimizer.step()
    torch.cuda.synchronize()
    assert torch.isfinite(parameter).all()

    report = {
        "python": platform.python_version(),
        "unsloth_source": json.loads(distribution("unsloth").read_text("direct_url.json") or "{}"),
        "decision_code_sha256": hashlib.sha256(
            Path(FastDecisionModel.from_pretrained.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "packages": {name: version(name) for name in
                     ("unsloth", "unsloth-zoo", "torch", "torchvision", "transformers",
                      "trl", "peft", "bitsandbytes", "triton", "xformers", "modelscope")},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
        "checks": ["decision_api", "bf16_backward", "adamw_8bit_step"],
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
