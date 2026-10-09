"""Wait for complete weights and a free GPU, then run the real outcome training."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def weights_complete(folder):
    adapter = folder / "adapter_config.json"
    if adapter.exists():
        base = Path(json.loads(adapter.read_text())["base_model_name_or_path"])
        return (folder / "joint_head.safetensors").is_file() and base != folder and weights_complete(base)
    index = folder / "model.safetensors.index.json"
    if not index.exists() or not (folder / "joint_head.safetensors").exists():
        return False
    files = set(json.loads(index.read_text())["weight_map"].values())
    return all((folder / filename).is_file() for filename in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=18000)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()
    status_path = args.output.parent / (args.output.name + "-status.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
        memory = [(int(a), int(b)) for a, b in (line.split(",") for line in rows.splitlines())]
        gpu, free = max(memory, key=lambda pair: pair[1])
        ready = weights_complete(args.model)
        phase = "waiting_weights" if not ready else "waiting_gpu" if free < args.minimum_free_mib else "starting"
        status_path.write_text(json.dumps({"phase": phase, "gpu": gpu, "free_mib": free,
            "minimum_free_mib": args.minimum_free_mib, "data": str(args.data), "model": str(args.model),
            "checked_at": datetime.now().astimezone().isoformat()}, indent=2))
        if phase == "starting":
            break
        print(phase, "best GPU", gpu, "free MiB", free, flush=True)
        time.sleep(45)
    import os
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
           "OMP_NUM_THREADS": "4", "TORCHINDUCTOR_COMPILE_THREADS": "4"}
    command = [sys.executable, "-u", str(Path(__file__).with_name("train.py")), str(args.data),
               str(args.output), "--model", str(args.model), "--load-in-4bit",
               "--max-seq-length", str(args.max_seq_length), "--learning-rate", "5e-5"]
    completed = subprocess.run(command, env=env)
    status_path.write_text(json.dumps({"phase": "finished" if completed.returncode == 0 else "failed",
                                     "exit_code": completed.returncode, "gpu": gpu}, indent=2))
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
