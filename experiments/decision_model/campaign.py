"""Finish the authorized V10.23 data, training and calibration comparisons.

Run locally in tmux. It never starts full searches or reads decision-test metrics.
Additional collection must already have a frozen, preflighted manifest.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
S3 = "/home/fzy/code/traceaad-decision-data/campaign_20261009"
S2 = "/home/fzy/code/traceaad-decision"
TRAIN_PYTHON = "/home/fzy/venvs/traceaad-decision/bin/python"
PROJECT_PYTHON = "/home/fzy/code/LLM4AD/TraceAAD/.venv/bin/python"


def command(args):
    subprocess.run([str(x) for x in args], check=True)


def remote(host, source):
    interpreter = TRAIN_PYTHON if host == "B3-server2" else PROJECT_PYTHON
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, interpreter + " -"],
                            input=source, text=True, capture_output=True, timeout=45, check=True)
    return json.loads(result.stdout)


def tmux_job(host, name, cmd):
    return remote(host, "import subprocess,json\n"
        f"name={name!r}\n"
        "exists=subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0\n"
        f"if not exists: subprocess.run(['tmux','new-session','-d','-s',name,{cmd!r}],check=True)\n"
        "print(json.dumps({'already_running':exists}))\n")


def wait_file(host, path, session):
    while True:
        result = remote(host, "from pathlib import Path\nimport subprocess,json\n"
            f"ready=Path({path!r}).is_file()\n"
            f"running=subprocess.run(['tmux','has-session','-t',{session!r}],capture_output=True).returncode==0\n"
            "print(json.dumps({'ready':ready,'running':running}))\n")
        if result["ready"] and not result["running"]:
            return
        if not result["ready"] and not result["running"]:
            raise RuntimeError(f"{session} stopped before producing {path}; inspect its log")
        time.sleep(45)


def gpu_command(gpu, arguments, log):
    assert gpu in (5, 6)
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "HF_HUB_OFFLINE=1", "OMP_NUM_THREADS=4", "TORCHINDUCTOR_COMPILE_THREADS=2"]
    return "cd " + shlex.quote(S2) + " && " + shlex.join(env + [TRAIN_PYTHON, "-u"] + arguments) + " >" + shlex.quote(S2 + "/logs/" + log) + " 2>&1"


def require_free_gpus():
    result = remote("B3-server2", "import subprocess,json\n"
        "free={i:int(subprocess.check_output(['nvidia-smi','-i',str(i),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for i in (5,6)}\n"
        "print(json.dumps(free))\n")
    if any(value < 18000 for value in result.values()):
        raise RuntimeError(f"authorized GPUs are currently occupied: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    status = args.root / "pipeline-status.json"

    def phase(name, **details):
        status.write_text(json.dumps({"phase": name, "updated_at": datetime.now().astimezone().isoformat(),
                                     "test_metrics_used": False, **details}, indent=2))
        print(name, flush=True)

    def export(dataset, collection, name, training_only=False):
        output, session = S3 + "/" + name, "traceaad-export-" + name
        existing = remote("B3-server3", f"import json;from pathlib import Path;print(json.dumps(Path({output + '/metadata.json'!r}).is_file()))")
        if not existing:
            cmd = "cd " + shlex.quote(S3 + "/analysis-runtime-v1023") + " && " + shlex.join(
                [PROJECT_PYTHON, "-u", "-m", "experiments.decision_model.export_replays", S3 + "/" + dataset,
                 S3 + "/" + collection, output, "--request-only"] + (["--splits", "train"] if training_only else []))
            tmux_job("B3-server3", session, cmd + " >" + shlex.quote(S3 + "/logs/" + name + ".log") + " 2>&1")
            wait_file("B3-server3", output + "/metadata.json", session)
        local = args.root / name
        command(["rsync", "-a", "B3-server3:" + output + "/", str(local) + "/"])
        command(["rsync", "-aH", "B3-server3:" + S3 + "/" + collection + "/", str(args.root / collection) + "/"])
        return local

    def fit(local, name):
        command(["rsync", "-a", str(local) + "/", "B3-server2:" + S2 + "/data/" + name + "-raw/"])
        session = "traceaad-fit-" + name
        output = S2 + "/data/" + name + "-fit"
        existing = remote("B3-server2", f"import json;from pathlib import Path;print(json.dumps(Path({output + '/metadata.json'!r}).is_file()))")
        if not existing:
            cmd = gpu_command(6, ["fit_data.py", "data/" + name + "-raw", "data/" + name + "-fit",
                "--tokenizer", "runs/aad-outcome-v1-warm-server2-seed3407/checkpoints/checkpoint-292", "--max-seq-length", "8192"], name + "-fit.log")
            tmux_job("B3-server2", session, cmd)
            wait_file("B3-server2", output + "/metadata.json", session)
        command(["rsync", "-a", "B3-server2:" + output + "/", str(args.root / (name + "-fit")) + "/"])
        return "data/" + name + "-fit"

    def evaluate(stage, training_data, models):
        require_free_gpus()
        started = []
        for index, (label, model) in enumerate(models.items()):
            gpu = 6 if index == 0 else 5
            if index > 1:
                for output, session in started:
                    wait_file("B3-server2", output, session)
                require_free_gpus()
            name = stage + "-" + label
            output = S2 + "/runs/" + name + "-calibration.json"
            session = "traceaad-eval-" + name
            cmd = gpu_command(gpu, ["evaluate_policy.py", model, "data/paired-request-fit", output,
                "--training-data", training_data, "--split", "calibration", "--load-in-4bit"], name + "-calibration.log")
            existing = remote("B3-server2", f"import json;from pathlib import Path;print(json.dumps(Path({output!r}).is_file()))")
            if not existing:
                tmux_job("B3-server2", session, cmd)
            started.append((output, session))
        reports = {}
        for (label, _), (output, session) in zip(models.items(), started):
            wait_file("B3-server2", output, session)
            target = args.root / (stage + "-" + label + "-calibration.json")
            command(["scp", "B3-server2:" + output, target])
            report = json.loads(target.read_text())
            assert report["split"] == "calibration" and report["horizon"] == 2 and report["states"] == 24
            reports[label] = {k: report[k] for k in ("equal_task_macro_gain_per_candidate", "offline_improvement")}
        return reports

    try:
        phase("waiting_for_request_training_and_fixed_pilot")
        for gpu, label in ((6, "warm"), (5, "base")):
            wait_file("B3-server2", S2 + f"/runs/request-v1023-{label}-seed3407/metrics.json",
                      f"traceaad-request-{label}-gpu{gpu}-20261009")
        wait_file("B3-server3", S3 + "/pilot-v1023/complete.json", "traceaad-decision-pilot-v1023-20261009")
        phase("exporting_same_state_request_outcomes")
        pilot = export("bootstrap", "pilot-v1023", "paired-request-raw")
        fit(pilot, "paired-request")
        models = {label: f"runs/request-v1023-{label}-seed3407/adapter" for label in ("warm", "base")}
        initial = evaluate("request-observed", "data/current-v1023-requests-fit", models | {"released": "models/clef-flash"})
        require_free_gpus()
        service_session = "traceaad-request-http-check-20261009"
        service_report = S2 + "/runs/request-http-check.json"
        service_cmd = gpu_command(5, ["check_service.py", models["warm"], "data/current-v1023-requests-fit/metadata.json",
            "data/paired-request-fit", service_report], "request-http-check.log")
        tmux_job("B3-server2", service_session, service_cmd)
        wait_file("B3-server2", service_report, service_session)
        command(["scp", "B3-server2:" + service_report, args.root / "request-http-check.json"])
        phase("waiting_for_training_enrichment", initial_calibration=initial)
        wait_file("B3-server3", S3 + "/enriched-v1023/complete.json", "traceaad-decision-enriched-20261009")
        enriched = export("enriched-v1023-training-states", "enriched-v1023", "enriched-request-raw", training_only=True)
        augmented = args.root / "augmented-request-raw"
        if not augmented.exists():
            command([sys.executable, "-m", "experiments.decision_model.augment_data", args.root / "current-v1023-requests",
                     pilot, enriched, augmented])
        training_data = fit(augmented, "augmented-request")
        require_free_gpus()
        phase("training_independent_comparative_labels", initial_calibration=initial)
        final_models = {}
        for gpu, label in ((6, "warm"), (5, "base")):
            model = "runs/aad-outcome-v1-warm-server2-seed3407/checkpoints/checkpoint-292" if label == "warm" else "models/clef-flash"
            name = f"request-paired-{label}-seed3407"
            output, session = S2 + "/runs/" + name, "traceaad-train-" + name
            cmd = gpu_command(gpu, ["train.py", training_data, output, "--model", model, "--load-in-4bit",
                "--max-seq-length", "8192", "--learning-rate", "5e-5", "--seed", "3407"], name + ".log")
            tmux_job("B3-server2", session, cmd)
            final_models[label] = output + "/adapter"
        for label in final_models:
            name = f"request-paired-{label}-seed3407"
            wait_file("B3-server2", S2 + "/runs/" + name + "/metrics.json", "traceaad-train-" + name)
        final = evaluate("request-paired", training_data, final_models)
        phase("calibration_results_ready_for_research_interpretation", initial_calibration=initial,
              paired_calibration=final, full_search_started=False, decision_test_evaluated=False)
    except Exception as exc:
        phase("needs_attention", error=str(exc))
        raise


if __name__ == "__main__":
    main()
