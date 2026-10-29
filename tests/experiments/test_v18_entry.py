import json

import pytest

from benchmarks.tasks import CO_TASKS
from experiments.traceaad_v18.run import main


@pytest.mark.parametrize("task", CO_TASKS)
def test_v18_entry_uses_shared_tasks_and_full_policy(task, capsys):
    main(["--task", task, "--budget", "8", "--roots", "2", "--eval-workers", "2", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["method"] == "traceaad_v18" and plan["config"]["roots"] == 2
    assert plan["config"]["first_checkpoint"] == 5
    assert plan["config"]["lds"] is plan["config"]["same_age"] is True
    assert plan["evaluation_execution"]["function_budget_scope"] == "mean_instance"
    assert plan["final_selection"] == "training"


@pytest.mark.parametrize("task", CO_TASKS)
def test_v18_prompts_keep_the_contract_without_private_evaluation_conditions(task, tmp_path):
    from benchmarks.tasks import training_task
    from tests.support import TokenLLM
    from traceaad.v18 import Config, TraceAADV18

    evaluation, _ = training_task(task, condition="traceaad")
    method = TraceAADV18(task=task, evaluation=evaluation, llm=TokenLLM(), run_dir=tmp_path,
                        config=Config(eval_timeout_seconds=120))
    prompt = method.prompts.build("Init")["prompt"]
    assert "[Function Contract]" in prompt and "[Evaluation Conditions]" not in prompt
    assert not any(s in prompt for s in ("16 training", "120-second", "CPU time", "must not exceed"))
    assert "supplied DSATUR" not in prompt and "supplied Giffler-Thompson" not in prompt


def test_v18_launch_roundtrips_every_policy_option(capsys):
    from experiments.traceaad_v18.launch_host import main as launch
    launch(["--batch", "v18_preview", "--suite", "co6", "--backends", "server3",
            "--eval-workers", "8", "--no-lds", "--no-same-age", "--no-early-first",
            "--max-seconds", "600", "--context-tokens", "49152", "--output-tokens", "8192"])
    batch = json.loads(capsys.readouterr().out)
    assert batch["method"] == "traceaad_v18" and len(batch["plan"]) == 18
    for item in batch["plan"]:
        assert item["command"][1:3] == ["-m", "experiments.traceaad_v18.run"]
        assert "--no-lds" in item["command"] and "--no-same-age" in item["command"]
        main([*item["command"][3:], "--dry-run"])
        single = json.loads(capsys.readouterr().out)
        for key,value in batch["search_policy"].items():
            if key not in {"seed", "eval_timeout_seconds"}:
                assert single["config"][key] == value, key
        assert single["config"]["seed"] == item["repeat"] - 1
