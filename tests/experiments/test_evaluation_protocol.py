import time

from core import Evaluation, SecureEvaluator


class Spin(Evaluation):
    def __init__(self, timeout):
        super().__init__(template_program="def score(x):\n    pass", timeout_seconds=timeout)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


def evaluate(evaluation, body):
    program = f"import time\ndef score(x):\n{body}\n    return 1.0\n"
    return SecureEvaluator(evaluation).evaluate_program_with_details(program)


def test_timeout_is_wall_clock_and_cpu_time_is_recorded():
    outcome = evaluate(Spin(0.5), "    time.sleep(2)")
    assert outcome.failure_kind == "timeout" and outcome.error == "evaluation exceeded 0.5s"
    outcome = evaluate(Spin(5), "    end = time.process_time() + 0.3\n    while time.process_time() < end: pass")
    assert outcome.result == 1.0 and 0.3 <= outcome.cpu_seconds < 2


def test_obp_program_state_does_not_leak_across_instances():
    from benchmarks.generated_data_config import get_generated_task_kwargs
    from benchmarks.online_bin_packing import OBPEvaluation

    program = ("import numpy as np\n_seen = [0]\n"
               "def priority(item, bins):\n"
               "    _seen[0] += 1\n"
               "    return -bins if _seen[0] < 3000 else bins\n")
    evaluation = OBPEvaluation(**get_generated_task_kwargs("online_bin_packing", "train"))
    names = list(evaluation._datasets)
    scores = []
    for order in (names, names[::-1]):
        evaluation._datasets = {name: evaluation._datasets[name] for name in order}
        namespace = {}
        exec(program, namespace)
        scores.append(evaluation.evaluate_program(program, namespace["priority"]))
    assert scores[0] == scores[1]


def test_vrptw_template_states_the_depot_rule():
    from benchmarks.vrptw_construct.template import task_description, template_program

    assert "when current_node == depot, return a" in template_program
    assert "never empty" in template_program
    assert "already at the depot" in task_description


def test_every_backend_receives_the_full_non_thinking_profile():
    from experiments.infra.base import SAMPLING_PROFILES, build_llm_client, llm_payload

    client = build_llm_client(base_url="http://127.0.0.1:1/v1", model="m", no_proxy="127.0.0.1",
                              max_tokens=16)
    body = client._merged_extra_body(None)
    assert (client.temperature, client.top_p) == (0.7, 0.8)
    profile = SAMPLING_PROFILES[False]
    assert {k: body[k] for k in profile if k not in ("temperature", "top_p")} == {
        k: v for k, v in profile.items() if k not in ("temperature", "top_p")}
    assert body["presence_penalty"] == 1.5 and body["top_k"] == 20
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    # A method may fix its own temperature; the rest of the profile still applies.
    client = build_llm_client(base_url="http://127.0.0.1:1/v1", model="m", no_proxy="127.0.0.1",
                              max_tokens=16, temperature=1.0)
    assert client.temperature == 1.0 and client.top_p == 0.8
    record = llm_payload(base_url="http://127.0.0.1:1/v1", model="m", no_proxy="x", max_tokens=16)
    assert record["temperature"] == 0.7 and record["sampling"] == profile
    thinking = llm_payload(base_url="http://127.0.0.1:1/v1", model="m", no_proxy="x", max_tokens=16,
                           enable_thinking=True)
    assert thinking["sampling"]["temperature"] == 1.0 and thinking["sampling"]["presence_penalty"] == 0.0
