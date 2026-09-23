import json
import tempfile
import threading
import unittest
from pathlib import Path

from core import Function
from core.evaluate import Evaluation
from baselines.eoh.observability import init_observability, record_sample_failure
from baselines.eoh.sampling import sample_thought_and_function, trim_braced_thought
from baselines.profiler import ProfilerBase


def make_function(label: int, score: float) -> Function:
    func = Function(name="heuristic", args="x", body=f"    return {label}")
    func.algorithm = f"algorithm-{label}"
    func.operator = "test"
    func.score = score
    func.sample_time = 0.01
    func.evaluate_time = 0.02
    return func


class DummyMethod:
    def __init__(self, profiler):
        self._profiler = profiler
        self._tot_sample_nums = 0
        self._debug_mode = False
        init_observability(self, max_consecutive_sample_failures=2)


class FakeEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program="def heuristic(x):\n    return x\n",
            task_description="Design a heuristic.",
            safe_evaluate=False,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(3)

    def evaluate(self, callable_func):
        return callable_func(3)


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def draw_sample(self, prompt):
        self.prompts.append(prompt)
        return self.response


class ProfilerObservabilityTest(unittest.TestCase):
    def test_common_jsonl_summary_and_sample_history_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = ProfilerBase(
                log_dir=tmpdir, create_random_path=False, log_style="simple"
            )
            profiler.register_function(make_function(1, 1.0), program="program-1")
            profiler.register_function(make_function(2, 2.0), program="program-2")
            profiler.log_llm_call(stage="generate", prompt="p", response="r")
            profiler.log_method_event(event="operator_start", operator="m1")
            profiler.log_method_state(phase="population", population_size=2)
            profiler.log_error("sample", TimeoutError("x" * 1200))
            profiler.write_run_summary(status="finished", method_sample_count=2)

            log_dir = Path(tmpdir)
            self.assertEqual(
                len((log_dir / "llm_calls.jsonl").read_text().splitlines()), 1
            )
            self.assertEqual(
                len((log_dir / "method_events.jsonl").read_text().splitlines()), 1
            )
            self.assertEqual(
                len((log_dir / "method_state.jsonl").read_text().splitlines()), 1
            )

            errors = [
                json.loads(line)
                for line in (log_dir / "errors.jsonl").read_text().splitlines()
            ]
            self.assertEqual(errors[0]["error_type"], "TimeoutError")
            self.assertLessEqual(len(errors[0]["error"]), 1000)

            samples = json.loads(
                (log_dir / "samples" / "samples_1~200.json").read_text()
            )
            self.assertEqual([entry["score"] for entry in samples], [1.0, 2.0])
            self.assertEqual(
                set(samples[0]),
                {"sample_order", "score", "operator", "program"},
            )
            self.assertFalse((log_dir / "samples" / "samples_best.json").exists())

            summary = json.loads((log_dir / "run_summary.json").read_text())
            self.assertEqual(summary["status"], "finished")
            self.assertEqual(summary["num_samples"], 2)
            self.assertEqual(summary["best_score"], 2.0)
            self.assertEqual(summary["llm_call_count"], 1)
            self.assertEqual(summary["error_count"], 1)

    def test_jsonl_append_is_thread_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = ProfilerBase(
                log_dir=tmpdir, create_random_path=False, log_style="simple"
            )

            def worker(offset):
                for idx in range(25):
                    profiler.log_method_event(event="thread_event", idx=offset + idx)

            threads = [
                threading.Thread(target=worker, args=(i * 25,)) for i in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            events = (Path(tmpdir) / "method_events.jsonl").read_text().splitlines()
            self.assertEqual(len(events), 100)
            self.assertTrue(
                all(json.loads(line)["event"] == "thread_event" for line in events)
            )

    def test_parameter_log_does_not_include_credentials(self):
        class LLMWithSecret:
            model = "test-model"
            api_key = "secret-that-must-not-be-logged"

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = ProfilerBase(
                log_dir=tmpdir, create_random_path=False, log_style="simple"
            )
            profiler.record_parameters(
                LLMWithSecret(), FakeEvaluation(), DummyMethod(None)
            )

            text = (Path(tmpdir) / "run_log.txt").read_text()
            self.assertIn("test-model", text)
            self.assertNotIn("secret-that-must-not-be-logged", text)

    def test_record_sample_failure_aborts_at_threshold_without_budget_increment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = ProfilerBase(
                log_dir=tmpdir, create_random_path=False, log_style="simple"
            )
            method = DummyMethod(profiler)

            aborted = record_sample_failure(
                method, TimeoutError("first"), stage="sample", operator="m1"
            )
            self.assertFalse(aborted)
            self.assertEqual(method._tot_sample_nums, 0)

            aborted = record_sample_failure(
                method, TimeoutError("second"), stage="sample", operator="m1"
            )
            self.assertTrue(aborted)
            self.assertTrue(method._search_aborted)
            self.assertEqual(method._tot_sample_nums, 0)

            errors = (Path(tmpdir) / "errors.jsonl").read_text().splitlines()
            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "method_events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(errors), 2)
            self.assertEqual(events[-1]["event"], "search_aborted")

    def test_simple_sampler_helper_parses_and_logs_llm_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = ProfilerBase(
                log_dir=tmpdir, create_random_path=False, log_style="simple"
            )
            llm = FakeLLM("{use x plus one}\ndef heuristic(x):\n    return x + 1\n")

            thought, function = sample_thought_and_function(
                llm,
                "prompt text",
                "def heuristic(x):\n    return x\n",
                profiler=profiler,
                operator="m1",
                sample_order=3,
            )

            self.assertEqual(thought, "{use x plus one}")
            self.assertEqual(trim_braced_thought("no thought"), None)
            self.assertIsNotNone(function)
            self.assertEqual(function.name, "heuristic")
            self.assertIn("return x + 1", function.body)

            calls = [
                json.loads(line)
                for line in (Path(tmpdir) / "llm_calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(calls[0]["operator"], "m1")
            self.assertEqual(calls[0]["sample_order"], 3)
            self.assertTrue(calls[0]["thought_parse_success"])
            self.assertTrue(calls[0]["function_parse_success"])


if __name__ == "__main__":
    unittest.main()
