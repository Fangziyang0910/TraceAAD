"""A model call survives a server restart and stops on real errors."""

import openai
import pytest

from core import llm as core_llm


class Flaky:
    model = "test"

    def __init__(self, failures, error):
        self.failures, self.error = failures, error

    def draw_sample_with_details(self, prompt, **kwargs):
        if self.failures:
            self.failures -= 1
            raise self.error
        return {"content": "ok", "finish_reason": "stop", "usage": {"completion_tokens": 1}}


def test_connection_errors_are_retried_until_the_server_returns(monkeypatch):
    slept = []
    monkeypatch.setattr(core_llm.time, "sleep", slept.append)
    result = core_llm.generate(Flaky(6, ConnectionError("down")), "p")
    assert result["content"] == "ok" and len(result["calls"]) == 7
    assert slept == [2, 4, 8, 16, 32, 60]


def test_waiting_ends_after_the_service_limit(monkeypatch):
    monkeypatch.setattr(core_llm.time, "sleep", lambda s: None)
    with pytest.raises(core_llm.ModelCallError) as caught:
        core_llm.generate(Flaky(10**6, ConnectionError("down")), "p")
    assert caught.value.transient
    waits = [min(60, 2 ** (i + 1)) for i in range(len(caught.value.calls) - 1)]
    assert sum(waits) >= core_llm.SERVICE_WAIT_SECONDS > sum(waits[:-1])


def test_a_client_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(core_llm.time, "sleep", lambda s: pytest.fail("no retry"))
    with pytest.raises(core_llm.ModelCallError) as caught:
        core_llm.generate(Flaky(1, ValueError("bad request")), "p")
    assert not caught.value.transient and len(caught.value.calls) == 1
