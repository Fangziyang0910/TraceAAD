from __future__ import annotations

from types import SimpleNamespace

import core.llm as openai_api
from core.llm import OpenAIAPI


def make_response(content="OK", reasoning=None, reasoning_content=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, reasoning=reasoning)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def install_fake_openai(monkeypatch, response):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai_api.openai, "OpenAI", lambda **kwargs: client)
    return calls


def test_completion_details_preserve_usage_and_length_even_with_empty_content(monkeypatch):
    response = make_response(content=None, finish_reason="length")
    response.usage = SimpleNamespace(
        model_dump=lambda: {"prompt_tokens": 12, "completion_tokens": 7}
    )
    response.id = "test-response"
    response.model = "test-model"
    clients = install_fake_openai(monkeypatch, response)
    llm = OpenAIAPI(base_url="http://localhost:8000/v1", api_key="EMPTY", model="test-model")
    details = llm.draw_sample_with_details("test", max_tokens=7)
    assert details == {
        "content": "",
        "finish_reason": "length",
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        "response_id": "test-response",
        "model": "test-model",
    }
    assert clients[0]["max_tokens"] == 7


def test_openai_api_allows_request_overrides(monkeypatch):
    clients = install_fake_openai(monkeypatch, make_response("custom"))
    llm = OpenAIAPI(
        base_url="http://localhost/v1",
        api_key="EMPTY",
        model="qwen3.6-27b-awq",
        max_tokens=1024,
        temperature=1.0,
        extra_body={"chat_template_kwargs": {"custom_flag": True}},
    )

    output = llm.draw_sample(
        [{"role": "user", "content": "hello"}],
        max_tokens=7,
        temperature=0.2,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "guided_choice": ["custom"]},
    )

    assert output == "custom"
    call = clients[0]
    assert call["max_tokens"] == 7
    assert call["temperature"] == 0.2
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["extra_body"]["chat_template_kwargs"] == {
        "custom_flag": True,
        "enable_thinking": True,
    }
    assert call["extra_body"]["guided_choice"] == ["custom"]


def test_high_thinking_is_sent_and_reasoning_is_kept_in_transport_records(monkeypatch):
    response = make_response(content="code", reasoning_content="reasoned about the algorithm")
    response.usage = SimpleNamespace(
        model_dump=lambda: {
            "prompt_tokens": 3,
            "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
    )
    clients = install_fake_openai(monkeypatch, response)
    llm = OpenAIAPI(
        base_url="http://localhost/v1",
        api_key="EMPTY",
        model="deepseek-v4.1-flash",
        enable_thinking=True,
        reasoning_effort="high",
    )
    assert llm.draw_sample("task") == "code"
    body = clients[0]["extra_body"]
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    assert llm.last_calls[0]["reasoning"] == "reasoned about the algorithm"
    assert llm.last_calls[0]["sampling"]["reasoning_effort"] == "high"
    assert llm.last_calls[0]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 5
