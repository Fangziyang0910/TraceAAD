from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.llm as openai_api
from core.llm import OpenAIAPI


def make_response(
    content="OK", reasoning=None, reasoning_content=None, finish_reason="stop"
):
    message = SimpleNamespace(content=content, reasoning=reasoning)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ]
    )


def install_fake_openai(monkeypatch, response):
    clients = []

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(openai_api.openai, "OpenAI", FakeOpenAI)
    return clients


def test_completion_details_preserve_usage_and_length_even_with_empty_content(monkeypatch):
    response = make_response(content=None, finish_reason='length')
    response.usage = SimpleNamespace(model_dump=lambda: {'prompt_tokens': 12, 'completion_tokens': 7})
    response.id = 'test-response'
    response.model = 'test-model'
    clients = install_fake_openai(monkeypatch, response)
    llm = OpenAIAPI(base_url='http://localhost:8000/v1', api_key='EMPTY', model='test-model')
    details = llm.draw_sample_with_details('test', max_tokens=7)
    assert details == {'content': '', 'finish_reason': 'length', 'usage': {'prompt_tokens': 12, 'completion_tokens': 7},
                       'response_id': 'test-response', 'model': 'test-model'}
    assert clients[0].chat.completions.calls[0]['max_tokens'] == 7


def test_openai_api_disables_thinking_by_default(monkeypatch):
    clients = install_fake_openai(monkeypatch, make_response())

    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
        temperature=0,
        top_p=0.9,
    )
    assert llm.draw_sample(" Say OK ") == "OK"

    client = clients[0]
    assert client.init_kwargs["base_url"] == "http://127.0.0.1:8001/v1"
    assert client.init_kwargs["api_key"] == "EMPTY"
    assert client.init_kwargs["timeout"] == 60

    call = client.chat.completions.calls[0]
    assert call["model"] == "Qwen3.6-27B"
    assert call["messages"] == [{"role": "user", "content": "Say OK"}]
    assert call["max_tokens"] == 16384
    assert call["temperature"] == 0
    assert call["top_p"] == 0.9
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_openai_api_allows_request_overrides(monkeypatch):
    clients = install_fake_openai(monkeypatch, make_response("custom"))
    llm = OpenAIAPI(
        base_url="http://222.201.145.8:8080/v1",
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
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "guided_choice": ["custom"],
        },
    )

    assert output == "custom"
    call = clients[0].chat.completions.calls[0]
    assert call["max_tokens"] == 7
    assert call["temperature"] == 0.2
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["extra_body"]["chat_template_kwargs"] == {
        "custom_flag": True,
        "enable_thinking": True,
    }
    assert call["extra_body"]["guided_choice"] == ["custom"]


def test_openai_api_reports_empty_content(monkeypatch):
    install_fake_openai(
        monkeypatch,
        make_response(content=None, reasoning="thinking only", finish_reason="length"),
    )
    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
        max_tokens=16,
    )

    with pytest.raises(
        RuntimeError, match="empty message.content.*enable_thinking.*max_tokens"
    ):
        llm.draw_sample("Say OK only.")


def test_openai_api_reports_reasoning_content_only_response(monkeypatch):
    install_fake_openai(
        monkeypatch,
        make_response(
            content="", reasoning_content="thinking only", finish_reason="length"
        ),
    )
    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
        max_tokens=16,
    )

    with pytest.raises(
        RuntimeError, match="empty message.content.*enable_thinking.*max_tokens"
    ):
        llm.draw_sample("Say OK only.")


def test_openai_api_close_delegates_to_client(monkeypatch):
    clients = install_fake_openai(monkeypatch, make_response())
    llm = OpenAIAPI(
        base_url="http://222.201.145.8:8080/v1",
        api_key="EMPTY",
        model="qwen3.6-27b-awq",
    )

    llm.close()

    assert clients[0].closed is True


def test_openai_api_counts_tokens_with_the_model_server_tokenizer(monkeypatch):
    install_fake_openai(monkeypatch, make_response())
    calls = []

    class FakeTokenResponse:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"tokens": [10, 20, 30]}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeTokenResponse()

    monkeypatch.setattr(
        openai_api, "requests", SimpleNamespace(post=fake_post), raising=False
    )
    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
    )

    assert llm.count_tokens("hello world") == 3
    assert calls[0][0] == "http://127.0.0.1:8001/tokenize"
    assert calls[0][1]["json"] == {
        "model": "Qwen3.6-27B",
        "prompt": "hello world",
    }
    assert llm.token_count_mode == "llm_count_tokens"


def test_openai_api_counts_complete_chat_prompt(monkeypatch):
    install_fake_openai(monkeypatch, make_response())
    calls = []

    class FakeTokenResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"count": 12}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeTokenResponse()

    monkeypatch.setattr(
        openai_api, "requests", SimpleNamespace(post=fake_post), raising=False
    )
    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
        enable_thinking=False,
    )

    assert llm.count_prompt_tokens("hello world") == 12
    assert calls[0][1]["json"] == {
        "model": "Qwen3.6-27B",
        "messages": [{"role": "user", "content": "hello world"}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert llm.prompt_token_count_mode == "llm_chat_template_tokens"


def test_openai_api_raises_after_tokenizer_retry_limit(monkeypatch):
    install_fake_openai(monkeypatch, make_response())
    calls = []

    def unavailable_post(url, **kwargs):
        calls.append((url, kwargs))
        raise RuntimeError("404")

    monkeypatch.setattr(
        openai_api,
        "requests",
        SimpleNamespace(post=unavailable_post),
        raising=False,
    )
    llm = OpenAIAPI(
        base_url="https://example.test/v1",
        api_key="secret",
        model="example-model",
    )

    with pytest.raises(openai_api.TokenizationError, match="after 3 attempts"):
        llm.count_tokens("你好")
    assert len(calls) == 3


def test_openai_api_counts_tokens_via_llamacpp_content_field(monkeypatch):
    install_fake_openai(monkeypatch, make_response())
    calls = []

    class EmptyThenContent:
        @staticmethod
        def raise_for_status():
            return None

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        body = kwargs["json"]
        if body.get("content") == "hello world":
            return EmptyThenContent({"tokens": [1, 2, 3, 4]})
        return EmptyThenContent({"tokens": []})

    monkeypatch.setattr(
        openai_api, "requests", SimpleNamespace(post=fake_post), raising=False
    )
    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.8-27B",
    )

    assert llm.count_tokens("hello world") == 4
    assert calls[0] == (
        "http://127.0.0.1:8001/tokenize",
        {"model": "Qwen3.8-27B", "prompt": "hello world"},
    )
    assert calls[1] == (
        "http://127.0.0.1:8001/tokenize",
        {"content": "hello world"},
    )


def test_openai_api_counts_chat_prompt_via_llamacpp_apply_template(monkeypatch):
    install_fake_openai(monkeypatch, make_response())
    calls = []

    class FakeResponse:
        @staticmethod
        def raise_for_status():
            return None

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        body = kwargs["json"]
        calls.append((url, body))
        if url.endswith("/apply-template"):
            return FakeResponse({"prompt": "<|im_start|>user\nhello<|im_end|>\n"})
        if body.get("content"):
            return FakeResponse({"tokens": list(range(9))})
        return FakeResponse({"tokens": []})

    monkeypatch.setattr(
        openai_api, "requests", SimpleNamespace(post=fake_post), raising=False
    )
    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.8-27B",
        enable_thinking=False,
    )

    assert llm.count_prompt_tokens("hello") == 9
    assert calls[1][0] == "http://127.0.0.1:8001/apply-template"
    assert calls[2] == (
        "http://127.0.0.1:8001/tokenize",
        {"content": "<|im_start|>user\nhello<|im_end|>\n"},
    )
