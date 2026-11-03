"""A single-request OpenAI-compatible client with a total wall-clock deadline."""

import json
from typing import Protocol
import urllib.error
import urllib.request

from .models import ModelReply
from .runtime import ProcessFailure, ProcessTimeout, isolated_call
from .schemas import response_format as role_response_format


CANDIDATE_RESPONSE_FORMAT = role_response_format("generator")


class Model(Protocol):
    def complete(self, prompt: str, *, temperature: float, max_tokens: int,
                 timeout: float, response_format: dict | None = None) -> str | ModelReply: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        return None


def _request(endpoint, api_key, payload, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(endpoint, json.dumps(payload).encode(), headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(8192).decode(errors="replace")
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        raise RuntimeError(f"Model endpoint returned HTTP {exc.code}: {detail}") from None
    choice = data["choices"][0]
    content = choice["message"].get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise TypeError("The model endpoint must return textual message content")
    return {"content": content, "model": data.get("model"),
            "finish_reason": choice.get("finish_reason"), "usage": data.get("usage") or {}}


class OpenAIClient:
    def __init__(self, base_url: str, model: str, *, api_key: str = "", request_options=None):
        if not base_url.startswith(("http://", "https://")) or not model:
            raise ValueError("Provide an HTTP(S) base_url and a model name")
        self.endpoint = base_url.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        self.model = model
        self._api_key = api_key
        self.request_options = {"response_format": CANDIDATE_RESPONSE_FORMAT,
                                **dict(request_options or {})}

    def complete(self, prompt: str, *, temperature: float, max_tokens: int,
                 timeout: float, response_format: dict | None = None) -> ModelReply:
        payload = {
            **self.request_options,
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            value, _ = isolated_call(_request, (self.endpoint, self._api_key, payload, timeout), timeout)
        except ProcessTimeout as exc:
            raise TimeoutError(f"Model request exceeded the total time limit of {timeout:g} seconds") from exc
        except ProcessFailure as exc:
            raise RuntimeError(str(exc)) from exc
        return ModelReply(**value)

    def identity(self):
        names = {"top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                 "repetition_penalty", "repeat_penalty", "chat_template_kwargs",
                 "enable_thinking", "thinking"}
        return {"model": self.model, "endpoint": self.endpoint,
                "sampling": {k: v for k, v in self.request_options.items() if k in names}}

