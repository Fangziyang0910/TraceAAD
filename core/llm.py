"""LLM interface: the abstract client contract and the OpenAI-compatible one."""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Sequence
from threading import local
from typing import Any
from uuid import uuid4

import openai
import requests


class LLM:
    def draw_sample(self, prompt, **kwargs):
        raise NotImplementedError

    def close(self):
        pass


TOKENIZE_RETRY_LIMIT = 3


class TokenizationError(RuntimeError):
    """The serving model could not count tokens for the exact request."""


class ModelCallError(RuntimeError):
    def __init__(self, error, calls, transient):
        super().__init__(str(error))
        self.calls, self.transient = calls, transient


# A model server that restarts (a crash, a reload) is back within minutes;
# connection and server errors are retried with capped backoff for this long.
SERVICE_WAIT_SECONDS = 900


def generate(llm, prompt, *, retry_limit=None, **kwargs):
    """One generation request, including every physical request and retry."""
    calls, waited = [], 0.0
    for attempt in itertools.count():
        started = time.monotonic()
        try:
            details = llm.draw_sample_with_details(prompt, **kwargs)
            error = None
        except Exception as exc:
            details, error = {}, exc
        call = {
            "transport_id": uuid4().hex,
            "prompt": prompt,
            "response": details.get("content", ""),
            "usage": details.get("usage") or {},
            "finish_reason": details.get("finish_reason"),
            "model": details.get("model", getattr(llm, "model", None)),
            "seconds": time.monotonic() - started,
            "error": f"{type(error).__name__}: {error}" if error else None,
            "sampling": {
                name: kwargs.get(name, getattr(llm, name, None))
                for name in (
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "enable_thinking",
                    "reasoning_effort",
                )
            },
        }
        if details.get("reasoning"):
            call["reasoning"] = details["reasoning"]
        if "messages" in kwargs:
            call["messages"] = kwargs["messages"]
        dispatch = details.get("scheduler") or getattr(llm, "last_dispatch", None)
        if dispatch is not None:
            call["scheduler"] = dict(dispatch)
            if not details.get('model') and dispatch.get('model'):
                call['model'] = dispatch['model']
        calls.append(call)
        if hasattr(llm, "_requests"):
            llm._requests.calls = calls
        if error is None:
            return {**details, "calls": calls}
        status = getattr(error, "status_code", None)
        # The hosted gateway reports a failure of the provider it routed to as 400; the same request succeeds again.
        upstream = status == 400 and "Upstream request failed" in str(error)
        transient = (
            status == 429
            or isinstance(status, int)
            and status >= 500
            or upstream
            or isinstance(
                error,
                (
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ),
            )
        )
        if not transient or waited >= SERVICE_WAIT_SECONDS or (retry_limit is not None and attempt >= retry_limit):
            raise ModelCallError(error, calls, transient) from error
        pause = min(60, 2 ** (attempt + 1))
        time.sleep(pause)
        waited += pause


class OpenAIAPI(LLM):
    """Chat requests and exact token counts for the configured model service."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60,
        max_tokens: int | None = 16384,
        temperature: float = 1.0,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        enable_thinking: bool | None = False,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        tokenizer: str | None = None,
        prompt_format: str = "{}",
        **client_kwargs: Any,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.prompt_format = prompt_format
        self._tokenizer = None
        if tokenizer:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(hf_hub_download(tokenizer, "tokenizer.json"))
        self.extra_body = copy.deepcopy(extra_body) if extra_body else {}
        self._requests = local()
        client_kwargs["max_retries"] = 0
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, **client_kwargs
        )

    def draw_sample(self, prompt: str | Any, *args: Any, **kwargs: Any) -> str:
        details = generate(self, prompt, **kwargs)
        if not details["content"]:
            error = RuntimeError(
                f"{type(self).__name__} received empty message.content; "
                f"finish_reason={details.get('finish_reason')!r}. "
                "Check chat_template_kwargs.enable_thinking and max_tokens."
            )
            raise ModelCallError(error, details["calls"], False)
        return details["content"]

    @property
    def last_calls(self):
        return getattr(self._requests, "calls", [])

    def draw_sample_with_details(self, prompt, **kwargs):
        """Send one request and retain content, usage and completion status."""
        messages = self._build_messages(prompt, kwargs.pop("messages", None))
        request = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "stream": False,
            "temperature": kwargs.pop("temperature", self.temperature),
        }

        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        top_p = kwargs.pop("top_p", self.top_p)
        stop = kwargs.pop("stop", self.stop)
        if top_p is not None:
            request["top_p"] = top_p
        if stop is not None:
            request["stop"] = stop

        request_extra_body = kwargs.pop("extra_body", None)
        extra_body = self._merged_extra_body(request_extra_body)
        if extra_body:
            request["extra_body"] = extra_body

        request.update(kwargs)
        response = self._client.chat.completions.create(**request)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        details = {
            "content": choice.message.content or "",
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage.model_dump() if usage else {},
            "model": getattr(response, "model", self.model),
            "response_id": getattr(response, "id", None),
        }

        reasoning = getattr(choice.message, "reasoning", None) or getattr(
            choice.message, "reasoning_content", None
        )
        if reasoning:
            details["reasoning"] = reasoning
        return details

    def count_tokens(self, text: str) -> int:
        """Count raw text tokens with the tokenizer serving this model."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
        return self._request_token_count({"model": self.model, "prompt": text})

    def count_prompt_tokens(self, prompt: str | Any) -> int:
        """Count the exact chat-templated tokens used by ``draw_sample``."""
        messages = self._build_messages(prompt, None)
        if self._tokenizer is not None:
            roles = [message.get("role") for message in messages]
            if (
                self.prompt_format == "<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜></think>"
                and roles in (["user"], ["system", "user"])
            ):
                text = "<｜begin▁of▁sentence｜>"
                if self.enable_thinking:
                    effort = getattr(self, "reasoning_effort", None) or "high"
                    budget = {"low": 50, "medium": 62, "high": 75, "max": 100}[effort]
                    text += f"<｜System｜>Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
                for message in messages:
                    role = message["role"]
                    if role == "user" or not self.enable_thinking:
                        text += "<｜System｜>" if role == "system" else "<｜User｜>"
                    text += message["content"]
                text += "<｜Assistant｜>" + ("<think>" if self.enable_thinking else "</think>")
            elif roles == ["user"]:
                text = self.prompt_format.format(messages[0]["content"])
            else:
                # shortcut: current baselines use one user or system/user; extend with the model template for other roles.
                raise ValueError("local prompt counting does not support this model/message format")
            return self.count_tokens(text)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": True,
        }
        extra_body = self._merged_extra_body(None)
        for key in ("chat_template", "chat_template_kwargs"):
            if key in extra_body:
                payload[key] = extra_body[key]
        return self._request_token_count(payload)

    def _request_token_count(self, payload: dict[str, Any]) -> int:
        base = self.base_url.rstrip("/")
        tokenize_url = (
            base[: -len("/v1")] + "/tokenize" if base.endswith("/v1") else base + "/tokenize"
        )
        headers = (
            {}
            if not self.api_key or self.api_key == "EMPTY"
            else {"Authorization": f"Bearer {self.api_key}"}
        )
        timeout = min(float(self.timeout), 30.0)
        last_error: Exception | None = None
        for _ in range(TOKENIZE_RETRY_LIMIT):
            try:
                response = requests.post(
                    tokenize_url, json=payload, headers=headers, timeout=timeout
                )
                response.raise_for_status()
                count = self._token_count_from_payload(response.json())
                if count == 0:
                    count = self._llamacpp_token_count(tokenize_url, payload, headers, timeout)
                if count == 0 and payload:
                    raise ValueError("tokenizer returned no tokens for non-empty text")
                return count
            except Exception as exc:
                last_error = exc
        raise TokenizationError(
            f"model tokenizer failed after {TOKENIZE_RETRY_LIMIT} attempts"
        ) from last_error

    def _llamacpp_token_count(
        self, tokenize_url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> int:
        """llama.cpp /tokenize expects ``content``; chat prompts need /apply-template."""
        text = payload.get("prompt")
        if not isinstance(text, str) or not text:
            messages = payload.get("messages")
            if not messages:
                return 0
            apply_url = tokenize_url.rsplit("/", 1)[0] + "/apply-template"
            template_payload = {key: value for key, value in payload.items() if key != "prompt"}
            response = requests.post(
                apply_url, json=template_payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            rendered = response.json()
            text = rendered.get("prompt") if isinstance(rendered, dict) else None
            if not isinstance(text, str) or not text:
                return 0
        response = requests.post(
            tokenize_url, json={"content": text}, headers=headers, timeout=timeout
        )
        response.raise_for_status()
        return self._token_count_from_payload(response.json())

    @property
    def token_count_mode(self) -> str:
        return "local_tokenizer" if self._tokenizer is not None else "llm_count_tokens"

    @property
    def prompt_token_count_mode(self) -> str:
        return (
            "local_chat_template_tokens"
            if self._tokenizer is not None
            else "llm_chat_template_tokens"
        )

    @staticmethod
    def _token_count_from_payload(payload: Any) -> int:
        if isinstance(payload, dict):
            if isinstance(payload.get("count"), int):
                return int(payload["count"])
            for key in ("tokens", "token_ids"):
                if isinstance(payload.get(key), list):
                    return len(payload[key])
        raise ValueError("tokenizer response does not contain a token count")

    @staticmethod
    def _build_messages(prompt: str | Any, messages: Any | None) -> list[dict[str, Any]]:
        if messages is not None:
            if isinstance(messages, dict):
                return [messages]
            return list(messages)
        if not isinstance(prompt, str):
            if isinstance(prompt, dict):
                return [prompt]
            return list(prompt)
        return [{"role": "user", "content": prompt.strip()}]

    def _merged_extra_body(self, request_extra_body: dict[str, Any] | None) -> dict[str, Any]:
        extra_body = copy.deepcopy(self.extra_body)
        if self.enable_thinking is not None:
            # Both top-level (DeepSeek official gateway) and chat_template_kwargs
            # (vLLM) are set so either server recognizes the thinking flag.
            extra_body.setdefault("enable_thinking", self.enable_thinking)
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
            chat_template_kwargs.setdefault("enable_thinking", self.enable_thinking)
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            # DeepSeek official request format, honored by opencode.ai zen/go.
            extra_body.setdefault(
                "thinking", {"type": "enabled" if self.enable_thinking else "disabled"}
            )
        if self.reasoning_effort is not None and self.enable_thinking:
            extra_body["reasoning_effort"] = self.reasoning_effort
            extra_body["chat_template_kwargs"]["reasoning_effort"] = self.reasoning_effort
        if request_extra_body:
            self._deep_update(extra_body, request_extra_body)
        return extra_body

    @classmethod
    def _deep_update(cls, target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                cls._deep_update(target[key], value)
            else:
                target[key] = value

    def close(self):
        self._client.close()
