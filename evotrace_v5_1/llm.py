"""OpenAI-compatible chat completions client using only the standard library."""

import json
import time
import urllib.error
import urllib.request

from .models import CANDIDATE_RESPONSE_FORMAT, ModelReply

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
MAX_RETRIES = 3


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the API key to another address."""

    def redirect_request(self, *args, **kwargs):
        return None


def _post(endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    body = json.dumps(payload).encode()
    opener = urllib.request.build_opener(_NoRedirect)
    deadline = time.monotonic() + timeout
    for attempt in range(MAX_RETRIES + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Model endpoint timed out after {timeout:g}s")
        request = urllib.request.Request(endpoint, body, headers, method="POST")
        try:
            with opener.open(request, timeout=remaining) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(8192).decode(errors="replace")
            if api_key:
                detail = detail.replace(api_key, "[REDACTED]")
            if exc.code in RETRYABLE_HTTP and attempt < MAX_RETRIES:
                delay = min(float(2**attempt), max(0.0, deadline - time.monotonic() - 1.0))
                if delay > 0:
                    time.sleep(delay)
                    continue
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError:
            if attempt < MAX_RETRIES:
                delay = min(float(2**attempt), max(0.0, deadline - time.monotonic() - 1.0))
                if delay > 0:
                    time.sleep(delay)
                    continue
            raise
    raise RuntimeError("Model endpoint request failed")


class OpenAIClient:
    def __init__(self, base_url: str, model: str, *, api_key: str = "", request_options=None):
        self.endpoint = base_url.rstrip("/").removesuffix("/chat/completions") + "/chat/completions"
        self.model = model
        self._api_key = api_key
        self.request_options = {
            "response_format": CANDIDATE_RESPONSE_FORMAT,
            **dict(request_options or {}),
        }

    def complete(
        self, prompt: str, *, temperature: float, max_tokens: int, timeout: float
    ) -> ModelReply:
        data = _post(
            self.endpoint,
            self._api_key,
            {
                **self.request_options,
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout,
        )
        choice = data["choices"][0]
        return ModelReply(
            choice["message"].get("content") or "",
            data.get("model"),
            choice.get("finish_reason"),
            data.get("usage") or {},
        )
