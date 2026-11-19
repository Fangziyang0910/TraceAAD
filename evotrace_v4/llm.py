"""OpenAI-compatible chat completions client using only the standard library."""

import json
import urllib.error
import urllib.request

from .models import CANDIDATE_RESPONSE_FORMAT, ModelReply


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the API key to another address."""

    def redirect_request(self, *args, **kwargs):
        return None


def _post(endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(endpoint, json.dumps(payload).encode(), headers, method="POST")
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(8192).decode(errors="replace")
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        raise RuntimeError(f"Model endpoint returned HTTP {exc.code}: {detail}") from None


class OpenAIClient:
    def __init__(self, base_url: str, model: str, *, api_key: str = "", request_options=None):
        self.endpoint = base_url.rstrip("/").removesuffix("/chat/completions") + "/chat/completions"
        self.model = model
        self._api_key = api_key
        self.request_options = {"response_format": CANDIDATE_RESPONSE_FORMAT, **dict(request_options or {})}

    def complete(self, prompt: str, *, temperature: float, max_tokens: int, timeout: float) -> ModelReply:
        data = _post(self.endpoint, self._api_key, {
            **self.request_options,
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }, timeout)
        choice = data["choices"][0]
        return ModelReply(choice["message"].get("content") or "", data.get("model"),
                          choice.get("finish_reason"), data.get("usage") or {})
