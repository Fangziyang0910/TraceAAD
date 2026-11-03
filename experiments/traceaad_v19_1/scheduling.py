"""Client for server4's existing resource protocol; no search code is shared."""

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import time
from urllib.parse import urlsplit

from traceaad_v19_1 import Evaluator, OpenAIClient
from traceaad_v19_1.llm import CANDIDATE_RESPONSE_FORMAT


# Same non-thinking Qwen controls as the existing experiment pool; temperature
# and output length come from the independently configured Base on every call.
REQUEST_OPTIONS = {
    "top_p": .8, "top_k": 20, "min_p": 0.,
    "presence_penalty": 1.5, "frequency_penalty": 0.,
    "repetition_penalty": 1., "repeat_penalty": 1.,
    "chat_template_kwargs": {"enable_thinking": False},
    "response_format": CANDIDATE_RESPONSE_FORMAT,
}


def load_environment(path):
    if path is None:
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def api_key(url):
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    host = urlsplit(url).hostname
    if host in {"222.201.145.6", "222.201.145.8"}:
        return os.environ.get("SERVER3_API_KEY", "EMPTY")
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


class Lease:
    def __init__(self, path, kind="cpu", label=""):
        self.kind, self.label = kind, label
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(10)
        try:
            self.socket.connect(str(path))
            self.stream = self.socket.makefile("rb")
        except BaseException:
            self.socket.close()
            raise

    def request(self, message):
        self.socket.sendall((json.dumps(message) + "\n").encode())
        raw = self.stream.readline(2**20)
        if not raw:
            raise ConnectionError("Resource scheduler disconnected")
        response = json.loads(raw)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def status(self):
        return self.request({"op": "status"})

    def poll(self, target, release=(), *, failed=False):
        return self.request({"op": "poll", "kind": self.kind, "label": self.label,
                             "target": target, "release": list(release),
                             "failed": failed, "endpoint": None})["resources"]

    def acquire(self):
        while True:
            if resources := self.poll(1):
                return resources[0]
            time.sleep(.1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()
        self.socket.close()


def scheduler_status(path):
    with Lease(path) as lease:
        return lease.status()


class ScheduledModel:
    model = "Qwen3.8-27B"

    def __init__(self, path, label, *, client_factory=OpenAIClient):
        self.path, self.label = str(path), label
        self.scheduler = scheduler_status(path)
        self.factory = client_factory
        self.client = None
        self.last_metadata = {}
        hosts = {urlsplit(e["base_url"]).hostname for e in self.scheduler["endpoints"]}
        hosts.update(os.environ.get("NO_PROXY", "").split(","))
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(sorted(h for h in hosts if h))

    @contextmanager
    def request_context(self):
        self.last_metadata = {}
        started = time.monotonic()
        with Lease(self.path, "gpu", self.label) as lease:
            state = lease.status()
            if state["instance_id"] != self.scheduler["instance_id"]:
                raise RuntimeError("Resource scheduler restarted; restart this experiment with --resume")
            slot = lease.acquire()
            endpoint = self.scheduler["endpoints"][int(slot.split(":")[0])]
            self.last_metadata = {"scheduler": {
                "socket": self.path, "slot": slot, "endpoint": endpoint["base_url"],
                "model": endpoint["model"], "routing": "shared",
                "queue_seconds": time.monotonic() - started,
            }, "request_options": REQUEST_OPTIONS}
            self.client = self.factory(endpoint["base_url"], endpoint["model"],
                                       api_key=api_key(endpoint["base_url"]),
                                       request_options=REQUEST_OPTIONS)
            self.failed = False
            try:
                yield
            finally:
                self.client = None
                lease.poll(0, [slot], failed=self.failed)

    def complete(self, prompt, **settings):
        if self.client is None:
            raise RuntimeError("Reserve a model slot before issuing a request")
        started = time.monotonic()
        try:
            reply = self.client.complete(prompt, **settings)
        except Exception:
            self.failed = True
            raise
        finally:
            self.last_metadata["scheduler"]["request_seconds"] = time.monotonic() - started
        return replace(reply, metadata=dict(self.last_metadata))


class ScheduledEvaluator(Evaluator):
    def __init__(self, task, path, label):
        super().__init__(task)
        self.path, self.label = str(path), label
        self.last_dispatch = {}

    @contextmanager
    def evaluation_context(self):
        started = time.monotonic()
        with Lease(self.path, "cpu", self.label) as lease:
            cpu = lease.acquire()
            self.last_dispatch = {"cpu": cpu, "queue_seconds": time.monotonic() - started}
            original = os.sched_getaffinity(0)
            try:
                os.sched_setaffinity(0, {int(cpu)})
                yield
            finally:
                os.sched_setaffinity(0, original)
                lease.poll(0, [cpu])
