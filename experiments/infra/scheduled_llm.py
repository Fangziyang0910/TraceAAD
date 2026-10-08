"""Route generation through a host's independent model-slot scheduler."""

import time

import openai

from core.llm import LLM, generate, ModelCallError
from core.scheduling import SchedulerError, SchedulerSession, scheduler_status
from .base import build_llm_client


class ScheduledLLM(LLM):
    def __init__(self, socket_path, *, max_tokens=8192, label='', client_factory=build_llm_client):
        super().__init__()
        self.socket_path, self.label = str(socket_path), label
        self.scheduler = scheduler_status(socket_path)
        self.endpoints = self.scheduler['endpoints']
        self.clients = []
        no_proxy = ','.join(sorted({part for e in self.endpoints for part in e.get('no_proxy', '').split(',') if part}))
        try:
            for endpoint in self.endpoints:
                self.clients.append(client_factory(base_url=endpoint['base_url'], model=endpoint['model'],
                    no_proxy=no_proxy, max_tokens=max_tokens, enable_thinking=False))
        except BaseException:
            self.close()
            raise
        self.last_dispatch = None

    def __getattr__(self, name):
        # Keep token-accounting and sampling metadata identical to the existing
        # model client. All endpoints must use the same tokenizer/template.
        clients = self.__dict__.get('clients')
        if clients:
            return getattr(clients[0], name)
        raise AttributeError(name)

    def count_tokens(self, text):
        return self._count('count_tokens', text)

    def count_prompt_tokens(self, prompt):
        return self._count('count_prompt_tokens', prompt)

    def _count(self, method, value):
        for i, client in enumerate(self.clients):
            try:
                return getattr(client, method)(value)
            except Exception:
                if i == len(self.clients) - 1:
                    raise

    def draw_sample_with_details(self, prompt, **kwargs):
        started = time.monotonic()
        self.last_dispatch = None
        with SchedulerSession(self.socket_path, 'gpu', self.label) as session:
            if session.status()['instance_id'] != self.scheduler['instance_id']:
                raise SchedulerError('scheduler restarted; recreate the model client before continuing')
            slot = session.acquire_gpu()
            index = int(slot.split(':')[0])
            endpoint = self.endpoints[index]
            self.last_dispatch = {'socket': self.socket_path, 'endpoint': endpoint['base_url'],
                'slot': slot, 'queue_seconds': time.monotonic() - started}
            request_started = time.monotonic()
            failed = False
            try:
                details = self.clients[index].draw_sample_with_details(prompt, **kwargs)
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                failed = (status == 429 or isinstance(status, int) and status >= 500 or
                          isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError,
                                           ConnectionError, TimeoutError, OSError)))
                raise
            finally:
                self.last_dispatch['request_seconds'] = time.monotonic() - request_started
                session.poll(0, [slot], failed=failed)
            return {**details, 'scheduler': dict(self.last_dispatch)}

    def draw_sample(self, prompt, **kwargs):
        details = generate(self, prompt, **kwargs)
        if not details['content']:
            raise ModelCallError(RuntimeError('model returned empty content'), details['calls'], False)
        return details['content']

    def close(self):
        for client in self.clients:
            client.close()
