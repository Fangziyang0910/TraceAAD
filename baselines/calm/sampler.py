"""LLM sampling for CALM via llm4ad LLM abstraction."""

from __future__ import annotations

from typing import List, Optional, Sequence

from core import LLM


class CALMSampler:
    def __init__(self, llm: LLM, profiler=None):
        self.llm = llm
        self._profiler = profiler

    @staticmethod
    def messages_to_prompt(messages: Sequence[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            parts.append(f'[{role}]\n{content}')
        return '\n\n'.join(parts)

    def sample(self, messages: Sequence[dict], *, n: int = 1, operator: str | None = None,
               sample_order: int | None = None) -> List[str]:
        prompt = self.messages_to_prompt(messages)
        responses = []
        for i in range(n):
            response = self.llm.draw_sample(prompt)
            responses.append(response)
            self._log_llm_call(
                stage='generate',
                operator=operator,
                sample_order=sample_order,
                generation_index=i,
                prompt=prompt,
                response=response,
            )
        return responses

    def _log_llm_call(self, **payload):
        logger = getattr(self._profiler, 'log_llm_call', None)
        if callable(logger):
            try:
                logger(**payload)
            except Exception:
                pass
