"""OpenAI-compatible embedding service used by Shinka's official novelty flow."""

import math
import os

from openai import OpenAI

from experiments.infra.env import load_repo_dotenv


def embedding_settings():
    load_repo_dotenv()
    return {
        'base_url': os.environ.get('SHINKA_EMBEDDING_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        'model': os.environ.get('SHINKA_EMBEDDING_MODEL', 'text-embedding-v4'),
        'api_key_env': os.environ.get('SHINKA_EMBEDDING_API_KEY_ENV', 'DASHSCOPE_API_KEY'),
    }


class CodeEmbedding:
    def __init__(self):
        self.settings = embedding_settings()
        self._client = None
        self.last_usage = {}

    def __call__(self, code):
        if self._client is None:
            key = os.environ.get(self.settings['api_key_env'])
            if not key:
                raise RuntimeError(f"Code embeddings require {self.settings['api_key_env']} in the environment or .env.")
            self._client = OpenAI(base_url=self.settings['base_url'], api_key=key,
                                  timeout=60, max_retries=2)
        result = self._client.embeddings.create(model=self.settings['model'], input=[code], encoding_format='float')
        self.last_usage = result.usage.model_dump() if result.usage is not None else {}
        vector = result.data[0].embedding
        if not vector or not all(math.isfinite(v) for v in vector) or not any(vector):
            raise ValueError('Embedding service returned an invalid code vector.')
        return vector

    def close(self):
        if self._client is not None:
            self._client.close()
