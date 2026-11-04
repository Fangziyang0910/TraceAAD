"""Qwen embedding endpoint; vectors come back L2-normalized."""

import json
import math
import os
import urllib.request

INSTRUCT = {
    'State': "Retrieve experience notes whose stated conditions or source algorithms match the current algorithm's representation and decision rules.",
    'Need': "Retrieve design lessons relevant to the requested modification and recent outcomes, including ways to revise an unsuccessful implementation or combine computations.",
    'Implementation': "Retrieve source computations relevant to the current function's implementation, even when their descriptions use different terminology.",
}


class EmbeddingClient:
    def __init__(self):
        self.model = os.getenv('TRACEAAD_EMBEDDING_MODEL', 'qwen3-embedding-4b-fp8')
        self.base = os.getenv('TRACEAAD_EMBEDDING_BASE_URL', 'http://127.0.0.1:18002/v1').rstrip('/')
        self.key = os.getenv(os.getenv('TRACEAAD_EMBEDDING_API_KEY_ENV', 'SERVER3_API_KEY'), '')
        self.last_usage = {}

    def encode(self, text):
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        body = json.dumps({'model': self.model, 'input': [text], 'encoding_format': 'float'}).encode()
        request = urllib.request.Request(self.base + '/embeddings', body, headers=headers)
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=60) as response:
            result = json.load(response)
        self.last_usage = result.get('usage') or {}
        vector = result['data'][0]['embedding']
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]
