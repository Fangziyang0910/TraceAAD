"""Embed the full program without exceeding the service's per-input limit."""

import numpy as np

from experiments.shinka_evo.embedding import CodeEmbedding as ServiceEmbedding


CHUNK_BYTES = 6000


class CodeEmbedding(ServiceEmbedding):
    def __call__(self, code):
        data = code.encode('utf-8')
        if not data:
            raise ValueError('Cannot embed an empty program')
        vectors, weights, usage = [], [], {}
        start = 0
        try:
            while start < len(data):
                end = min(start + CHUNK_BYTES, len(data))
                while end < len(data) and data[end] & 0xC0 == 0x80:
                    end -= 1
                chunk = data[start:end].decode('utf-8')
                vectors.append(super().__call__(chunk))
                weights.append(end - start)
                for key, value in self.last_usage.items():
                    if isinstance(value, (int, float)):
                        usage[key] = usage.get(key, 0) + value
                start = end
        finally:
            self.last_usage = usage
        return np.average(vectors, axis=0, weights=weights).tolist()
