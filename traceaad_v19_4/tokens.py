"""Token counts from the selected generator's locally available tokenizer."""

import hashlib
from pathlib import Path


class TokenCounter:
    def __init__(self, name):
        from tokenizers import Tokenizer
        path = Path(name).expanduser()
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.is_file():
            from huggingface_hub import hf_hub_download
            try:
                path = Path(hf_hub_download(name, "tokenizer.json", local_files_only=True))
            except Exception as exc:
                raise ValueError("Provide a local tokenizer.json or an already cached tokenizer repository") from exc
        self.tokenizer = Tokenizer.from_file(str(path))
        self.identity = {"tokenizer": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def __call__(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
