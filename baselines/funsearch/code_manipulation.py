# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Token-level call renaming from google-deepmind/funsearch ``code_manipulation``."""

from __future__ import annotations

import io
import tokenize
from collections.abc import Iterator, MutableSet, Sequence


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    return tokenize.tokenize(io.BytesIO(code.encode()).readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    code = tokenize.untokenize(tokens)  # bytes, since the token stream starts with ENCODING
    return code.decode() if isinstance(code, bytes) else code


def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    """Yields each token with a bool indicating whether it is a function call."""
    prev_token = None
    is_attribute_access = False
    for token in _tokenize(code):
        if (prev_token and prev_token.type == tokenize.NAME
                and token.type == tokenize.OP and token.string == '('):
            yield prev_token, not is_attribute_access
            is_attribute_access = False
        else:
            if prev_token:
                is_attribute_access = prev_token.type == tokenize.OP and prev_token.string == '.'
                yield prev_token, False
        prev_token = token
    if prev_token:
        yield prev_token, False


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """Renames function calls from `source_name` to `target_name`."""
    if source_name not in code:
        return code
    tokens = []
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            token = tokenize.TokenInfo(type=token.type, string=target_name, start=token.start,
                                       end=token.end, line=token.line)
        tokens.append(token)
    return _untokenize(tokens)


def get_functions_called(code: str) -> MutableSet[str]:
    """Returns the set of all functions called in `code`."""
    return {token.string for token, is_call in _yield_token_and_is_call(code) if is_call}
