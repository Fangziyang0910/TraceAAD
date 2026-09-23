"""Parsing helpers ported from reference_code/CALM/utils.py."""

from __future__ import annotations

import re


def dedent(s: str) -> str:
    lines = s.split('\n')
    first_indent = None
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            first_indent = len(line) - len(stripped)
            break
    if first_indent is None:
        return s
    new_lines = []
    for line in lines:
        if line.startswith(' ' * first_indent):
            new_lines.append(line[first_indent:])
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


def get_code(response: str):
    code_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    matches = code_pattern.findall(response)
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else matches


def extract_function_from_string(file_content: str):
    namespace: dict = {}
    try:
        exec(file_content, namespace)
    except Exception:
        return None
    functions = [
        value for value in namespace.values()
        if callable(value) and hasattr(value, '__code__')
    ]
    if len(functions) != 1:
        return None
    return functions[0]


def extract_first_double_braced(text: str):
    start = text.find('{{')
    end = text.find('}}', start + 1)
    if start != -1 and end != -1:
        return text[start + 2:end]
    return None


def extract_idea_description(text: str) -> str:
    pattern = r"The idea of the algorithm is to.*?(?=(\n?`{3,}[^`]*\n)|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group().strip() if match else ''


def idea_distance(base_idea: str, new_idea: str) -> float:
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))

    base_words = tokenize(base_idea)
    new_words = tokenize(new_idea)
    if not new_words:
        return 0.0
    new_introduced_words = new_words - base_words
    return len(new_introduced_words) / len(new_words)
