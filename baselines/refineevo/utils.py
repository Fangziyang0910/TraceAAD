"""RefineEvo 7de015d5 helpers: response extraction from evo_interface.py and utils/utils.py."""
import re
from pathlib import Path

PROMPTS = Path(__file__).with_name('prompts')


def text(name):
    return (PROMPTS / 'common' / (name + '.txt')).read_text()


def messages(system, user):
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def extract_algorithm_and_code(response):
    """EvolutionOperators._extract_algorithm_and_code; None where upstream raises ValueError."""
    # Extract algorithm description
    algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
    if not algorithm_match:
        return None
    algorithm = algorithm_match.group(1)

    # Extract code
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    code = match.group(1)

    if len(algorithm) == 0 or len(code) == 0:
        return None

    return code, algorithm


def extract_json_block(text):
    """
    Extract JSON content from text, supporting multiple formats:
    1. ```json ... ``` code blocks
    2. Direct JSON arrays or objects
    """
    if not text or not isinstance(text, str):
        return text

    # Try to extract ```json ... ``` code block
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to extract ```..``` code block (without json marker)
    match = re.search(r"```\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to directly find JSON array or object
    # Find array starting with [ and ending with ]
    match = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Find object starting with { and ending with }
    match = re.search(r'(\{\s*".*?\})', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If none found, return original text
    return text.strip()


def ensure_required_imports(code):
    """Automatically add common missing dependencies based on code content."""
    header_lines = []
    stripped_code = code.lstrip()

    needs_numpy = bool(re.search(r"\bnp\.", code))
    has_numpy_import = bool(re.search(r"import\s+numpy\s+as\s+np", code))
    if needs_numpy and not has_numpy_import:
        header_lines.append("import numpy as np")

    if header_lines:
        return "\n".join(header_lines) + "\n\n" + stripped_code
    return code
