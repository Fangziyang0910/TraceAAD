"""HSEvo f3afffab utils/utils.py: code extraction, prompt filtering and harmony-search parsing."""
import ast
import io
import json
import re
import subprocess
import sys
import tokenize

from core.code import function_from_source

# Upstream also imports torch; it is not installed for the shared evaluator and no task uses it.
GLOBAL_IMPORTS = "import numpy as np\nimport random\nimport math\nimport scipy\n"
# Upstream execs the parameter_ranges block in its own process; here it runs in a child process.
RANGES_SCRIPT = ("import json, sys\nnamespace = {}\nexec(sys.stdin.read(), namespace)\n"
                 "print(json.dumps({str(k): [float(x) for x in v] "
                 "for k, v in namespace['parameter_ranges'].items()}))")
RANGES_TIMEOUT = 10


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end + 1])

    if code_string is None:
        return None

    code_string = GLOBAL_IMPORTS + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def parameter_ranges(block, timeout=RANGES_TIMEOUT):
    """The block's `parameter_ranges` as {name: (begin, end)}, or None when upstream would fail."""
    try:
        done = subprocess.run([sys.executable, '-I', '-c', RANGES_SCRIPT], input=block, text=True,
                              capture_output=True, timeout=timeout)
        ranges = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else None
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    # Upstream unpacks each value into two bounds; other shapes would end its run.
    if not isinstance(ranges, dict) or any(len(value) != 2 for value in ranges.values()):
        return None
    return {name: tuple(value) for name, value in ranges.items()}


def extract_to_hs(input_string: str):
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        if any(keyword in parameter_ranges_block for keyword in ['inf', 'np.inf', 'None']):
            return None, None
        ranges = parameter_ranges(parameter_ranges_block)
        if ranges is None:
            return None, None
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i:i + 3] == 'def':
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if char == ':' and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[signature_start_index:signature_end_index + 1]

        # Clean up the function signature from potential default values that might be corrupted (e.g. .eps suffix)
        # This regex looks for parameter definitions and cleans any trailing garbage before the next comma or closing paren
        function_signature = re.sub(r'(\w+\s*:\s*\w+\s*=\s*[\d.e-]+)(\.[a-zA-Z]+)', r'\1', function_signature)

        for param in ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(pattern, replacement, function_signature, flags=re.DOTALL)
        function_block = function_block[:signature_start_index] + function_signature + function_block[
                                                                                       signature_end_index + 1:]

    return ranges, function_block


def rename(code, old, new):
    """Rename one identifier token, or return None for code that does not tokenize."""
    try:
        return tokenize.untokenize(token._replace(string=new)
                                   if token.type == tokenize.NAME and token.string == old else token
                                   for token in tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None


def code_to_function(code, name):
    """The task callable, found as upstream eval.py's get_heuristic_name finds it."""
    try:
        defined = {node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
    except (SyntaxError, ValueError):
        return None
    found = next((n for n in (name, name + '_v1', name + '_v2', name + '_v3') if n in defined), None)
    renamed = rename(code, found, name) if found is not None else None
    return function_from_source(renamed, name) if renamed is not None else None
