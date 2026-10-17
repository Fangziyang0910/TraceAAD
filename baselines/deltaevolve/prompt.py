"""Appendix C templates and strict SEARCH/REPLACE plus delta parsing."""

import ast
import re

from core.code import Program, parse_function
from baselines.shinka_evo.sampler import PATCH_PATTERN


DELTA_LOGGING = """---------------------- Delta Logging Instructions (New in DeltaEvolve) ---------------
You MUST summarize your changes at the very end of your response using the strict
format below. This log is critical for the evolution memory system. Failure to
follow these rules will break the experiment.

###CRITICAL RULES for DELTA SUMMARY:
1. NO META-TALK: Do NOT say Replace 22 lines with 230 lines, Updated code, or Changed logic.
2. ALGORITHMIC ONLY: Describe the strategy change (e.g., Switched from Greedy to Simulated Annealing).
3. NO TEMPLATES: Do NOT output placeholder text like high-level plan summary in one sentence. Write actual content.
4. NO CODE SYNTAX: Do not write Python code in the summary (e.g., no def function()). Use natural language.

#DELTA-SUMMARY-START
FROM: <One-sentence summary of the OLD strategy (the parent node's approach)>
TO: <One-sentence summary of the NEW strategy (your current approach)>
#DELTA-SUMMARY-END

###CRITICAL RULES for DELTA PLAN DETAILS
You must explain HOW and WHY the logic changed using the strict format below.
1. Target Audience: A researcher trying to reproduce your experiment without seeing the code.
2. BE QUANTITATIVE: Do not say increased parameters. Say increased grid resolution from 10 to 50.
3. NAME THE ALGORITHM: Use standard terminology (e.g., Coordinate Descent, Simulated Annealing, Penalty Method).
4. NO META-TALK: Do not say I defined a function. Describe the LOGIC flow.

#DELTA-PLAN-DETAILS-START
[Modification 1]
COMPONENT: <The specific module changed. e.g., Initialization Strategy, Constraint Handling, Optimization Loop>
OLD LOGIC: <Brief summary of what was removed. e.g., Random noise injection>
NEW LOGIC: <DETAILED mechanism. MUST include specific hyperparameters, formulas used, or heuristic rules.>
HYPOTHESIS: <The scientific reasoning.>

[Modification 2] (If applicable)
COMPONENT: ...
OLD LOGIC: ...
NEW LOGIC: ...
HYPOTHESIS: ...
#DELTA-PLAN-DETAILS-END"""

PATCH_INSTRUCTIONS = (
    'Suggest improvements to the program that will improve its FITNESS SCORE. The system\n'
    'maintains diversity across these dimensions: complexity, diversity Different\n'
    'solutions with similar fitness but different features are valuable.\n'
    'You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:\n'
    '<<<<<<< SEARCH\n'
    '<Original code to find and replace (must match exactly)>\n'
    '=======\n'
    '<New replacement code>\n'
    '>>>>>>> REPLACE'
)


def build_prompt(task, parent, top, diverse, iteration, recent_window):
    system = "--------------------------------- Problem Description ---------------------------------\n"
    system += task + "\n\n" + DELTA_LOGGING
    user = ("---------------------------- Current Program Information ----------------------------\n"
            "- Focus areas: Improve solution quality within the task constraints.\n"
            f"- Feedback: FITNESS SCORE to maximize: {-parent['fitness']}\n"
            "------------ Inspirations: Delta of History Nodes (New in DeltaEvolve) ------------\n"
            "Below we provide history delta plans from prior nodes, including top-performing\n"
            "programs and highly diverse alternatives to the parent. Each delta highlights\n"
            "concrete strategy differences that may inspire new solution directions rather than\n"
            "direct reuse.\n")
    rendered = []
    for label, nodes in (("Top", top), ("Diverse", diverse)):
        for index, node in enumerate(nodes, 1):
            level = 2 if node["iteration"] > iteration - recent_window else 1
            content = node["plan"] if level == 2 else node["summary"]
            user += (f"\n### {label} Delta Plan {index}\n"
                     f"[iter: {node['iteration']} | {node['change']}]\n{content}\n"
                     f"- Feedback: {node['change']}\n")
            rendered.append({"node_id": node["id"], "level": level, "role": label.lower()})
    user += "\n----------------------------------- Parent Program -----------------------------------\n"
    user += parent["code"] + "\n" + PATCH_INSTRUCTIONS
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], rendered


def parse_response(response, parent_code, template):
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    text = text.replace('\r\n', '\n')
    text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*(DELTA-(?:SUMMARY|PLAN-DETAILS)-(?:START|END))[ \t]*$",
                  r"#\1", text)
    parts = []
    end = 0
    for label, fields in (("SUMMARY", ("FROM", "TO")),
                          ("PLAN-DETAILS", ("COMPONENT", "OLD LOGIC", "NEW LOGIC", "HYPOTHESIS"))):
        markers = [f"#DELTA-{label}-{suffix}" for suffix in ("START", "END")]
        pattern = rf"(?m)^{markers[0]}[ \t]*\n(.*?)^{markers[1]}[ \t]*$"
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))
        if len(matches) != 1 or any(text.count(marker) != 1 for marker in markers):
            raise ValueError(f"Expected one complete DELTA-{label} block")
        if matches[0].start() < end:
            raise ValueError("Delta blocks must follow SUMMARY then PLAN-DETAILS order")
        end = matches[0].end()
        content = matches[0].group(1).strip()
        entries = re.split(r"\[Modification\s+\d+\]", content)[1:] if label == "PLAN-DETAILS" else [content]
        if not entries:
            raise ValueError(f"Incomplete DELTA-{label} fields")
        for entry in entries:
            values = re.split(r"(?m)^(" + "|".join(fields) + r"):[ \t]*", entry)
            if tuple(values[1::2]) != fields or any(not value.strip() for value in values[2::2]):
                raise ValueError(f"Incomplete or duplicate DELTA-{label} fields")
        parts.append(content)
    patch_text = text.split("#DELTA-SUMMARY-START", 1)[0]
    patches = PATCH_PATTERN.findall(patch_text)
    if not patches or patch_text.count("<<<<<<< SEARCH") != len(patches):
        raise ValueError("No SEARCH/REPLACE patch before delta logs")
    code = parent_code
    for search, replacement in patches:
        if not search.strip() or code.count(search) != 1:
            raise ValueError("SEARCH must match exactly one nonempty region")
        code = code.replace(search, replacement, 1)
    return program_from_code(code, template), parts[0], parts[1]


def program_from_code(code, template):
    """Preserve the patched module; a completion trimmer may silently discard valid code."""
    tree = ast.parse(code)
    compile(tree, "<deltaevolve>", "exec")
    target = template.functions[0]
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == target.name]
    if len(entries) != 1 or ast.unparse(entries[0].args) != target.args:
        raise ValueError("Patch changed or removed the task entrypoint contract")
    # Normalize one-line suites before splitting the entrypoint from its module.
    code = ast.unparse(tree)
    entry = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == target.name)
    function = parse_function(ast.get_source_segment(code, entry))
    lines = code.splitlines(keepends=True)
    function.source_prefix = ''.join(lines[:entry.lineno - 1])
    function.source_suffix = ''.join(lines[entry.end_lineno:])
    program = Program(preface='', functions=[function])
    compile(str(program), "<deltaevolve>", "exec")
    return program
