from __future__ import annotations

import ast
import re

from core.code import function_from_source, parse_program


class EoHSampler:
    # EoH 47254578, eoh/evolution.py::_extract and _call_llm.
    def __init__(self, llm, template_program, profiler=None):
        self.llm = llm
        self._template_program = parse_program(template_program) if isinstance(template_program, str) else template_program
        self._profiler = profiler

    def _extract(self, response: str):
        if not response:
            return [], []

        # ── code ──────────────────────────────────────────────────────────────
        # 1. Fenced code blocks (most reliable)
        code = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)

        if not code:
            # 2. Locate the first top-level Python statement at the start of a line
            #    (import / from / def / class / decorator), then trim trailing prose
            #    by iteratively removing the last line until the snippet parses.
            start = re.search(r'^(?:import |from |def |class |@)', response, re.MULTILINE)
            if start:
                candidate = response[start.start():].strip()
                lines = candidate.splitlines()
                for trim in range(len(lines)):
                    snippet = '\n'.join(lines[:len(lines) - trim]).strip()
                    if not snippet:
                        break
                    try:
                        ast.parse(snippet)
                        code = [snippet]
                        break
                    except SyntaxError:
                        continue

        # Strip any leading {description} line the LLM sometimes puts inside the code block
        code = [re.sub(r'^\s*\{[^}]*\}\s*\n+', '', c, flags=re.DOTALL).strip() for c in code]
        code = [c for c in code if c]

        # ── algorithm description ──────────────────────────────────────────────
        # Search only in text BEFORE the code to avoid matching Python dict literals.
        if '```' in response:
            pre_code = response[:response.find('```')].strip()
        elif code:
            # Find where the extracted code begins in the original response
            idx = response.find(code[0][:60]) if code[0] else -1
            pre_code = response[:idx].strip() if idx > 0 else response.strip()
        else:
            pre_code = response.strip()

        # Require at least 8 chars to skip empty {}, single-letter vars, dict snippets
        algorithm = re.findall(r'\{([^{}]{8,})\}', pre_code)

        if not algorithm and pre_code:
            # Fall back: everything before the code is treated as the description
            algorithm = [pre_code]

        return algorithm, code

    def get_thought_and_function(self, prompt, *, operator=None, sample_order=None):
        for attempt in range(4):
            response = self.llm.draw_sample(prompt)
            algorithms, codes = self._extract(response)
            logger = getattr(self._profiler, 'log_llm_call', None)
            if callable(logger):
                logger(llm=self.llm, stage='generate', operator=operator,
                       sample_order=sample_order, attempt=attempt, prompt=prompt,
                       response=response, thought_parse_success=bool(algorithms),
                       function_parse_success=bool(codes))
            if algorithms and codes:
                code = codes[0]
                missing = [line for line in self._template_program.preface.splitlines()
                           if line.startswith(('import ', 'from ')) and line not in code]
                code = '\n'.join(missing + [code])
                return algorithms[0], function_from_source(code, self._template_program.functions[0].name)
        return None, None
