from __future__ import annotations
import re
import io
import tokenize
from core.code import function_from_source, parse_program
from .parsing import ResponseRules, extract_code_from_generator


class PathWiseSampler:
    def __init__(self, llm, template_program):
        self.llm = llm
        self._template_program = template_program

    def draw_sample(self, prompt, *args, **kwargs):
        return self.llm.draw_sample(prompt, *args, **kwargs)

    @staticmethod
    def with_global_imports(program):
        return parse_program(program) if isinstance(program, str) else program

    @staticmethod
    def response_to_function(response, template_program):
        template = parse_program(template_program) if isinstance(template_program, str) else template_program
        code = extract_code_from_generator(response)
        if code is None:
            return None
        name = template.functions[0].name
        code = tokenize.untokenize(token._replace(string=name)
            if token.type == tokenize.NAME and token.string == name + '_v2' else token
                for token in tokenize.generate_tokens(io.StringIO(code).readline))
        return function_from_source(code, name)

    @classmethod
    def parse_initialization_response(cls, response, template_program):
        try:
            description, rationale = ResponseRules().parse_initialization_response(response)
            function = cls.response_to_function(response, template_program)
            return (function, description, rationale) if function else None
        except (ValueError, tokenize.TokenError):
            return None

    @classmethod
    def parse_world_model_response(cls, response, template_program):
        try:
            code, description = ResponseRules().parse_world_model_response(response)
            template = parse_program(template_program) if isinstance(template_program, str) else template_program
            name = template.functions[0].name
            code = tokenize.untokenize(token._replace(string=name)
                if token.type == tokenize.NAME and token.string == name + '_v2' else token
                for token in tokenize.generate_tokens(io.StringIO(code).readline))
            function = function_from_source(code, name)
            return (function, description) if function else None
        except (ValueError, tokenize.TokenError):
            return None

    @staticmethod
    def parse_policy_response(response, state):
        rules = ResponseRules()
        rules.invalid_parent_selections = 0
        rules.parent_selection_counts = []
        return rules.parse_policy_response(response, state, None)
