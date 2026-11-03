"""Separate structured outputs for generation, parent choice, short-term and long-term reflection."""

from copy import deepcopy

GENERATOR_SCHEMA = {'type': 'object',
 'properties': {'code': {'type': 'string'},
                'analysis': {'type': 'string'},
                'design': {'type': 'string'}},
 'required': ['code', 'analysis', 'design'],
 'additionalProperties': False}

REFLECTOR_SCHEMA = {'type': 'object',
 'properties': {'reflection': {'type': 'string'}},
 'required': ['reflection'],
 'additionalProperties': False}

SELECTOR_SCHEMA = {'type': 'object',
 'properties': {'assessment': {'type': 'string'}, 'choice': {'type': 'integer'}},
 'required': ['assessment', 'choice'],
 'additionalProperties': False}

_EXPERIENCE = {'type': 'object', 'properties': {'summary': {'type': 'string'}, 'applicable_when': {'type': 'string'}},
           'required': ['summary', 'applicable_when'], 'additionalProperties': False}

LONG_TERM_SCHEMA = {'type': 'object',
 'properties': {'positive': {'type': 'array', 'items': _EXPERIENCE},
                'negative': {'type': 'array', 'items': _EXPERIENCE}},
 'required': ['positive', 'negative'],
 'additionalProperties': False}

SCHEMAS = {"generator": GENERATOR_SCHEMA, "selector": SELECTOR_SCHEMA, "reflector": REFLECTOR_SCHEMA, "long_term_reflector": LONG_TERM_SCHEMA}


def response_format(role):
    schema = SCHEMAS[role]
    return {"type": "json_schema", "json_schema": {"name": "traceaad_v194_" + role, "strict": True, "schema": deepcopy(schema)}}
