"""Separate structured outputs for the two model roles."""

from copy import deepcopy

GENERATOR_SCHEMA = {'type': 'object',
 'properties': {'code': {'type': 'string'},
                'analysis': {'type': 'string'},
                'design': {'type': 'string'}},
 'required': ['code', 'analysis', 'design'],
 'additionalProperties': False}

REFLECTOR_SCHEMA = {'type': 'object',
 'properties': {'positive': {'type': 'array',
                             'items': {'type': 'object',
                                       'properties': {'summary': {'type': 'string'},
                                                      'recommendations': {'type': 'array',
                                                                          'items': {'type': 'string'}},
                                                      'applicable_when': {'type': 'string'}},
                                       'required': ['summary',
                                                    'recommendations',
                                                    'applicable_when'],
                                       'additionalProperties': False}},
                'negative': {'type': 'array',
                             'items': {'type': 'object',
                                       'properties': {'summary': {'type': 'string'},
                                                      'recommendations': {'type': 'array',
                                                                          'items': {'type': 'string'}},
                                                      'applicable_when': {'type': 'string'}},
                                       'required': ['summary',
                                                    'recommendations',
                                                    'applicable_when'],
                                       'additionalProperties': False}}},
 'required': ['positive', 'negative'],
 'additionalProperties': False}

def response_format(role):
    schema = REFLECTOR_SCHEMA if role == "reflector" else GENERATOR_SCHEMA
    return {"type": "json_schema", "json_schema": {"name": "traceaad_v193_" + role, "strict": True, "schema": deepcopy(schema)}}
