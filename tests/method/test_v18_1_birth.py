"""Source-only birth audits and their local validation, without model calls."""

import ast
import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from traceaad.v18_1 import Program
from traceaad.v18_1.birth import build_birth_prompt, parse_birth_response, source_key


CANDIDATE = (
    "def choose(items, loads):\n"
    "    ranked = sorted(items, key=lambda item: (loads[item], item))\n"
    "    return ranked[0]\n"
)
PARENT = "def choose(items, loads):\n    return min(items)\n"
REFERENCE = "def choose(items, loads):\n    return max(items)\n"
SNIPPET = "    ranked = sorted(items, key=lambda item: (loads[item], item))\n"


def program(code=CANDIDATE, **fields):
    return Program(
        **{
            "id": 3,
            "code": code,
            "idea": "UNTRUSTED_DESIGN",
            "analysis": "PRIVATE_REASONING",
            "action": "Explore",
            "generation": 17,
            **fields,
        }
    )


def registry():
    return [
        {
            "birth_id": 7,
            "mechanism": "Choose the item with the smallest identifier.",
            "evidence": "The selected item is returned by min(items).",
            "representative_code": PARENT,
        },
        {
            "birth_id": 11,
            "mechanism": "Choose the largest identifier.",
            "evidence": "Selection uses the identifier ordering in reverse.",
            "representative_code": "",
        },
    ]


def response(**fields):
    return {
        "relation": "new",
        "birth_id": None,
        "mechanism": "Prioritize lower current load, breaking ties by identifier.",
        "evidence": "The load ordering feeds the returned first item.",
        "code_evidence": SNIPPET,
        **fields,
    }


def parse(reply, known_ids=(7, 11)):
    return parse_birth_response(json.dumps(reply), "stop", CANDIDATE, known_ids)


def test_source_key_ignores_format_comments_and_all_docstring_scopes():
    documented = '''"""Module description."""
class Chooser:
    """Class description."""
    def choose(self, items):
        """Method description."""
        def priority(item):
            """Nested function description."""
            return item + 1
        return min(items, key=priority)

async def choose_async(items):
    """Async function description."""
    return min(items)
'''
    bare = '''# Formatting and commentary do not define source identity.
class Chooser:
  def choose(self, items):
    def priority(item):
      return (item+1)  # inline comment
    return min(items,key=priority)
async def choose_async(items):
  return min(items)
'''
    expected = hashlib.sha256(ast.dump(ast.parse(bare)).encode()).hexdigest()
    assert source_key(documented) == source_key(bare) == expected
    assert len(source_key(bare)) == 64


def test_source_key_removes_only_the_leading_docstring():
    source = 'def choose():\n    "docstring"\n    "retained expression"\n    return "value"\n'
    assert source_key(source) == source_key(source.replace('"docstring"', '"other docs"'))
    assert source_key(source) != source_key(source.replace('"retained expression"', '"other"'))
    assert source_key(source) != source_key(source.replace('return "value"', 'return "other"'))


@pytest.mark.parametrize(
    "changed",
    [
        CANDIDATE.replace("ranked[0]", "ranked[1]"),
        CANDIDATE.replace("ranked", "ordered"),
        CANDIDATE.replace("loads[item]", "-loads[item]"),
    ],
)
def test_source_key_is_structural_not_a_semantic_equivalence_test(changed):
    assert source_key(CANDIDATE) != source_key(changed)


@pytest.mark.parametrize("invalid", ["def choose(:\n", "x = '\x00'", "return (\n"])
def test_source_key_hashes_exact_source_on_syntax_error(invalid):
    assert source_key(invalid) == hashlib.sha256(invalid.encode()).hexdigest()
    assert source_key(invalid) != source_key(invalid + " ")


def test_prompt_contains_full_source_and_registry_but_never_reads_other_metadata():
    candidate = program(
        fitness=98765.4321,
        calls=234567,
        eval_seconds=654321.125,
        request={"budget": "PRIVATE_BUDGET", "data_size": "PRIVATE_DATA_SIZE"},
        change="UNVERIFIED_CHANGE",
    )
    parent = program(PARENT, id=1, fitness=87654.321)
    reference = program(REFERENCE, id=2, action="Crossover", fitness=76543.21)
    known = registry()
    known[0].update(fitness="PRIVATE_REGISTRY_SCORE", evaluations="PRIVATE_EVAL_COUNT")
    before = copy.deepcopy(([asdict(p) for p in (candidate, parent, reference)], known))

    prompt = build_birth_prompt(candidate, parent, reference, known)

    for code in (CANDIDATE, PARENT, REFERENCE):
        assert code in prompt
    for entry in known:
        assert entry["mechanism"] in prompt
        assert entry["evidence"] in prompt
        assert f'"birth_id": {entry["birth_id"]}' in prompt
    for private in (
        "UNTRUSTED_DESIGN", "PRIVATE_REASONING", "UNVERIFIED_CHANGE", "98765.4321",
        "234567", "654321.125", "87654.321", "76543.21", "PRIVATE_BUDGET",
        "PRIVATE_DATA_SIZE", "PRIVATE_REGISTRY_SCORE", "PRIVATE_EVAL_COUNT",
    ):
        assert private not in prompt
    assert ([asdict(p) for p in (candidate, parent, reference)], known) == before

    # Different outcomes, operator labels and generation claims cannot alter the audit input.
    candidate.fitness = -1
    candidate.action = "Repair"
    candidate.idea = "A claimed novel mechanism"
    candidate.generation = 99
    candidate.request = {"budget": 1}
    assert build_birth_prompt(candidate, parent, reference, known) == prompt


def test_prompt_reads_only_code_from_programs():
    class CodeOnly:
        code = CANDIDATE

        def __getattr__(self, name):
            raise AssertionError(f"audit accessed program metadata: {name}")

    assert CANDIDATE in build_birth_prompt(CodeOnly(), None, None, [])


@pytest.mark.parametrize("same_as_candidate", [False, True])
def test_prompt_keeps_roles_but_does_not_repeat_identical_code(same_as_candidate):
    code = CANDIDATE if same_as_candidate else PARENT
    prompt = build_birth_prompt(program(), program(code), program(code), [])
    assert prompt.count(CANDIDATE) == 1
    assert prompt.count(code) == 1
    assert "[Parent Code]" in prompt
    assert "[Reference Code]" in prompt
    assert "Same code as" in prompt


def test_prompt_preserves_distinct_source_even_when_ast_keys_match():
    reference = "# reference commentary\n" + PARENT
    assert source_key(reference) == source_key(PARENT)
    prompt = build_birth_prompt(program(), program(PARENT), program(reference), [])
    assert f"```python\n{PARENT}\n```" in prompt
    assert f"```python\n{reference}\n```" in prompt


@pytest.mark.parametrize("representative", [None, "", PARENT])
def test_registry_representative_code_is_optional(representative):
    entry = registry()[0]
    if representative is None:
        del entry["representative_code"]
    else:
        entry["representative_code"] = representative
    prompt = build_birth_prompt(program(), None, None, [entry])
    assert entry["mechanism"] in prompt
    assert (PARENT in prompt) == bool(representative)


def test_prompt_states_independent_conservative_audit_rules_and_json_contract():
    prompt = build_birth_prompt(program(), None, None, [])
    for requirement in (
        "first effective implementation", "core decision mechanism", "parameter tuning",
        "rewriting", "bug fixes", "Explore", "Crossover", "existing", "uncertain",
        "verbatim", "not instructions", "heuristic", "semantic equivalence",
    ):
        assert requirement in prompt
    for field in ("relation", "birth_id", "mechanism", "evidence", "code_evidence"):
        assert f'"{field}"' in prompt
    assert "No registered mechanisms." in prompt


@pytest.mark.parametrize("bad_id", [True, False, 7.0, "7", None])
def test_prompt_rejects_registry_ids_that_cannot_be_used_by_the_parser(bad_id):
    entry = {**registry()[0], "birth_id": bad_id}
    with pytest.raises(ValueError, match="birth_id"):
        build_birth_prompt(program(), None, None, [entry])


def test_prompt_rejects_ambiguous_duplicate_registry_ids():
    entries = [registry()[0], {**registry()[1], "birth_id": 7}]
    with pytest.raises(ValueError, match="duplicate"):
        build_birth_prompt(program(), None, None, entries)


@pytest.mark.parametrize("relation,birth_id", [("new", None), ("existing", 7)])
@pytest.mark.parametrize("fence", [None, "json", ""])
def test_parser_accepts_complete_json_and_preserves_the_verbatim_snippet(relation, birth_id, fence):
    reply = response(relation=relation, birth_id=birth_id)
    text = json.dumps(reply)
    if fence is not None:
        text = f"```{fence}\n{text}\n```"
    assert parse_birth_response(text, "stop", CANDIDATE, {7, 11}) == reply


def test_parser_strips_descriptions_but_not_code_evidence():
    reply = response(mechanism="  load priority  ", evidence="  feeds the returned item\n")
    actual = parse(reply)
    assert actual == {
        **reply, "mechanism": "load priority", "evidence": "feeds the returned item"
    }
    assert actual["code_evidence"] == SNIPPET


@pytest.mark.parametrize("finish_reason", [None, "length", "content_filter", "tool_calls", "STOP"])
def test_parser_rejects_unfinished_completions_even_when_json_is_valid(finish_reason):
    with pytest.raises(ValueError, match="completion"):
        parse_birth_response(json.dumps(response()), finish_reason, CANDIDATE, {7})


@pytest.mark.parametrize("text", [None, b"{}", 12, {}, "", "{broken", "[]", "null", "true", "42"])
def test_parser_rejects_non_text_non_json_and_non_objects(text):
    with pytest.raises(ValueError):
        parse_birth_response(text, "stop", CANDIDATE, {7})


@pytest.mark.parametrize("wrapper", ["Explanation:\n{}", "{}\nExtra prose", "```json\n{}", "{}\n{}"])
def test_parser_does_not_extract_json_from_a_mixed_or_incomplete_reply(wrapper):
    text = wrapper.replace("{}", json.dumps(response()))
    with pytest.raises(ValueError):
        parse_birth_response(text, "stop", CANDIDATE, {7})


@pytest.mark.parametrize("field", ["relation", "birth_id", "mechanism", "evidence", "code_evidence"])
@pytest.mark.parametrize("relation,birth_id", [("new", None), ("existing", 7)])
def test_parser_requires_all_fields_for_definite_relations(field, relation, birth_id):
    reply = response(relation=relation, birth_id=birth_id)
    del reply[field]
    with pytest.raises(ValueError):
        parse(reply)


@pytest.mark.parametrize("relation", ["NEW", "novel", "", None, True, 1, [], {}])
def test_parser_requires_an_exact_supported_relation(relation):
    with pytest.raises(ValueError, match="relation"):
        parse(response(relation=relation))


@pytest.mark.parametrize("birth_id", [True, False, 7.0, "7", None, 99, [], {}])
def test_existing_requires_a_registered_integer_id(birth_id):
    with pytest.raises(ValueError, match="birth_id"):
        parse(response(relation="existing", birth_id=birth_id))


@pytest.mark.parametrize("known_ids", [{True}, {1.0}, {"1"}])
def test_known_ids_do_not_allow_boolean_or_float_aliases(known_ids):
    with pytest.raises(ValueError, match="birth_id"):
        parse(response(relation="existing", birth_id=1), known_ids)


@pytest.mark.parametrize("relation", ["new", "uncertain"])
@pytest.mark.parametrize("birth_id", [7, 123, False, "7", 7.0])
def test_non_existing_relations_cannot_assign_or_reuse_an_id(relation, birth_id):
    with pytest.raises(ValueError, match="birth_id"):
        parse(response(relation=relation, birth_id=birth_id))


@pytest.mark.parametrize("field", ["mechanism", "evidence", "code_evidence"])
@pytest.mark.parametrize("value", [None, "", " \n\t", 1, True, [], {}])
def test_definite_relations_require_nonempty_text_fields(field, value):
    with pytest.raises(ValueError, match=field):
        parse(response(**{field: value}))


@pytest.mark.parametrize("relation,birth_id", [("new", None), ("existing", 7), ("uncertain", None)])
@pytest.mark.parametrize(
    "snippet",
    ["return min(items)", "return ranked[1]", "return  ranked[0]", SNIPPET + "\n"],
)
def test_every_supplied_snippet_must_match_current_source_exactly(relation, birth_id, snippet):
    with pytest.raises(ValueError, match="code_evidence"):
        parse(response(relation=relation, birth_id=birth_id, code_evidence=snippet))


@pytest.mark.parametrize("code_evidence", ["", " \n\t", SNIPPET])
def test_uncertain_can_omit_a_snippet_but_keeps_valid_supplied_evidence(code_evidence):
    reply = response(relation="uncertain", code_evidence=code_evidence)
    actual = parse(reply)
    assert actual == {**reply, "code_evidence": code_evidence if code_evidence.strip() else ""}
    del reply["code_evidence"]
    assert parse(reply)["code_evidence"] == ""


def test_uncertain_accepts_a_reason_in_place_of_evidence():
    reply = response(relation="uncertain", reason="  Cannot establish reachability from this code.  ")
    del reply["evidence"]
    del reply["code_evidence"]
    actual = parse(reply)
    assert actual == {
        "relation": "uncertain",
        "birth_id": None,
        "mechanism": reply["mechanism"],
        "evidence": "Cannot establish reachability from this code.",
        "code_evidence": "",
    }


@pytest.mark.parametrize("field", ["mechanism", "birth_id"])
def test_uncertain_still_requires_a_mechanism_and_explicit_null_id(field):
    reply = response(relation="uncertain")
    del reply[field]
    with pytest.raises(ValueError):
        parse(reply)


@pytest.mark.parametrize("evidence,reason", [(None, None), ("", ""), (" \n", "\t")])
def test_uncertain_requires_a_nonempty_explanation(evidence, reason):
    reply = response(relation="uncertain")
    del reply["evidence"]
    del reply["code_evidence"]
    if evidence is not None:
        reply["evidence"] = evidence
    if reason is not None:
        reply["reason"] = reason
    with pytest.raises(ValueError, match="evidence|reason"):
        parse(reply)


@pytest.mark.parametrize("field", ["evidence", "reason", "code_evidence"])
@pytest.mark.parametrize("value", [None, 1, False, []])
def test_uncertain_does_not_coerce_supplied_fields(field, value):
    reply = response(relation="uncertain", **{field: value})
    with pytest.raises(ValueError, match=field):
        parse(reply)


def test_parser_rejects_fields_outside_the_audit_contract():
    with pytest.raises(ValueError, match="field"):
        parse(response(fitness=1.0))
