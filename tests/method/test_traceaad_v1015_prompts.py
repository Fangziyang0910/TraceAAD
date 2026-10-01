"""Reviewable, byte-for-byte snapshots of the V10.15 generation contracts."""

from pathlib import Path

import pytest

from tests.support import TinyEvaluation, TokenLLM
from traceaad.v10_15 import Config
from traceaad.v10_15.prompts import ContextTooLong, PromptBuilder


SNAPSHOTS = Path(__file__).parents[1] / "snapshots" / "v1015"


class ContractEvaluation(TinyEvaluation):
    def __init__(self):
        super().__init__()
        self.timeout_seconds = 20


def node(identifier, *, parent=None, value=None):
    value = identifier if value is None else value
    return {"id": identifier, "key": str(identifier),
            "code": f"def score(x):\n    return x + {value}\n",
            "score": float(value), "fitness": float(value),
            "parent_id": parent["id"] if parent else None,
            "action": "Refine" if parent else "Init",
            "idea": f"Add {value} to the input.", "depth": parent["depth"] + 1 if parent else 0,
            "reference_id": None, "eval_seconds": 1.5}


def rendered_prompts():
    root = node(1)
    child = node(2, parent=root)
    donor = node(3)
    archive = {n["id"]: n for n in (root, child, donor)}
    builder = PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive, Config())
    roots = [node(i) for i in range(1, 5)]
    return {
        "init_independent": builder.initial([])["prompt"],
        "init_reference": builder.initial(roots)["prompt"],
        "refine_root": builder.build("Refine", root)["prompt"],
        "refine_history": builder.build("Refine", child)["prompt"],
        "explore": builder.build("Explore", child, best_score=3)["prompt"],
        "explore_cards": PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive, Config(explore_cards=4)).build(
            "Explore", child, references=[donor], best_score=3)["prompt"],
        "crossover": builder.build("Crossover", child, reference=donor)["prompt"],
        "repair": builder.repair("def score(x)\n return x", "Fix the signature.",
                                 "SyntaxError: expected ':'", parent=root)["prompt"],
    }


def test_prompts_match_frozen_text_snapshots():
    for name, prompt in rendered_prompts().items():
        assert prompt + "\n" == (SNAPSHOTS / f"{name}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("action", ["Refine", "Crossover"])
def test_history_trimming_drops_whole_oldest_steps_and_keeps_complete_diff(action):
    sequence = [node(1)]
    for identifier in range(2, 6):
        sequence.append(node(identifier, parent=sequence[-1]))
    archive = {n["id"]: n for n in sequence}
    llm = TokenLLM()
    probe = PromptBuilder(llm, None, ContractEvaluation(), archive, Config())
    history, _ = probe._formation(sequence, 1)
    donor = node(6)
    kwargs = {"reference": donor} if action == "Crossover" else {}
    full = probe.build(action, sequence[-1], **kwargs)
    old_history, _ = probe._formation(sequence, 4)
    limit = llm.count_prompt_tokens(full["prompt"].replace(old_history, history))
    builder = PromptBuilder(llm, None, ContractEvaluation(), archive,
                            Config(max_input_tokens=limit))
    request = builder.build(action, sequence[-1], **kwargs)
    assert request["trims"] == ["oldest_history"] * 3
    assert request["history_edge_ids"] == [5]
    assert "Code diff" in request["prompt"] and "Change:" not in request["prompt"]
    assert '-    return x + 4\n+    return x + 5' in request["prompt"]
    assert request["input_tokens"] <= limit
    assert "Step 5" not in request["prompt"]
    assert "Step 4" in request["prompt"]
    if action == "Refine":
        with pytest.raises(ContextTooLong):
            PromptBuilder(llm, None, ContractEvaluation(), archive,
                          Config(max_input_tokens=limit - 1)).build(action, sequence[-1])


def test_large_history_keeps_all_changed_lines_at_each_step():
    root = node(1)
    child = node(2, parent=root)
    child["code"] = ('def score(x):\n' + ''.join(f'    x += {i}\n' for i in range(800))
                     + '    return x\n')
    grandchild = node(3, parent=child)
    grandchild["code"] = child["code"].replace('x += 799', 'x += 999')
    archive = {n["id"]: n for n in (root, child, grandchild)}
    builder = PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive, Config())
    history, _ = builder._formation([root, child, grandchild], 2)
    assert builder.block_count(history) > 3000
    for action in ("Refine", "Crossover"):
        kwargs = {"reference": node(4)} if action == "Crossover" else {}
        request = builder.build(action, grandchild, **kwargs)
        assert request["history_edge_ids"] == [2, 3] and request["trims"] == []
        assert '+    x += 799' in request["prompt"] and '+    x += 999' in request["prompt"]
        assert request["prompt"].count('Code diff (previous → current):') == 2
        assert 'more diff lines not shown' not in request["prompt"]


def test_explore_uses_reference_ideas_without_lineage_or_reference_code():
    root = node(1)
    parent = node(2, parent=root)
    references = [node(i) for i in range(3, 7)]
    archive = {n["id"]: n for n in [root, parent, *references]}
    builder = PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive, Config(explore_cards=4))
    request = builder.build("Explore", parent, references=references, best_score=6)
    assert request["explore_reference_ids"] == [3, 4, 5, 6]
    assert request["history_edge_ids"] == request["reference_history_edge_ids"] == []
    assert "Reference 4 · Score 6 · Design: Add 6" in request["prompt"]
    assert "Earlier Ideas" not in request["prompt"] and "Step 1" not in request["prompt"]
    assert "return x + 3" not in request["prompt"]
    assert request["prompt"].count("```python") == 3
    # A tight total budget removes cards and records exactly what was displayed.
    one = builder.build("Explore", parent, references=references[:1], best_score=6)
    tight = PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive,
                          Config(explore_cards=4, max_input_tokens=one["input_tokens"]))
    trimmed = tight.build("Explore", parent, references=references, best_score=6)
    assert trimmed["explore_reference_ids"] == [3]
    assert trimmed["trims"] == ["explore_reference:6", "explore_reference:5", "explore_reference:4"]
    empty = tight.build("Explore", parent, best_score=6)
    assert empty["action"] == "Explore" and empty["explore_reference_ids"] == []


def test_explore_shows_no_cards_by_default_and_each_program_states_its_cost():
    root = node(1)
    parent = node(2, parent=root)
    references = [node(i) for i in range(3, 7)]
    archive = {n["id"]: n for n in [root, parent, *references]}
    builder = PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive, Config())
    request = builder.build("Explore", parent, references=references, best_score=6)
    prompt = request["prompt"]
    assert request["explore_reference_ids"] == []
    assert "Reference Designs" not in prompt and "[Search Best]" in prompt
    assert "keep the computation efficient" not in prompt and "Returning a previously" not in prompt
    assert "[Current Algorithm]\nScore: 2 · Evaluation time: about 1.5 s" in prompt
    crossover = builder.build("Crossover", parent, reference=references[0])["prompt"]
    assert crossover.count("Evaluation time: about 1.5 s") == 2  # current and reference
    unknown = dict(parent, eval_seconds=None)
    archive[2] = unknown
    assert "Evaluation time" not in PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive, Config()).build(
        "Explore", unknown, best_score=6)["prompt"]


def test_crossover_renders_and_trims_both_histories_independently():
    main, donor = [node(1)], [node(11)]
    for i in range(2, 7):
        main.append(node(i, parent=main[-1]))
        donor.append(node(i + 10, parent=donor[-1]))
    archive = {n["id"]: n for n in [*main, *donor]}
    builder = PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive, Config())
    full = builder.build("Crossover", main[-1], reference=donor[-1])
    assert full["history_edge_ids"] == [3, 4, 5, 6]
    assert full["reference_history_edge_ids"] == [13, 14, 15, 16]
    assert full["prompt"].count("Code diff (previous → current):") == 8
    assert "latest: produced the reference algorithm" in full["prompt"]
    reduced = full["prompt"]
    for sequence, subject, title in [(main, "current", "How the Current Algorithm Was Formed"),
                                     (donor, "reference", "How the Reference Algorithm Was Formed")]:
        old, _ = builder._formation(sequence, 4, title=title, subject=subject)
        recent, _ = builder._formation(sequence, 1, title=title, subject=subject)
        reduced = reduced.replace(old, recent)
    limit = builder.count(reduced)
    tight = PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive, Config(max_input_tokens=limit))
    request = tight.build("Crossover", main[-1], reference=donor[-1])
    assert request["action"] == "Crossover" and request["input_tokens"] <= limit
    assert request["history_edge_ids"] == [6] and request["reference_history_edge_ids"] == [16]
    assert request["trims"].count("oldest_history") == 3
    assert request["trims"].count("oldest_reference_history") == 3
    assert request["prompt"].count("Code diff (previous → current):") == 2
    fallback = PromptBuilder(TokenLLM(), None, ContractEvaluation(), archive,
                             Config(max_input_tokens=limit - 1)).build("Crossover", main[-1], reference=donor[-1])
    assert fallback["action"] == "Refine" and fallback["reference_history_edge_ids"] == []


def test_initial_reference_trimming_keeps_latest_root():
    roots = [node(i) for i in range(1, 5)]
    llm = TokenLLM()
    archive = {n['id']: n for n in roots}
    probe = PromptBuilder(llm, None, ContractEvaluation(), archive, Config())
    one_root = probe.initial(roots)['prompt'].split('[Algorithms Designed So Far]\n', 1)[1]
    one_root = '[Algorithms Designed So Far]\n' + one_root.split('\n\n[Your Task:', 1)[0]
    # The cap is chosen between one and two displayed roots.
    single = '[Algorithms Designed So Far]\nAlgorithm 1 · Score: 4 · Evaluation time: about 1.5 s · Design: Add 4 to the input.\n'
    single += '```python\ndef score(x):\n    return x + 4\n```'
    assert llm.count_tokens(one_root) > llm.count_tokens(single)
    builder = PromptBuilder(llm, None, ContractEvaluation(), archive,
                            Config(root_tokens=llm.count_tokens(single)))
    request = builder.initial(roots)
    assert request['trims'] == ['root:1', 'root:2', 'root:3']
    assert 'return x + 4' in request['prompt']
    assert 'return x + 1' not in request['prompt']
