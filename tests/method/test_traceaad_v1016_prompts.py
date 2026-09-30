"""Reviewable, byte-for-byte snapshots of the V10.16 generation contracts."""

from pathlib import Path

from tests.support import TinyEvaluation, TokenLLM
from traceaad.v10_16 import Config
from traceaad.v10_16.prompts import PromptBuilder


SNAPSHOTS = Path(__file__).parents[1] / "snapshots" / "v1016"


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
            "reference_id": None}


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
        "crossover": builder.build("Crossover", child, reference=donor)["prompt"],
        "repair": builder.repair("def score(x)\n return x", "Fix the signature.",
                                 "SyntaxError: expected ':'", parent=root)["prompt"],
    }


def test_prompts_match_frozen_text_snapshots():
    for name, prompt in rendered_prompts().items():
        assert prompt + "\n" == (SNAPSHOTS / f"{name}.txt").read_text(encoding="utf-8")


def test_history_trimming_drops_oldest_before_latest_diff():
    sequence = [node(1)]
    for identifier in range(2, 6):
        sequence.append(node(identifier, parent=sequence[-1]))
    archive = {n["id"]: n for n in sequence}
    llm = TokenLLM()
    probe = PromptBuilder(llm, None, ContractEvaluation(), archive, Config())
    summary, _ = probe._formation(sequence, 1, diff=False)
    limit = llm.count_tokens(summary)
    builder = PromptBuilder(llm, None, ContractEvaluation(), archive,
                            Config(history_tokens=limit))
    request = builder.build("Refine", sequence[-1])
    assert request["trims"] == ["oldest_history"] * 3 + ["latest_diff_to_summary"]
    assert request["history_edge_ids"] == [5]
    assert "Code diff" not in request["prompt"]
    assert "Step 5" not in request["prompt"]
    assert "Step 4" in request["prompt"]


def test_initial_reference_trimming_keeps_latest_root():
    roots = [node(i) for i in range(1, 5)]
    llm = TokenLLM()
    archive = {n['id']: n for n in roots}
    probe = PromptBuilder(llm, None, ContractEvaluation(), archive, Config())
    one_root = probe.initial(roots)['prompt'].split('[Algorithms Designed So Far]\n', 1)[1]
    one_root = '[Algorithms Designed So Far]\n' + one_root.split('\n\n[Your Task:', 1)[0]
    # The cap is chosen between one and two displayed roots.
    single = '[Algorithms Designed So Far]\nAlgorithm 1 · Score 4 · Idea: Add 4 to the input.\n'
    single += '```python\ndef score(x):\n    return x + 4\n```'
    assert llm.count_tokens(one_root) > llm.count_tokens(single)
    builder = PromptBuilder(llm, None, ContractEvaluation(), archive,
                            Config(root_tokens=llm.count_tokens(single)))
    request = builder.initial(roots)
    assert request['trims'] == ['root:1', 'root:2', 'root:3']
    assert 'return x + 4' in request['prompt']
    assert 'return x + 1' not in request['prompt']
