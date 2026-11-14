from copy import deepcopy
import math

import pytest

from evotrace_v2.allocation import parent_probabilities
from evotrace_v2.context import ContextConfig, build_evolution_context
from evotrace_v2.models import Edge, Node, TreePopulation
from evotrace_v2.reflection import build_variation_prompt, reflect_and_apply
from evotrace_v2.topology import ConsolidationPlan, PathContraction
from evotrace_v2.topology import TopologyConfig, quality_representatives


def add(tree, parent, score, code, op='Refine'):
    nid = tree.latest_node_id + 1
    tree.add(Node(nid, code, f'Design {nid}', score),
             Edge(parent, nid, op if parent else None), evaluations=1)
    return nid


def test_recent_clones_do_not_change_relative_weight_when_representatives_stay_fixed():
    tree = TreePopulation()
    a = add(tree, 0, 1., 'def solve(x): return x + 1')
    b = add(tree, 0, 2., 'def solve(x): return sum(x)')
    cfg = TopologyConfig(quality_floor=0, clone_distance=0)
    before = parent_probabilities(tree, topology_config=cfg)
    for _ in range(15):
        add(tree, 0, 100., tree.nodes[b].code)
    assert quality_representatives(tree, cfg) == {a, b}
    after = parent_probabilities(tree, topology_config=cfg)
    assert after[a] / after[b] == pytest.approx(before[a] / before[b])
    assert set(after) == set(tree.nodes)
    assert sum(after.values()) == pytest.approx(1.)


def test_quality_configuration_and_degenerate_scale_are_respected():
    tree = TreePopulation()
    for i in range(6):
        add(tree, 0, float(i), f'def solve(x): return x + {i}')
    wide = TopologyConfig(quality_floor=0, clone_distance=0)
    champion = TopologyConfig(quality_floor=1, clone_distance=0)
    p = parent_probabilities(tree, topology_config=champion)
    assert quality_representatives(tree, champion) == {1}
    assert p[1] > parent_probabilities(tree, topology_config=wide)[1]
    assert all(math.isfinite(x) and x > 0 for x in p.values())
    assert sum(p.values()) == pytest.approx(1.)


def test_route_views_preserve_entry_evidence_without_expanding_old_routes():
    tree = TreePopulation()
    ancient = add(tree, 0, 9., 'def solve(x): return ancient_code(x)')
    origin = add(tree, ancient, 8., 'def solve(x): return origin_code(x)', 'Explore')
    entry = add(tree, origin, 7., 'def solve(x): return entry_code(x)', 'Explore')
    current = add(tree, entry, 6., 'def solve(x): return current_code(x)')
    ref = add(tree, origin, 5., 'def solve(x): return reference_code(x)', 'Explore')
    tree.edges[entry].variation_experience = 'ENTRY_EVIDENCE'
    tree.nodes[origin].selection_experience = 'ORIGIN_EVIDENCE'
    tree.nodes[ancient].selection_experience = 'ANCIENT_EVIDENCE'
    before = deepcopy(tree)
    for op, reference in [('Refine', None), ('Crossover', ref)]:
        text, _, _, meta = build_evolution_context(tree, current, op, reference)
        assert 'ENTRY_EVIDENCE' in text and 'ORIGIN_EVIDENCE' in text
        assert 'ANCIENT_EVIDENCE' not in text
        assert 'ancient_code' not in text and 'origin_code' not in text
        assert 'entry_code' in text and 'current_code' in text
        if op == 'Refine':
            assert meta['path'] == [entry, current]
        else:
            assert 'reference_code' in text and not meta['same_route']
        muted, _, _, _ = build_evolution_context(
            tree, current, op, reference, ContextConfig(include_experiences=False))
        assert 'ENTRY_EVIDENCE' not in muted and 'ORIGIN_EVIDENCE' not in muted
    assert tree == before


def test_negative_scores_are_compared_by_program_and_persisted_on_the_edge():
    import json
    tree = TreePopulation()
    start = add(tree, 0, -14.848125, 'def solve(x): return x')
    middle = add(tree, start, -14.696875, 'def solve(x): return x + 1', 'Explore')
    end = add(tree, middle, -11.508125, 'def solve(x): return x * 2', 'Explore')
    region = PathContraction(start, end, (middle,), 'Explore', None)
    prompt = build_variation_prompt(tree, region, task_description='Minimize negative reward')
    assert 'Overall measured outcome: regressed' in prompt
    assert 'Step 1 [Explore] (Removed Intermediate): regressed' in prompt
    assert 'Step 2 [Explore] (Retained Endpoint): regressed' in prompt
    assert '-14.8481' not in prompt and '-14.6969' not in prompt
    class Model:
        def complete(self, *args, **kwargs):
            return json.dumps({'experience': 'The computation changed from identity to multiplication.'})
    new, records = reflect_and_apply(
        tree, ConsolidationPlan((start, end), (region,)), model=Model())
    assert not records[0].used_fallback
    assert '[-14.8481 -> -11.5081 (regressed)]' in new.edges[end].variation_experience
