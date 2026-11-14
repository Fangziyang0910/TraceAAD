from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import build_variation_prompt, reflect_and_apply
from evotrace_v3.topology import ConsolidationPlan, PathContraction


def add(tree, parent, score, code, op='Refine'):
    nid = tree.latest_node_id + 1
    tree.add(Node(nid, code, f'Design {nid}', score),
             Edge(parent, nid, op if parent else None), evaluations=1)
    return nid


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
