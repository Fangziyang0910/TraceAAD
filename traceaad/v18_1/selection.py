"""Same-age ranks supplement the unchanged basic parent distribution."""

import math

from ._basic_selection import sample_parent as basic_sample_parent, weights as basic_weights


def maturity(program):
    return program.request.get("maturity")


def age_distribution(nodes, programs, births, config):
    histories = {}
    for program in sorted(programs.values(), key=lambda item: item.id):
        record = maturity(program)
        if program.valid and record:
            # Freeze the first arrival at each age; later retries must not rewrite history.
            histories.setdefault((record["birth_id"], record["age"]), program.fitness)

    representatives = {}
    for node in nodes:
        record = maturity(node)
        if not record:
            continue
        birth = births[str(record["birth_id"])]
        if (
            not birth["confirmed"]
            or birth["protected_used"] >= config.protection_attempts
            or birth["max_age"] >= config.max_refinement_age
            or record["age"] != birth["max_age"]
        ):
            continue
        previous = representatives.get(record["birth_id"])
        if previous is None or (node.fitness, node.id) < (previous.fitness, previous.id):
            representatives[record["birth_id"]] = node

    ranks, peers = {}, {}
    for birth_id, node in representatives.items():
        age = maturity(node)["age"]
        comparison = [
            fitness for (other_birth, other_age), fitness in histories.items()
            if other_birth != birth_id and other_age == age
        ]
        ties = sum(math.isclose(node.fitness, value, rel_tol=1e-9, abs_tol=1e-9)
                   for value in comparison)
        strict_wins = sum(
            node.fitness < value
            and not math.isclose(node.fitness, value, rel_tol=1e-9, abs_tol=1e-9)
            for value in comparison
        )
        ranks[node.id] = (1 + strict_wins + 0.5 * ties) / (2 + len(comparison))
        peers[node.id] = len(comparison)
    total = math.fsum(ranks.values())
    probabilities = {pid: rank / total for pid, rank in ranks.items()} if total else {}
    return probabilities, ranks, peers


def sample_parent(nodes, programs, births, config, rng):
    nodes = sorted(nodes, key=lambda node: node.id)
    if not config.same_age or config.age_fraction == 0:
        node, record = basic_sample_parent(nodes, programs, rng)
        return node, {**record, "channel": "ordinary", "age_probability": 0.0}
    age_prob, ranks, peers = age_distribution(nodes, programs, births, config)
    if not age_prob:
        node, record = basic_sample_parent(nodes, programs, rng)
        return node, {**record, "channel": "ordinary", "age_probability": 0.0}
    base_prob, stats = basic_weights(nodes, programs)
    fraction = config.age_fraction
    mixed = [
        (1 - fraction) * ordinary + fraction * age_prob.get(node.id, 0.0)
        for node, ordinary in zip(nodes, base_prob)
    ]
    protected = rng.random() < fraction
    channel = [age_prob.get(node.id, 0.0) for node in nodes] if protected else base_prob
    index = rng.choices(range(len(nodes)), weights=channel, k=1)[0]
    node = nodes[index]
    return node, {
        "channel": "same_age" if protected else "ordinary",
        "probability": mixed[index],
        "ordinary_probability": base_prob[index],
        "age_probability": age_prob.get(node.id, 0.0),
        "same_age_rank": ranks.get(node.id),
        "same_age_peers": peers.get(node.id, 0),
        "beta": stats["beta"],
        "levels_ess": stats["levels_ess"],
        "improvement_rate": stats["rate"],
        "tried": stats["tried"][node.id],
        "improved": stats["improved"][node.id],
        "prospect": stats["prospect"][node.id],
        "ess": 1 / math.fsum(p * p for p in mixed),
        "eligible": len(nodes),
        "protected_candidates": len(age_prob),
    }
