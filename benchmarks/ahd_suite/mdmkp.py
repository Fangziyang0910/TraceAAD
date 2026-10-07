"""Feasible single-item and exchange search with an evolved move-ranking function."""

import numpy as np

from .validation import scores


def feasible(data, selected):
    return (np.all(data["A_leq"] @ selected <= data["b_leq"]) and
            np.all(data["A_geq"] @ selected >= data["b_geq"]))


def eligible_moves(data, selected):
    selected_ids, other = np.flatnonzero(selected), np.flatnonzero(1 - selected)
    singles = np.concatenate((np.column_stack((selected_ids, np.full(len(selected_ids), -1))),
                              np.column_stack((np.full(len(other), -1), other))))
    exchanges = np.column_stack((np.repeat(selected_ids, len(other)), np.tile(other, len(selected_ids))))
    moves = np.vstack((singles, exchanges)).astype(np.int64)
    valid = np.ones(len(moves), dtype=bool)
    for matrix, limit, upper in ((data["A_leq"], data["b_leq"], True),
                                 (data["A_geq"], data["b_geq"], False)):
        padded = np.column_stack((matrix, np.zeros(len(matrix), dtype=matrix.dtype)))
        usage = matrix @ selected
        trial = usage[:, None] + padded[:, moves[:, 1]] - padded[:, moves[:, 0]]
        valid &= np.all(trial <= limit[:, None], axis=0) if upper else np.all(trial >= limit[:, None], axis=0)
    return moves[valid]


def solve(data, heuristic, steps=32, tabu_tenure=5):
    selected = data["initial_solution"].astype(np.int8).copy()
    if not feasible(data, selected):
        raise ValueError("prepared MDMKP initial solution is infeasible")
    profits = data["cost_vector"]
    best, best_profit = selected.copy(), float(profits @ selected)
    tabu = np.zeros(len(selected), dtype=np.int64)
    for step in range(steps):
        moves = eligible_moves(data, selected)
        if not len(moves):
            break
        allowed = np.ones(len(moves), dtype=bool)
        for column in (0, 1):
            ids = moves[:, column]
            allowed &= (ids < 0) | (tabu[np.maximum(ids, 0)] <= step)
        if allowed.any():
            moves = moves[allowed]
        value = heuristic(profits.copy(), data["A_leq"].copy(), data["b_leq"].copy(),
                          data["A_geq"].copy(), data["b_geq"].copy(), selected.copy(), moves.copy())
        remove, add = moves[int(np.argmax(scores(value, len(moves), "score_moves")))]
        for item, state in ((remove, 0), (add, 1)):
            if item >= 0:
                selected[item] = state
                tabu[item] = step + tabu_tenure + 1
        profit = float(profits @ selected)
        if profit > best_profit:
            best, best_profit = selected.copy(), profit
    if not feasible(data, best) or np.any((best != 0) & (best != 1)):
        raise ValueError("outer MDMKP search returned an infeasible binary solution")
    return best, float(profits @ best)
