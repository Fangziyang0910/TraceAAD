from __future__ import annotations

import os

from pathlib import Path
from baselines.profiler import ProfilerBase


class MAProfiler(ProfilerBase):

    def __init__(self, run_dir=None, **kwargs):
        super().__init__(run_dir, **kwargs)
        self._cur_gen = 0
        if self._log_dir:
            self._mcts_state_path = os.path.join(self._log_dir, 'mcts_state.jsonl')
            self._mcts_events_path = os.path.join(self._log_dir, 'mcts_events.jsonl')
            self._llm_calls_path = str(Path(self._log_dir).parent / 'calls.jsonl')


    def log_mcts_state(self, *, phase: str, sample_order: int, max_sample_nums, mcts, selected_node=None):
        if not self._log_dir:
            return
        root_children = [self._node_summary(node) for node in mcts.root.children]
        payload = {
            'phase': phase,
            'sample_order': sample_order,
            'max_sample_nums': max_sample_nums,
            'q_min': mcts.q_min,
            'q_max': mcts.q_max,
            'rank_list': list(mcts.rank_list),
            'root_visits': mcts.root.visits,
            'root_q': mcts.root.Q,
            'root_children': root_children,
        }
        if selected_node is not None:
            payload['selected_node'] = self._node_summary(selected_node)
        self._append_jsonl(self._mcts_state_path, payload)
        self.log_method_state(method='mcts_ahd', **payload)

        best = max(mcts.rank_list) if mcts.rank_list else None
        subtree_sizes = [child['subtree_size'] for child in root_children]
        self.log_message(
            f"MCTS state {phase}: samples={sample_order}/{max_sample_nums}, "
            f"rank_count={len(mcts.rank_list)}, best={best}, root_subtree_sizes={subtree_sizes}"
        )

    def log_mcts_event(self, **payload):
        if not self._log_dir:
            return
        self._append_jsonl(self._mcts_events_path, payload)
        self.log_method_event(method='mcts_ahd', **payload)

        event = payload.get('event', 'event')
        status = payload.get('status')
        operator = payload.get('operator')
        sample_order = payload.get('sample_order')
        parent_score = payload.get('parent_score')
        child_score = payload.get('child_score')
        self.log_message(
            f"MCTS event {event}: status={status}, op={operator}, "
            f"samples={sample_order}, parent_score={parent_score}, child_score={child_score}"
        )

    @staticmethod
    def _node_summary(node):
        individual = getattr(node, 'individual', None)
        raw_info = getattr(node, 'raw_info', None)
        score = None
        if raw_info is not None:
            score = getattr(raw_info, 'score', None)
        if score is None and individual is not None:
            score = getattr(individual, 'score', None)
        return {
            'score': score,
            'q': getattr(node, 'Q', None),
            'depth': getattr(node, 'depth', None),
            'visits': getattr(node, 'visits', None),
            'children_count': len(getattr(node, 'children', [])),
            'subtree_size': len(getattr(node, 'subtree', [])),
            'is_root_child': (
                getattr(getattr(node, 'parent', None), 'is_root', False)
                or getattr(getattr(node, 'parent', None), 'code', None) == 'Root'
            ),
        }
