"""
Monte Carlo Tree Search (MCTS) with Bayesian Belief Steering
Integrated with HGM-inspired Clade Metaproductivity concepts

Enhancements (Inspired by HGM):
1. Clade-based Thompson Sampling: Select nodes based on subtree performance
2. Pseudo evaluations: Stabilize sampling for sparse nodes  
3. Adaptive temperature (cool_down): Shift from exploration to exploitation

Key Differences from HGM:
    - HGM uses binary success/failure (coding tasks)
    - We use normalized continuous performance (optimization)
    - HGM separates expansion/evaluation (async)
    - We keep synchronous MCTS structure

Author: MCTS-AHD Framework  
Date: 2025-10-30
"""

import random
import copy
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque

from .bayesian_node import BayesianNode
from .score_normalizer import ScoreNormalizer


class MCTS:
    """
    MCTS with HGM-inspired Clade Metaproductivity.
    
    Core Features:
    - Thompson Sampling based on clade (subtree) performance
    - Pseudo evaluations for stability
    - Adaptive temperature (cool_down)
    - Original MCTS operator execution (removed operator selection logic)
    
    Attributes:
        root: Root node
        normalizer: Score normalization system
        num_pseudo: Number of pseudo evaluations (HGM-inspired)
        cool_down_enabled: Enable temperature annealing (HGM-inspired)
        cool_down_beta: Temperature decay exponent (HGM-inspired)
    """
    
    def __init__(
        self,
        root_code: str,
        optimization_type: str = 'min',
        max_depth: int = 10,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        num_pseudo: int = 10,
        cool_down_enabled: bool = False,
        cool_down_beta: float = 1.0,
        pruning_enabled: bool = False,
        pruning_threshold: float = 0.3,
        pruning_min_visits: int = 10,
        backprop_lambda: float = 1.0
    ):
        """
        Initialize MCTS.
        
        Args:
            root_code: Initial code or "Root"
            optimization_type: 'min' or 'max'
            max_depth: Maximum tree depth
            prior_alpha: Prior α for Beta distribution
            prior_beta: Prior β for Beta distribution
            num_pseudo: Pseudo evaluation count (default: 10, like HGM)
            cool_down_enabled: Enable adaptive temperature
            cool_down_beta: Temperature decay rate (default: 1.0, like HGM)
            pruning_enabled: Enable dynamic pruning/freezing (HGM-inspired)
            pruning_threshold: Performance threshold for pruning (default: 0.3)
            pruning_min_visits: Minimum visits before considering pruning (default: 10)
            backprop_lambda: Credit assignment decay (default: 1.0 = no decay, 0.9 = 10% decay per level)
        """
        self.root = BayesianNode(
            parent=None,
            code=root_code,
            algorithm="Root node",
            depth=0,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta
        )
        
        self.normalizer = ScoreNormalizer(
            optimization_type=optimization_type,
            epsilon=1e-10
        )
        
        self.max_depth = max_depth
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.optimization_type = optimization_type
        
        # HGM-inspired parameters
        self.num_pseudo = num_pseudo
        self.cool_down_enabled = cool_down_enabled
        self.cool_down_beta = cool_down_beta
        
        # Dynamic Pruning/Freezing (HGM-inspired)
        self.pruning_enabled = pruning_enabled
        self.pruning_threshold = pruning_threshold
        self.pruning_min_visits = pruning_min_visits
        
        # Credit Assignment with Lambda Decay
        self.backprop_lambda = backprop_lambda
        
        # MCTS expansion control (compatibility)
        self.alpha = 0.5
        
        # Tracking
        self.selection_history: List[BayesianNode] = []
        self.expansion_history: List[BayesianNode] = []
        self.best_node_history: List[BayesianNode] = []
        self.rank_list: List[float] = []
    
    def select_clade_thompson(
        self, 
        candidates: List[BayesianNode],
        progress_ratio: float = 0.0
    ) -> int:
        """
        **CORE: Clade-based Thompson Sampling (HGM-inspired)**
        
        Select candidate index using Thompson Sampling on clade performance.
        
        Inspired by HGM's TS_sample():
        1. Filter out frozen nodes (pruning)
        2. Aggregate clade scores (with pseudo evaluations)
        3. Compute Beta(α, β) from aggregated scores
        4. Apply temperature if cool_down enabled
        5. Sample and select argmax
        
        Args:
            candidates: List of candidate nodes
            progress_ratio: Search progress in [0,1] for temperature
            
        Returns:
            Index of selected candidate
        """
        if not candidates:
            return 0
        
        # Filter out frozen nodes (Dynamic Pruning)
        active_candidates = []
        active_indices = []
        for idx, node in enumerate(candidates):
            if not node.is_frozen:
                active_candidates.append(node)
                active_indices.append(idx)
        
        # If all candidates are frozen (shouldn't happen), select first
        if not active_candidates:
            print("  [WARNING] All candidates frozen, selecting first candidate")
            return 0
        
        # Get clade scores for each active candidate (HGM's get_decendant_evals equivalent)
        clade_scores_list = [node.get_clade_scores(self.num_pseudo) for node in active_candidates]
        
        # Compute Beta parameters (HGM's formula)
        alphas = [1.0 + np.sum(scores) for scores in clade_scores_list]
        betas = [1.0 + len(scores) - np.sum(scores) for scores in clade_scores_list]
        
        # Apply temperature (cool down) - HGM's formula
        if self.cool_down_enabled and progress_ratio > 0.0:
            remaining = 1.0 - progress_ratio
            if remaining < 0.0001:
                temperature = 10000.0  # Near end, strong exploitation
            else:
                # HGM formula: (1 / remaining)^beta
                temperature = (1.0 / remaining) ** self.cool_down_beta
            
            alphas = [a * temperature for a in alphas]
            betas = [b * temperature for b in betas]
        
        # Thompson Sampling: sample from Beta and select max
        samples = [np.random.beta(max(a, 1e-10), max(b, 1e-10)) for a, b in zip(alphas, betas)]
        selected_active_idx = int(np.argmax(samples))
        
        # Map back to original candidate index
        selected_idx = active_indices[selected_active_idx]
        
        return selected_idx
    
    def select(self, root_node: Optional[BayesianNode] = None) -> BayesianNode:
        """
        Selection phase: Traverse to a leaf using Thompson Sampling.
        
        At each non-leaf node:
        1. Sample from each child's individual Beta distribution
        2. Select child with highest sample
        3. Repeat until reaching a leaf
        
        Args:
            root_node: Starting node (default: self.root)
            
        Returns:
            Selected leaf node
        """
        if root_node is None:
            root_node = self.root
        
        current_node = root_node
        path = [current_node]
        
        # Traverse until leaf
        while not current_node.is_leaf():
            best_child = None
            max_sample = -float('inf')
            
            for child in current_node.children:
                # Sample from child's belief distribution
                sample = child.sample_from_belief()
                
                if sample > max_sample:
                    max_sample = sample
                    best_child = child
            
            if best_child is None:
                break
            
            current_node = best_child
            path.append(current_node)
        
        self.selection_history.append(current_node)
        return current_node
    
    def should_expand(self, node: BayesianNode) -> bool:
        """
        Check if node should be expanded.
        
        Args:
            node: Node to check
            
        Returns:
            True if expandable
        """
        if node.code == "Root" and len(node.children) > 0:
            return False
        
        if node.depth >= self.max_depth:
            return False
        
        return node.is_leaf()
    
    def expand(
        self,
        leaf_node: BayesianNode,
        new_code: str,
        new_algorithm: str = "",
        operator: Optional[str] = None
    ) -> BayesianNode:
        """
        Expansion phase: Create a new child node.
        
        Args:
            leaf_node: Parent node
            new_code: Generated heuristic code
            new_algorithm: Algorithm description
            operator: Operator used
            
        Returns:
            New child node
        """
        new_child = BayesianNode(
            parent=leaf_node,
            code=new_code,
            algorithm=new_algorithm,
            depth=leaf_node.depth + 1,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
            operator=operator
        )
        
        leaf_node.add_child(new_child)
        self.expansion_history.append(new_child)
        
        return new_child
    
    def simulate(
        self,
        node: BayesianNode,
        raw_objective: float
    ) -> float:
        """
        Simulation phase: Evaluate and normalize performance.
        
        Args:
            node: Node to evaluate
            raw_objective: Raw objective value
            
        Returns:
            Normalized score in [0, 1]
        """
        node.objective = raw_objective
        self.normalizer.update(raw_objective)
        normalized_score = self.normalizer.normalize(raw_objective)
        
        return normalized_score
    
    def backpropagate(self, node: BayesianNode, normalized_score: float):
        """
        Backpropagation: Update beliefs along path to root with lambda decay.
        
        Implements credit assignment with exponential decay:
        - Child (self): λ^0 = 1.0 (full credit)
        - Parent: λ^1 (e.g., 0.9)
        - Grandparent: λ^2 (e.g., 0.81)
        ...
        
        Args:
            node: Leaf node evaluated
            normalized_score: Performance in [0, 1]
        """
        success_evidence = normalized_score
        failure_evidence = 1.0 - normalized_score
        
        current_node = node
        decay_factor = 1.0  # Start with λ^0 = 1.0
        
        while current_node is not None:
            # Update belief with decayed evidence
            current_node.alpha += success_evidence * decay_factor
            current_node.beta += failure_evidence * decay_factor
            current_node.visits += 1
            
            # Track subtree (for root's children)
            if (current_node.code != 'Root' and 
                current_node.parent is not None and 
                current_node.parent.code == 'Root'):
                current_node.subtree.append(node)
            
            current_node = current_node.parent
            decay_factor *= self.backprop_lambda  # Apply lambda decay
        
        # Update rank list
        expected_perf = node.expected_performance
        if expected_perf not in self.rank_list:
            self.rank_list.append(expected_perf)
            self.rank_list.sort(reverse=True)  # Higher is better in normalized space
    
    def prune_tree(self, global_best_performance: float):
        """
        **Dynamic Pruning/Freezing (HGM-inspired)**
        
        Identify and freeze underperforming clades to focus computational
        resources on promising regions of the search space.
        
        Pruning Criteria (all must be satisfied):
        1. Node has been visited enough times (>= pruning_min_visits)
        2. Node's clade performance is significantly worse than global best
        3. Node is not already frozen
        
        CRITICAL: Normalized scores are ALREADY adjusted for optimization type
        by ScoreNormalizer (inverts for 'min'), so higher is ALWAYS better.
        Pruning logic is the same for both min/max problems.
        
        Args:
            global_best_performance: Current best normalized performance [0, 1]
        """
        if not self.pruning_enabled:
            return
        
        frozen_count = 0
        
        def check_and_freeze(node: BayesianNode):
            nonlocal frozen_count
            
            # Skip if already frozen or is root
            if node.is_frozen or node.code == "Root":
                return
            
            # Skip if not visited enough
            if node.visits < self.pruning_min_visits:
                return
            
            # Get clade performance (aggregated subtree performance)
            clade_scores = node.get_clade_scores(num_pseudo=0)  # No pseudo for pruning decision
            if len(clade_scores) == 0:
                return
            
            clade_mean_performance = np.mean(clade_scores)
            
            # Pruning decision: Is this clade significantly underperforming?
            # NOTE: ScoreNormalizer inverts scores for 'min' problems, 
            # so higher normalized score is ALWAYS better, regardless of optimization type.
            # If clade_mean < threshold * global_best, freeze it
            if clade_mean_performance < self.pruning_threshold * global_best_performance:
                node.is_frozen = True
                node.frozen_reason = (
                    f"Clade mean {clade_mean_performance:.4f} < "
                    f"{self.pruning_threshold} * {global_best_performance:.4f} "
                    f"(threshold * global_best)"
                )
                frozen_count += 1
                
                # Recursively freeze all descendants (they inherit the freeze)
                def freeze_subtree(n: BayesianNode):
                    nonlocal frozen_count  # Upstream otherwise raises when freezing a descendant.
                    for child in n.children:
                        if not child.is_frozen:
                            child.is_frozen = True
                            child.frozen_reason = f"Parent frozen: {node.algorithm}"
                            frozen_count += 1
                        freeze_subtree(child)
                
                freeze_subtree(node)
            else:
                # If not frozen, check children
                for child in node.children:
                    check_and_freeze(child)
        
        # Start pruning from root's children
        for child in self.root.children:
            check_and_freeze(child)
        
        if frozen_count > 0:
            print(f"  [Pruning] Froze {frozen_count} nodes based on clade performance")
    
    def get_best_node(self) -> BayesianNode:
        """
        Get node with highest expected performance.
        
        Returns:
            Best node
        """
        best_node = self.root
        best_performance = -float('inf')
        
        def traverse(node: BayesianNode):
            nonlocal best_node, best_performance
            
            if node.visits > 0 and node.expected_performance > best_performance:
                best_performance = node.expected_performance
                best_node = node
            
            for child in node.children:
                traverse(child)
        
        traverse(self.root)
        return best_node
    
    def get_best_heuristic(self) -> Tuple[str, str, float]:
        """
        Get best heuristic (code, algorithm, objective).
        
        Returns:
            Tuple of (code, algorithm, objective)
        """
        best_node = self.get_best_node()
        return (
            best_node.code,
            best_node.algorithm,
            best_node.objective if best_node.objective is not None else float('inf')
        )
    
    def get_tree_statistics(self) -> Dict:
        """
        Get tree statistics.
        
        Returns:
            Dictionary of statistics
        """
        all_nodes = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            all_nodes.append(node)
            queue.extend(node.children)
        
        visited_nodes = [n for n in all_nodes if n.visits > 0]
        
        if not visited_nodes:
            return {
                'total_nodes': len(all_nodes),
                'visited_nodes': 0,
                'max_depth': 0,
                'avg_expected_performance': 0,
                'best_expected_performance': 0,
                'avg_uncertainty': 0,
                'normalizer_stats': self.normalizer.get_statistics()
            }
        
        return {
            'total_nodes': len(all_nodes),
            'visited_nodes': len(visited_nodes),
            'max_depth': max(n.depth for n in all_nodes),
            'avg_expected_performance': np.mean([n.expected_performance for n in visited_nodes]),
            'best_expected_performance': max(n.expected_performance for n in visited_nodes),
            'avg_uncertainty': np.mean([n.uncertainty for n in visited_nodes]),
            'total_visits': sum(n.visits for n in visited_nodes),
            'normalizer_stats': self.normalizer.get_statistics()
        }
    
    def get_all_nodes(self) -> List[BayesianNode]:
        """
        Get all nodes (BFS order).
        
        Returns:
            List of all nodes
        """
        nodes = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            nodes.append(node)
            queue.extend(node.children)
        
        return nodes
    
    def print_tree_summary(self):
        """Print tree summary."""
        stats = self.get_tree_statistics()
        print("\n" + "="*60)
        print("MCTS Tree Summary (HGM-Inspired)")
        print("="*60)
        print(f"Total Nodes:               {stats['total_nodes']}")
        print(f"Visited Nodes:             {stats['visited_nodes']}")
        print(f"Maximum Depth:             {stats['max_depth']}")
        print(f"Best Expected Performance: {stats['best_expected_performance']:.4f}")
        print(f"Avg Expected Performance:  {stats['avg_expected_performance']:.4f}")
        print(f"Avg Uncertainty:           {stats['avg_uncertainty']:.6f}")
        print(f"Total Visits:              {stats['total_visits']}")
        
        print(f"\nHGM-inspired Settings:")
        print(f"  Num Pseudo Evals:        {self.num_pseudo}")
        print(f"  Cool Down Enabled:       {self.cool_down_enabled}")
        if self.cool_down_enabled:
            print(f"  Cool Down Beta:          {self.cool_down_beta}")
        
        norm_stats = stats['normalizer_stats']
        print("\nNormalizer Statistics:")
        print(f"  Optimization Type:       {norm_stats['optimization_type']}")
        if norm_stats['num_scores'] > 0:
            print(f"  Score Range:             [{norm_stats['min_score']:.4f}, {norm_stats['max_score']:.4f}]")
            print(f"  Num Evaluations:         {norm_stats['num_scores']}")
        print("="*60 + "\n")
