"""
Bayesian Node Implementation for MCTS

This module implements the core node structure for Bayesian Belief Steering MCTS.
Unlike traditional MCTS nodes that store Q-values and visit counts, BayesianNode
maintains a Beta distribution representing the belief about the node's true potential.

Key Concepts:
    - Beta Distribution: Beta(α, β) models performance in [0, 1]
    - Prior: Beta(1, 1) = Uniform distribution (uninformed prior)
    - Update: Bayesian update via conjugate prior properties
    - Thompson Sampling: Sample from belief for selection decisions

Mathematical Foundation:
    - Expected Performance: E[θ] = α / (α + β)
    - Uncertainty: Var[θ] = (α·β) / ((α+β)²·(α+β+1))
    - Higher α → Better performance
    - Higher β → Worse performance
    - Higher (α+β) → More evidence, less uncertainty

"""

import numpy as np
from typing import Optional, List


class BayesianNode:
    """
    Core node structure for MCTS.
    
    Each node maintains a Bayesian belief distribution (Beta distribution)
    about its true performance potential, updated through evidence from
    heuristic evaluations.
    
    Attributes:
        parent: Parent node in the MCTS tree
        code: Heuristic code (Python function as string)
        algorithm: Algorithm description/thought
        children: List of child nodes
        alpha: Beta distribution α parameter (success evidence)
        beta: Beta distribution β parameter (failure evidence)
        visits: Number of times this node has been visited
        depth: Depth in the tree (0 = root)
        raw_info: Additional metadata
        operator: Operator used to generate this node (e.g., 'e1', 'm1')
        objective: Raw objective value (before normalization)
    """
    
    def __init__(
        self,
        parent: Optional['BayesianNode'],
        code: str,
        algorithm: str = "",
        depth: int = 0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        operator: Optional[str] = None,
        raw_info: Optional[dict] = None
    ):
        """
        Initialize a Bayesian MCTS node.
        
        Args:
            parent: Parent node (None for root)
            code: Heuristic code as string
            algorithm: Description of the algorithm/design idea
            depth: Depth in the tree
            prior_alpha: Prior α parameter for Beta distribution (default: 1.0)
            prior_beta: Prior β parameter for Beta distribution (default: 1.0)
            operator: Evolutionary operator used to create this node
            raw_info: Additional information dictionary
        """
        self.parent = parent
        self.code = code
        self.algorithm = algorithm
        self.depth = depth
        self.children: List[BayesianNode] = []
        
        # --- Core Bayesian Belief Parameters ---
        # Beta(1, 1) is a uniform prior: before any observation,
        # we believe any performance value in [0, 1] is equally likely
        self.alpha = prior_alpha
        self.beta = prior_beta
        # --------------------------------------
        
        self.visits = 0
        self.operator = operator  # Track which operator created this node
        self.raw_info = raw_info if raw_info is not None else {}
        
        # Store raw objective value (for tracking and debugging)
        self.objective: Optional[float] = None
        
        # Track subtree for root's children (similar to original MCTS)
        self.subtree: List[BayesianNode] = []
        
        # --- Pruning/Freezing mechanism (HGM-inspired) ---
        self.is_frozen = False  # If True, this node and its subtree are pruned
        self.frozen_reason = ""  # Explanation for why this node was frozen
        # --------------------------------------
        
        # --- Compatibility with original MCTS interface ---
        # These attributes are needed for compatibility with existing code
        self.children_info = []  # List of offspring info dicts
        self.Q = 0.0  # Q-value (can be synced with expected_performance)
        self.reward = 0.0  # Reward value
    
    @property
    def expected_performance(self) -> float:
        """
        Calculate the expected performance (mean of Beta distribution).
        
        This is the Bayesian point estimate of the node's true potential.
        
        Returns:
            Expected performance in [0, 1], where 1 is best
        """
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def num_evaluations(self) -> int:
        """
        Get number of evaluations (evidence count).
        
        Returns:
            Number of times this node has been evaluated
        """
        return self.visits
    
    @property
    def uncertainty(self) -> float:
        """
        Calculate belief uncertainty (variance of Beta distribution).
        
        Higher uncertainty indicates we need more evidence to be confident
        about the node's true potential.
        
        Returns:
            Variance of the Beta distribution
        """
        total_evidence = self.alpha + self.beta
        return (self.alpha * self.beta) / (
            total_evidence ** 2 * (total_evidence + 1)
        )
    
    @property
    def total_evidence(self) -> float:
        """
        Total evidence accumulated (α + β).
        
        This increases with each evaluation, reflecting our confidence level.
        
        Returns:
            Sum of α and β
        """
        return self.alpha + self.beta
    
    def sample_from_belief(self) -> float:
        """
        Perform Thompson Sampling: sample from the node's belief distribution.
        
        This is the core of the selection strategy. By sampling from Beta(α, β),
        we naturally balance exploration (high uncertainty → wider distribution)
        and exploitation (high expected value → higher samples).
        
        Returns:
            A sample from Beta(α, β) in [0, 1]
        """
        return np.random.beta(self.alpha, self.beta)
    
    def update_belief(self, normalized_performance: float):
        """
        Update belief distribution using Bayesian inference.
        
        This leverages the conjugate prior property of Beta-Bernoulli:
        - Normalized performance acts as "success" evidence
        - (1 - normalized_performance) acts as "failure" evidence
        
        Args:
            normalized_performance: Performance score in [0, 1]
                                   (1 = best, 0 = worst)
        """
        # Bayesian update via conjugate prior
        success_evidence = normalized_performance
        failure_evidence = 1.0 - normalized_performance
        
        self.alpha += success_evidence
        self.beta += failure_evidence
        self.visits += 1
    
    def get_clade_scores(self, num_pseudo: int = 10) -> List[float]:
        """
        Get aggregated performance scores from entire subtree (clade).
        
        **Inspired by HGM's get_decendant_evals():**
        This method aggregates the "success" evidence from the entire subtree,
        treating each node's Beta distribution as a distribution over performance.
        
        Why this matters:
        - A node with poor individual performance might lead to great descendants
        - Aggregating clade evidence helps identify "stepping stone" nodes
        - Pseudo evaluations stabilize sampling for sparsely-visited nodes
        
        Args:
            num_pseudo: Number of pseudo evaluations (using node's own mean)
            
        Returns:
            List of normalized scores [0,1] from clade
        """
        # Start with pseudo evaluations (HGM technique for sparse nodes)
        # Use expected_performance as representative value
        clade_scores = [self.expected_performance] * num_pseudo
        
        # Aggregate all descendants' evidence
        # We extract "scores" from each node's Beta distribution
        for descendant in self.get_subtree()[1:]:  # Skip self
            # Each descendant contributes its expected performance
            # weighted by its evidence count (visits)
            if descendant.visits > 0:
                clade_scores.extend([descendant.expected_performance] * descendant.visits)
        
        return clade_scores
    
    def get_clade_alpha_beta(self, num_pseudo: int = 10) -> tuple:
        """
        Get aggregated Beta parameters from entire clade.
        
        Alternative to get_clade_scores: directly aggregate α and β.
        This is more efficient and theoretically sound.
        
        Args:
            num_pseudo: Pseudo evaluation weight
            
        Returns:
            Tuple of (aggregated_alpha, aggregated_beta)
        """
        # Start with pseudo evidence based on self
        total_alpha = self.alpha * num_pseudo / (self.alpha + self.beta)
        total_beta = self.beta * num_pseudo / (self.alpha + self.beta)
        
        # Aggregate from all descendants
        for descendant in self.get_subtree()[1:]:  # Skip self
            total_alpha += descendant.alpha - 1.0  # Remove prior
            total_beta += descendant.beta - 1.0
        
        # Add back prior
        return (1.0 + total_alpha, 1.0 + total_beta)
    
    def get_subtree(self) -> List['BayesianNode']:
        """
        Get all nodes in subtree (self + all descendants).
        
        Uses iterative DFS to avoid recursion depth limits.
        
        Returns:
            List of nodes in depth-first order
        """
        result = []
        stack = [self]
        
        while stack:
            node = stack.pop()
            result.append(node)
            # Add children in reverse order so they're processed in correct order
            for child in reversed(node.children):
                stack.append(child)
        
        return result
    
    def add_child(self, child_node: 'BayesianNode'):
        """
        Add a child node to this node.
        
        Args:
            child_node: The child node to add
        """
        self.children.append(child_node)
    
    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf (has no children).
        
        Returns:
            True if leaf node, False otherwise
        """
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """
        Check if this node is the root.
        
        Returns:
            True if root node, False otherwise
        """
        return self.parent is None
    
    def get_path_from_root(self) -> List['BayesianNode']:
        """
        Get the path from root to this node.
        
        Returns:
            List of nodes from root to this node (inclusive)
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_best_child(self) -> Optional['BayesianNode']:
        """
        Get the child with highest expected performance.
        
        Returns:
            Best child node, or None if no children
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.expected_performance)
    
    def get_ucb_score(self, exploration_constant: float = 1.0) -> float:
        """
        Calculate Upper Confidence Bound score (for comparison).
        
        This is included for compatibility and comparison purposes.
        In MCTS, we use Thompson Sampling instead of UCB.
        
        Args:
            exploration_constant: Exploration parameter
            
        Returns:
            UCB score
        """
        if self.parent is None or self.visits == 0:
            return float('inf')
        
        exploitation = self.expected_performance
        exploration = exploration_constant * np.sqrt(
            np.log(self.parent.visits + 1) / self.visits
        )
        return exploitation + exploration
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about this node.
        
        Returns:
            Dictionary containing all relevant statistics
        """
        return {
            'depth': self.depth,
            'visits': self.visits,
            'alpha': self.alpha,
            'beta': self.beta,
            'expected_performance': self.expected_performance,
            'uncertainty': self.uncertainty,
            'total_evidence': self.total_evidence,
            'num_children': len(self.children),
            'operator': self.operator,
            'objective': self.objective,
            'algorithm': self.algorithm[:100] if self.algorithm else ""  # Truncate for display
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"BayesianNode(E[θ]={self.expected_performance:.4f}, "
                f"Var[θ]={self.uncertainty:.6f}, visits={self.visits}, "
                f"depth={self.depth}, children={len(self.children)})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


class BayesianMCTSTree:
    """
    Container class for the Bayesian MCTS tree.
    
    Provides utility methods for tree-wide operations like finding
    the best node, gathering statistics, etc.
    """
    
    def __init__(self, root_code: str = "Root"):
        """
        Initialize the Bayesian MCTS tree.
        
        Args:
            root_code: Code for the root node (usually "Root")
        """
        self.root = BayesianNode(
            parent=None,
            code=root_code,
            algorithm="Root node",
            depth=0,
            prior_alpha=1.0,
            prior_beta=1.0
        )
    
    def get_best_node_by_expectation(self) -> BayesianNode:
        """
        Find the node with highest expected performance in the entire tree.
        
        Returns:
            Node with maximum expected performance
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
    
    def get_all_nodes(self) -> List[BayesianNode]:
        """
        Get all nodes in the tree.
        
        Returns:
            List of all nodes (BFS order)
        """
        nodes = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)
        
        return nodes
    
    def get_tree_statistics(self) -> dict:
        """
        Gather comprehensive tree statistics.
        
        Returns:
            Dictionary with tree-wide statistics
        """
        all_nodes = self.get_all_nodes()
        
        visited_nodes = [n for n in all_nodes if n.visits > 0]
        
        if not visited_nodes:
            return {
                'total_nodes': len(all_nodes),
                'visited_nodes': 0,
                'max_depth': 0,
                'avg_expected_performance': 0,
                'best_expected_performance': 0,
                'avg_uncertainty': 0
            }
        
        return {
            'total_nodes': len(all_nodes),
            'visited_nodes': len(visited_nodes),
            'max_depth': max(n.depth for n in all_nodes),
            'avg_expected_performance': np.mean([n.expected_performance for n in visited_nodes]),
            'best_expected_performance': max(n.expected_performance for n in visited_nodes),
            'avg_uncertainty': np.mean([n.uncertainty for n in visited_nodes]),
            'total_visits': sum(n.visits for n in visited_nodes)
        }
    
    def get_leaf_nodes(self) -> List[BayesianNode]:
        """
        Get all leaf nodes in the tree.
        
        Returns:
            List of leaf nodes
        """
        return [n for n in self.get_all_nodes() if n.is_leaf()]
