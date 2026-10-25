"""
Score Normalizer for BBS-MCTS

This module implements a dynamic score normalization system that maps raw
objective values to the [0, 1] interval required for Bayesian belief updates.

Key Challenges:
    1. Raw objective values can span arbitrary ranges (e.g., TSP distances)
    2. Min/max values are unknown beforehand and evolve during search
    3. Normalization must be consistent across the entire search process
    4. Need to handle both minimization and maximization problems

Solution:
    - Track global min/max values dynamically
    - Update normalization bounds as new extremes are discovered
    - Map to [0, 1] where 1 = best, 0 = worst
    - Support both 'min' and 'max' optimization directions

Author: BBS-MCTS Framework
Date: 2025-10-23
"""

import numpy as np
from typing import Optional, List


class ScoreNormalizer:
    """
    Dynamic score normalization for Bayesian belief updates.
    
    This class maintains running statistics of observed objective values
    and provides normalization to [0, 1] where 1 represents the best
    performance and 0 represents the worst.
    
    Attributes:
        min_score: Minimum observed objective value
        max_score: Maximum observed objective value
        optimization_type: 'min' or 'max' - defines what 'better' means
        epsilon: Small constant to avoid division by zero
        score_history: Historical record of all observed scores
    """
    
    def __init__(self, optimization_type: str = 'min', epsilon: float = 1e-10):
        """
        Initialize the score normalizer.
        
        Args:
            optimization_type: Either 'min' (lower is better) or 'max' (higher is better)
            epsilon: Small constant to prevent division by zero
        
        Raises:
            ValueError: If optimization_type is not 'min' or 'max'
        """
        if optimization_type not in ['min', 'max']:
            raise ValueError("optimization_type must be 'min' or 'max'")
        
        self.optimization_type = optimization_type
        self.epsilon = epsilon
        
        # Initialize with extreme values
        self.min_score = float('inf')
        self.max_score = float('-inf')
        
        # Track all observed scores for statistics
        self.score_history: List[float] = []
    
    def update(self, score: float):
        """
        Update min/max bounds with a new observed score.
        
        Args:
            score: New objective value to incorporate
        """
        if not np.isfinite(score):
            # Skip invalid scores (inf, -inf, nan)
            return
        
        self.score_history.append(score)
        
        if score < self.min_score:
            self.min_score = score
        if score > self.max_score:
            self.max_score = score
    
    def normalize(self, score: float) -> float:
        """
        Normalize a score to [0, 1] where 1 = best, 0 = worst.
        
        Normalization formula:
            - For minimization: normalized = (max - score) / (max - min)
            - For maximization: normalized = (score - min) / (max - min)
        
        Edge cases:
            - If max == min: return 0.5 (neutral performance)
            - If score is outside [min, max]: clip to [0, 1]
        
        Args:
            score: Raw objective value to normalize
            
        Returns:
            Normalized score in [0, 1]
        """
        if not np.isfinite(score):
            # Invalid score defaults to worst performance
            return 0.0
        
        # Handle case where we have no range yet
        if self.max_score == self.min_score:
            return 0.5  # Neutral: no basis for comparison
        
        # Normalize based on optimization type
        score_range = self.max_score - self.min_score + self.epsilon
        
        if self.optimization_type == 'min':
            # Lower is better: flip the scale
            normalized = (self.max_score - score) / score_range
        else:
            # Higher is better: standard normalization
            normalized = (score - self.min_score) / score_range
        
        # Clip to [0, 1] to handle scores outside observed range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def normalize_batch(self, scores: List[float]) -> List[float]:
        """
        Normalize a batch of scores.
        
        Args:
            scores: List of raw objective values
            
        Returns:
            List of normalized scores in [0, 1]
        """
        return [self.normalize(score) for score in scores]
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive normalization statistics.
        
        Returns:
            Dictionary containing normalization state and statistics
        """
        if not self.score_history:
            return {
                'min_score': None,
                'max_score': None,
                'range': 0,
                'num_scores': 0,
                'optimization_type': self.optimization_type,
                'mean_score': None,
                'std_score': None
            }
        
        return {
            'min_score': self.min_score,
            'max_score': self.max_score,
            'range': self.max_score - self.min_score,
            'num_scores': len(self.score_history),
            'optimization_type': self.optimization_type,
            'mean_score': np.mean(self.score_history),
            'std_score': np.std(self.score_history),
            'median_score': np.median(self.score_history)
        }
    
    def denormalize(self, normalized_score: float) -> float:
        """
        Convert a normalized score back to raw objective value.
        
        Useful for interpretability and reporting.
        
        Args:
            normalized_score: Score in [0, 1]
            
        Returns:
            Estimated raw objective value
        """
        if self.max_score == self.min_score:
            return self.min_score
        
        score_range = self.max_score - self.min_score
        
        if self.optimization_type == 'min':
            # Reverse the flip
            raw_score = self.max_score - (normalized_score * score_range)
        else:
            raw_score = self.min_score + (normalized_score * score_range)
        
        return raw_score
    
    def reset(self):
        """
        Reset the normalizer to initial state.
        
        Useful for starting a new search or experiment.
        """
        self.min_score = float('inf')
        self.max_score = float('-inf')
        self.score_history = []
    
    def is_initialized(self) -> bool:
        """
        Check if normalizer has observed any scores yet.
        
        Returns:
            True if at least one score has been observed
        """
        return len(self.score_history) > 0
    
    def get_percentile(self, score: float) -> float:
        """
        Calculate what percentile a score falls into.
        
        Args:
            score: Raw objective value
            
        Returns:
            Percentile rank (0-100)
        """
        if not self.score_history:
            return 50.0
        
        if self.optimization_type == 'min':
            # For minimization, lower scores are better (higher percentile)
            better_scores = sum(1 for s in self.score_history if s >= score)
        else:
            # For maximization, higher scores are better (higher percentile)
            better_scores = sum(1 for s in self.score_history if s <= score)
        
        percentile = (better_scores / len(self.score_history)) * 100
        return percentile
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"ScoreNormalizer(type={self.optimization_type}, "
                f"range=[{self.min_score:.4f}, {self.max_score:.4f}], "
                f"n_scores={len(self.score_history)})")


class AdaptiveScoreNormalizer(ScoreNormalizer):
    """
    Advanced normalizer with outlier detection and adaptive bounds.
    
    Extends the basic normalizer by:
    1. Detecting and handling outliers
    2. Using robust statistics (percentiles instead of min/max)
    3. Providing smoother normalization over time
    """
    
    def __init__(
        self,
        optimization_type: str = 'min',
        epsilon: float = 1e-10,
        percentile_bounds: tuple = (5, 95)
    ):
        """
        Initialize adaptive normalizer.
        
        Args:
            optimization_type: 'min' or 'max'
            epsilon: Small constant for numerical stability
            percentile_bounds: (lower, upper) percentiles for robust bounds
        """
        super().__init__(optimization_type, epsilon)
        self.percentile_bounds = percentile_bounds
    
    def get_robust_bounds(self) -> tuple:
        """
        Calculate robust min/max using percentiles instead of extremes.
        
        This helps avoid normalization distortion from outliers.
        
        Returns:
            Tuple of (robust_min, robust_max)
        """
        if len(self.score_history) < 10:
            # Use simple bounds if insufficient data
            return self.min_score, self.max_score
        
        lower_bound = np.percentile(self.score_history, self.percentile_bounds[0])
        upper_bound = np.percentile(self.score_history, self.percentile_bounds[1])
        
        return lower_bound, upper_bound
    
    def normalize(self, score: float, use_robust: bool = True) -> float:
        """
        Normalize with optional robust bounds.
        
        Args:
            score: Raw objective value
            use_robust: If True, use percentile-based bounds
            
        Returns:
            Normalized score in [0, 1]
        """
        if not np.isfinite(score):
            return 0.0
        
        if use_robust and len(self.score_history) >= 10:
            min_val, max_val = self.get_robust_bounds()
        else:
            min_val, max_val = self.min_score, self.max_score
        
        if max_val == min_val:
            return 0.5
        
        score_range = max_val - min_val + self.epsilon
        
        if self.optimization_type == 'min':
            normalized = (max_val - score) / score_range
        else:
            normalized = (score - min_val) / score_range
        
        return np.clip(normalized, 0.0, 1.0)
