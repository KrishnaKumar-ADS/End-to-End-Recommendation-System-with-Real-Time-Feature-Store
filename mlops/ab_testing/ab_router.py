import hashlib
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single A/B experiment."""
    experiment_id:  str
    description:    str
    variants:       List[str]           # e.g., ["mf", "sasrec"]
    traffic_split:  List[float]         # e.g., [0.50, 0.50]
    is_active:      bool = True
    created_at:     float = field(default_factory=time.time)

    def __post_init__(self):
        assert len(self.variants) == len(self.traffic_split), (
            "variants and traffic_split must have the same length"
        )
        assert abs(sum(self.traffic_split) - 1.0) < 1e-6, (
            f"traffic_split must sum to 1.0, got {sum(self.traffic_split):.4f}"
        )
        assert all(v > 0 for v in self.traffic_split), (
            "All traffic split values must be > 0"
        )

    @property
    def cumulative_splits(self) -> List[float]:
        """
        Cumulative traffic boundaries for bucket assignment.
        Example: [0.5, 0.5] → [0.5, 1.0]
        User with bucket 0.37 → falls in [0.0, 0.5) → variant "mf"
        User with bucket 0.73 → falls in [0.5, 1.0) → variant "sasrec"
        """
        cumulative = []
        total = 0.0
        for split in self.traffic_split:
            total += split
            cumulative.append(total)
        return cumulative

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "description":   self.description,
            "variants":      self.variants,
            "traffic_split": self.traffic_split,
            "is_active":     self.is_active,
            "created_at":    self.created_at,
        }


class ABRouter:
    """
    Routes users to experiment variants using deterministic hash-based assignment.
    
    Lifecycle:
      1. Create and register experiments at server startup
      2. In each request handler: call assign(user_id, experiment_id)
      3. Execute the pipeline for the assigned variant
      4. Log the exposure via ABLogger
    """

    def __init__(self):
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._register_default_experiments()

    def _register_default_experiments(self) -> None:
        """Register the standard DS19 experiments."""
        self.register_experiment(
            experiment_id="retrieval_v1",
            description="Matrix Factorization vs SASRec Transformer retrieval comparison",
            variants=["mf", "sasrec"],
            traffic_split=[0.50, 0.50],
        )
        self.register_experiment(
            experiment_id="ranking_blend_v1",
            description="Pure LightGBM vs LightGBM + Thompson Sampling blend",
            variants=["lgbm_only", "lgbm_bandit"],
            traffic_split=[0.50, 0.50],
        )

    def register_experiment(
        self,
        experiment_id: str,
        description: str,
        variants: List[str],
        traffic_split: List[float],
        is_active: bool = True,
    ) -> None:
        """
        Register a new experiment.
        
        Args:
            experiment_id: Unique ID (used in hash and logging)
            description:   Human-readable description
            variants:      List of variant names
            traffic_split: List of traffic fractions (must sum to 1.0)
            is_active:     If False, all users get the first variant (control)
        """
        config = ExperimentConfig(
            experiment_id=experiment_id,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
            is_active=is_active,
        )
        self._experiments[experiment_id] = config
        logger.info(
            f"Experiment registered: '{experiment_id}' | "
            f"variants={variants} | "
            f"split={traffic_split} | "
            f"active={is_active}"
        )

    def assign(self, user_id: int, experiment_id: str) -> str:
        """
        Assign a user to a variant for a given experiment.
        
        Algorithm:
          1. Hash the string "{user_id}:{experiment_id}" with MD5
          2. Convert first 8 hex chars to int
          3. Compute bucket = hash_int % 10000 / 10000.0  → float in [0, 1)
          4. Find which cumulative split boundary the bucket falls under
          5. Return the corresponding variant name
        
        Args:
            user_id:       The user's integer index
            experiment_id: Which experiment to assign for
        
        Returns:
            Variant name (e.g., "mf" or "sasrec")
        
        Raises:
            KeyError: If experiment_id is not registered
        """
        if experiment_id not in self._experiments:
            raise KeyError(
                f"Experiment '{experiment_id}' not registered. "
                f"Available: {list(self._experiments.keys())}"
            )

        config = self._experiments[experiment_id]

        # If experiment is paused, return control (first variant)
        if not config.is_active:
            return config.variants[0]

        # Compute deterministic hash
        hash_input  = f"{user_id}:{experiment_id}"
        hash_hex    = hashlib.md5(hash_input.encode()).hexdigest()
        hash_int    = int(hash_hex[:8], 16)
        bucket      = (hash_int % 10000) / 10000.0  # → [0.0, 1.0)

        # Find variant by cumulative split boundaries
        cumulative = config.cumulative_splits
        for i, boundary in enumerate(cumulative):
            if bucket < boundary:
                return config.variants[i]

        # Fallback (shouldn't happen due to float precision)
        return config.variants[-1]

    def assign_batch(
        self, user_ids: List[int], experiment_id: str
    ) -> Dict[int, str]:
        """Assign multiple users to variants in one call."""
        return {uid: self.assign(uid, experiment_id) for uid in user_ids}

    def get_variant_distribution(
        self, user_ids: List[int], experiment_id: str
    ) -> Dict[str, int]:
        """
        Compute the actual variant distribution for a set of users.
        Useful for validating the 50/50 split is working correctly.
        """
        assignments = self.assign_batch(user_ids, experiment_id)
        distribution: Dict[str, int] = {}
        for variant in assignments.values():
            distribution[variant] = distribution.get(variant, 0) + 1
        return distribution

    def deactivate_experiment(self, experiment_id: str) -> None:
        """Stop an experiment — all users go to control variant."""
        if experiment_id in self._experiments:
            self._experiments[experiment_id].is_active = False
            logger.info(f"Experiment deactivated: '{experiment_id}'")

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[Dict]:
        return [exp.to_dict() for exp in self._experiments.values()]