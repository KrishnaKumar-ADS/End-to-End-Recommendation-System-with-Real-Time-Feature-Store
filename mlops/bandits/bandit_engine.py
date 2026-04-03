import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# BANDIT ARM — Represents one item's click belief
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BanditArm:
    """
    A single Beta-distributed bandit arm for one item.
    
    alpha: number of clicks    + 1  (prior: 1 click seen)
    beta:  number of no-clicks + 1  (prior: 1 miss seen)
    
    Starting at Beta(1, 1) = uniform = "I know nothing about this item"
    """
    item_idx: int
    alpha: float = 1.0    # prior α = 1 → at least one success assumed
    beta: float = 1.0     # prior β = 1 → at least one failure assumed

    @property
    def total_trials(self) -> int:
        """Total observations: (alpha - 1) + (beta - 1)"""
        return int(self.alpha + self.beta - 2)

    @property
    def expected_ctr(self) -> float:
        """
        Expected click-through rate: mean of the Beta distribution.
        E[Beta(α, β)] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """
        Variance of the Beta distribution.
        Var[Beta(α, β)] = αβ / [(α+β)² (α+β+1)]
        High variance = high uncertainty = item needs more exploration.
        """
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self) -> float:
        """
        Sample one value from Beta(alpha, beta).
        This is the Thompson Sampling step.
        
        Returns a float in [0, 1].
        High alpha → high sample → item gets promoted.
        High beta  → low sample  → item gets demoted.
        New items (alpha=1, beta=1) → high variance → occasionally promoted.
        """
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: int) -> None:
        """
        Update the arm with a binary reward.
        
        reward = 1: user clicked this item → alpha += 1
        reward = 0: user did NOT click     → beta  += 1
        
        This is the Bayesian posterior update:
          Prior:     Beta(α, β)
          Likelihood: Bernoulli(reward)
          Posterior: Beta(α + reward, β + (1 - reward))
        """
        if reward not in (0, 1):
            raise ValueError(f"reward must be 0 or 1, got: {reward}")
        if reward == 1:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def to_dict(self) -> Dict:
        return {
            "item_idx": self.item_idx,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_trials": self.total_trials,
            "expected_ctr": round(self.expected_ctr, 4),
            "uncertainty": round(self.uncertainty, 6),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BanditArm":
        return cls(
            item_idx=int(d["item_idx"]),
            alpha=float(d["alpha"]),
            beta=float(d["beta"]),
        )

    def __repr__(self) -> str:
        return (
            f"BanditArm(item={self.item_idx}, "
            f"α={self.alpha:.1f}, β={self.beta:.1f}, "
            f"E[CTR]={self.expected_ctr:.3f}, "
            f"trials={self.total_trials})"
        )


# ──────────────────────────────────────────────────────────────────────
# THOMPSON SAMPLING BANDIT — Manages all arms in memory
# ──────────────────────────────────────────────────────────────────────

class ThompsonSamplingBandit:
    """
    Thompson Sampling Bandit for item re-ranking.
    
    Usage in the recommendation pipeline:
      1. LightGBM produces top-10 candidate items (ranked by predicted CTR)
      2. ThompsonSamplingBandit.rerank(candidate_item_ids) 
         → samples from each arm's Beta distribution
         → re-ranks by sampled values
         → returns final ordered list
      3. User receives recommendations
      4. User feedback (click or no-click) arrives via /feedback endpoint
      5. ThompsonSamplingBandit.update(item_id, reward) is called
         → arm alpha or beta is incremented
         → state is persisted to Redis via BanditStore
    
    In-memory design:
      Arms are cached in self._arms dict for fast access.
      Persistence (across server restarts) handled by BanditStore.
      BanditStore.load() restores all arms from Redis on startup.
    """

    def __init__(self, n_items: int, seed: Optional[int] = None):
        """
        Args:
            n_items: Total number of items in the catalog.
                     Used only to report coverage statistics.
            seed:    Optional random seed (for reproducibility in tests only).
                     In production, leave as None for true randomness.
        """
        self.n_items = n_items
        self._arms: Dict[int, BanditArm] = {}
        if seed is not None:
            np.random.seed(seed)
        logger.info(f"ThompsonSamplingBandit initialized | catalog_size={n_items}")

    def get_arm(self, item_idx: int) -> BanditArm:
        """
        Retrieve arm for an item. If not seen before, create with prior Beta(1,1).
        
        This lazy initialization means we never pre-allocate all 53,889 arms.
        New items automatically get a fresh uniform prior.
        """
        if item_idx not in self._arms:
            self._arms[item_idx] = BanditArm(item_idx=item_idx)
        return self._arms[item_idx]

    def rerank(
        self,
        candidate_item_ids: List[int],
        lgbm_scores: Optional[List[float]] = None,
        blend_weight: float = 0.3,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank a list of candidate items using Thompson Sampling.
        
        Two modes:
          blend_weight = 0.0: Pure Thompson Sampling (ignore LightGBM scores)
          blend_weight = 1.0: Pure LightGBM ranking (ignore bandit)
          blend_weight = 0.3: 30% LightGBM + 70% Thompson Sampling (default)
        
        Blending formula:
          final_score = blend_weight * lgbm_score_normalized
                      + (1 - blend_weight) * thompson_sample
        
        This ensures:
          - LightGBM's offline knowledge is not completely thrown away
          - Thompson Sampling drives exploration towards uncertain items
          - As more data is collected, Thompson Sampling converges
            to the true CTR, so blending becomes less important
        
        Args:
            candidate_item_ids: List of item indices (top-K from LightGBM)
            lgbm_scores:        Corresponding LightGBM predicted scores
                                (optional, used for blending)
            blend_weight:       Weight for LightGBM scores (0.0 to 1.0)
        
        Returns:
            List of (item_idx, final_score) tuples, sorted descending by score.
        """
        if not candidate_item_ids:
            return []

        n = len(candidate_item_ids)

        # Normalize LightGBM scores to [0, 1] if provided
        lgbm_normalized = np.zeros(n)
        if lgbm_scores is not None and blend_weight > 0.0:
            scores_arr = np.array(lgbm_scores, dtype=np.float64)
            s_min, s_max = scores_arr.min(), scores_arr.max()
            if s_max > s_min:
                lgbm_normalized = (scores_arr - s_min) / (s_max - s_min)
            else:
                lgbm_normalized = np.ones(n) * 0.5

        # Sample from each arm's Beta distribution
        thompson_samples = np.array([
            self.get_arm(item_idx).sample()
            for item_idx in candidate_item_ids
        ])

        # Blend scores
        if lgbm_scores is not None and blend_weight > 0.0:
            final_scores = (
                blend_weight * lgbm_normalized
                + (1.0 - blend_weight) * thompson_samples
            )
        else:
            final_scores = thompson_samples

        # Sort descending by final score
        sorted_indices = np.argsort(-final_scores)
        ranked = [
            (candidate_item_ids[i], float(final_scores[i]))
            for i in sorted_indices
        ]

        logger.debug(
            f"Bandit rerank: {n} candidates → "
            f"blend_weight={blend_weight} | "
            f"top_item={ranked[0][0]} score={ranked[0][1]:.4f}"
        )
        return ranked

    def update(self, item_idx: int, reward: int) -> None:
        """
        Update a single arm after observing a reward.
        
        Args:
            item_idx: The item that was shown to the user
            reward:   1 if user clicked, 0 if user did not click
        """
        arm = self.get_arm(item_idx)
        arm.update(reward)
        logger.debug(
            f"Bandit update: item={item_idx} | reward={reward} | "
            f"new_arm={arm}"
        )

    def batch_update(self, updates: List[Tuple[int, int]]) -> None:
        """
        Update multiple arms at once.
        
        Args:
            updates: List of (item_idx, reward) tuples
        """
        for item_idx, reward in updates:
            self.update(item_idx, reward)

    def get_stats(self) -> Dict:
        """Return summary statistics about the bandit state."""
        if not self._arms:
            return {"total_arms": 0, "total_trials": 0}

        alphas = [arm.alpha for arm in self._arms.values()]
        betas  = [arm.beta  for arm in self._arms.values()]
        trials = [arm.total_trials for arm in self._arms.values()]
        ctrs   = [arm.expected_ctr for arm in self._arms.values()]

        return {
            "total_arms":          len(self._arms),
            "catalog_coverage":    f"{len(self._arms)}/{self.n_items} "
                                   f"({100*len(self._arms)/self.n_items:.1f}%)",
            "total_trials":        sum(trials),
            "avg_trials_per_arm":  float(np.mean(trials)),
            "max_trials":          int(np.max(trials)),
            "avg_expected_ctr":    float(np.mean(ctrs)),
            "max_expected_ctr":    float(np.max(ctrs)),
            "min_expected_ctr":    float(np.min(ctrs)),
            "arms_with_10+_trials": sum(1 for t in trials if t >= 10),
        }

    def top_items_by_ctr(self, n: int = 20) -> List[Dict]:
        """Return the N items with highest expected CTR (most learned arms)."""
        sorted_arms = sorted(
            self._arms.values(),
            key=lambda a: a.expected_ctr,
            reverse=True
        )
        return [arm.to_dict() for arm in sorted_arms[:n]]

    def load_arms(self, arms_dict: Dict[int, BanditArm]) -> None:
        """Bulk-load arms (called by BanditStore on startup)."""
        self._arms.update(arms_dict)
        logger.info(f"Loaded {len(arms_dict)} arms from store")

    def __len__(self) -> int:
        return len(self._arms)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ThompsonSamplingBandit("
            f"arms={stats['total_arms']}, "
            f"trials={stats['total_trials']}, "
            f"avg_ctr={stats.get('avg_expected_ctr', 0):.3f})"
        )