import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.bandits.bandit_engine import BanditArm, ThompsonSamplingBandit


class TestBanditArm:
    """Unit tests for BanditArm."""

    def test_initial_state(self):
        """New arm starts with Beta(1,1) = uniform prior."""
        arm = BanditArm(item_idx=42)
        assert arm.alpha == 1.0
        assert arm.beta  == 1.0
        assert arm.total_trials == 0
        assert arm.expected_ctr == 0.5  # E[Beta(1,1)] = 0.5

    def test_update_click(self):
        """After a click, alpha increases by 1."""
        arm = BanditArm(item_idx=1)
        arm.update(reward=1)
        assert arm.alpha == 2.0
        assert arm.beta  == 1.0
        assert arm.total_trials == 1

    def test_update_no_click(self):
        """After a no-click, beta increases by 1."""
        arm = BanditArm(item_idx=1)
        arm.update(reward=0)
        assert arm.alpha == 1.0
        assert arm.beta  == 2.0
        assert arm.total_trials == 1

    def test_invalid_reward_raises(self):
        """Non-binary rewards should raise ValueError."""
        arm = BanditArm(item_idx=1)
        with pytest.raises(ValueError):
            arm.update(reward=2)
        with pytest.raises(ValueError):
            arm.update(reward=-1)

    def test_expected_ctr_converges(self):
        """After many observations, expected CTR converges to true CTR."""
        np.random.seed(42)
        arm = BanditArm(item_idx=1)
        true_ctr = 0.3
        for _ in range(1000):
            reward = int(np.random.random() < true_ctr)
            arm.update(reward)
        assert abs(arm.expected_ctr - true_ctr) < 0.05  # within 5%

    def test_sample_range(self):
        """Samples must be in [0, 1]."""
        arm = BanditArm(item_idx=1, alpha=10, beta=5)
        for _ in range(100):
            s = arm.sample()
            assert 0.0 <= s <= 1.0

    def test_serialization_roundtrip(self):
        """to_dict() and from_dict() are inverse operations."""
        arm = BanditArm(item_idx=99, alpha=15.0, beta=7.0)
        d = arm.to_dict()
        arm2 = BanditArm.from_dict(d)
        assert arm2.item_idx == 99
        assert arm2.alpha    == 15.0
        assert arm2.beta     == 7.0

    def test_uncertainty_decreases_with_observations(self):
        """More observations → lower variance → lower uncertainty."""
        arm_new  = BanditArm(item_idx=1, alpha=1,   beta=1)
        arm_mid  = BanditArm(item_idx=1, alpha=10,  beta=10)
        arm_lots = BanditArm(item_idx=1, alpha=100, beta=100)
        assert arm_new.uncertainty > arm_mid.uncertainty > arm_lots.uncertainty


class TestThompsonSamplingBandit:
    """Unit tests for ThompsonSamplingBandit."""

    def test_get_arm_creates_new(self):
        """Getting a new item creates a fresh arm with prior Beta(1,1)."""
        bandit = ThompsonSamplingBandit(n_items=100)
        arm = bandit.get_arm(42)
        assert arm.item_idx == 42
        assert arm.alpha    == 1.0

    def test_get_arm_returns_same(self):
        """Getting the same arm twice returns the same object."""
        bandit = ThompsonSamplingBandit(n_items=100)
        arm1 = bandit.get_arm(7)
        arm1.update(1)
        arm2 = bandit.get_arm(7)
        assert arm2.alpha == 2.0  # same object, updated

    def test_rerank_returns_all_items(self):
        """Reranked list contains exactly the same items as input."""
        bandit = ThompsonSamplingBandit(n_items=1000, seed=42)
        candidates = [10, 20, 30, 40, 50]
        ranked = bandit.rerank(candidates)
        returned_ids = [item_id for item_id, _ in ranked]
        assert set(returned_ids) == set(candidates)
        assert len(returned_ids) == len(candidates)

    def test_rerank_scores_in_range(self):
        """All bandit scores should be in [0, 1]."""
        bandit = ThompsonSamplingBandit(n_items=1000, seed=42)
        candidates = list(range(10))
        ranked = bandit.rerank(candidates)
        for _, score in ranked:
            assert 0.0 <= score <= 1.0

    def test_good_item_ranked_higher_on_average(self):
        """An item with many clicks should rank above one with no clicks."""
        bandit = ThompsonSamplingBandit(n_items=1000, seed=42)

        # Give item 1 lots of clicks
        for _ in range(100):
            bandit.update(1, reward=1)   # 100 clicks

        # Give item 2 lots of no-clicks
        for _ in range(100):
            bandit.update(2, reward=0)   # 100 no-clicks

        # Over many samples, item 1 should rank first more often
        item1_wins = 0
        n_trials = 500
        for _ in range(n_trials):
            ranked = bandit.rerank([1, 2])
            if ranked[0][0] == 1:
                item1_wins += 1

        # Item 1 should win > 95% of the time
        win_rate = item1_wins / n_trials
        assert win_rate > 0.95, f"Item 1 win rate too low: {win_rate:.2f}"

    def test_empty_candidates_returns_empty(self):
        """Reranking an empty list returns empty list."""
        bandit = ThompsonSamplingBandit(n_items=100)
        result = bandit.rerank([])
        assert result == []

    def test_update_and_stats(self):
        """After updates, stats reflect the accumulated data."""
        bandit = ThompsonSamplingBandit(n_items=1000)
        bandit.update(1, reward=1)
        bandit.update(1, reward=1)
        bandit.update(2, reward=0)

        stats = bandit.get_stats()
        assert stats["total_arms"]   == 2
        assert stats["total_trials"] == 3

    def test_blend_weight_zero_pure_bandit(self):
        """With blend_weight=0, LightGBM scores are completely ignored."""
        bandit = ThompsonSamplingBandit(n_items=1000, seed=42)
        candidates  = [1, 2, 3]
        lgbm_scores = [100.0, 100.0, 100.0]  # all equal → blend doesn't matter
        ranked = bandit.rerank(candidates, lgbm_scores=lgbm_scores, blend_weight=0.0)
        assert len(ranked) == 3