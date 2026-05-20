"""Unit tests for cost_tracking rates and tracker."""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import pytest

from cost_tracking.models import BudgetExceededError
from cost_tracking.rates import estimate_cost_usd, get_rate
from cost_tracking.tracker import CostTracker


class TestCostRateEstimation:
    def test_qwen_max_costs_more_than_qwen_coder(self):
        cost_max = estimate_cost_usd("openrouter", "qwen/qwen-max", 1000, 200)
        cost_coder = estimate_cost_usd("openrouter", "qwen/qwen-2.5-coder-32b-instruct", 1000, 200)
        assert cost_max > cost_coder

    def test_gemini_embedding_is_free(self):
        cost = estimate_cost_usd("gemini", "text-embedding-004", 10000, 0)
        assert cost == 0.0

    def test_cost_increases_with_token_count(self):
        cost_small = estimate_cost_usd("openrouter", "qwen/qwen-max", 100, 50)
        cost_large = estimate_cost_usd("openrouter", "qwen/qwen-max", 1000, 500)
        assert cost_large > cost_small

    def test_unknown_model_uses_default_rate(self):
        cost = estimate_cost_usd("openrouter", "some-unknown-model-xyz", 1000, 200)
        assert cost > 0.0

    def test_get_rate_exact_match(self):
        rate = get_rate("openrouter", "qwen/qwen-max")
        assert rate.input_per_1m > 0
        assert rate.output_per_1m > 0

    def test_get_rate_case_insensitive(self):
        rate1 = get_rate("OpenRouter", "Qwen/Qwen-Max")
        rate2 = get_rate("openrouter", "qwen/qwen-max")
        assert rate1 == rate2

    def test_output_tokens_more_expensive_than_input_for_qwen_max(self):
        rate = get_rate("openrouter", "qwen/qwen-max")
        assert rate.output_per_1m > rate.input_per_1m


@pytest.fixture
def tracker_with_fake_redis():
    tracker = CostTracker()
    tracker._redis = fakeredis.FakeRedis(decode_responses=True)
    return tracker


class TestCostTrackerRecord:
    @pytest.mark.asyncio
    async def test_record_returns_positive_cost(self, tracker_with_fake_redis):
        with patch("cost_tracking.tracker.get_settings") as mock_s:
            mock_s.return_value.daily_budget_usd = 100.0
            mock_s.return_value.redis_url = "redis://localhost"

            cost = await tracker_with_fake_redis.record(
                "openrouter", "qwen/qwen-max", 1000, 200, check_budget=False
            )
        assert cost > 0.0

    @pytest.mark.asyncio
    async def test_record_increments_counters(self, tracker_with_fake_redis):
        with patch("cost_tracking.tracker.get_settings") as mock_s:
            mock_s.return_value.daily_budget_usd = 100.0
            mock_s.return_value.redis_url = "redis://localhost"

            await tracker_with_fake_redis.record(
                "openrouter", "qwen/qwen-max", 1000, 200, check_budget=False
            )
            await tracker_with_fake_redis.record(
                "openrouter", "qwen/qwen-max", 500, 100, check_budget=False
            )

            summary = await tracker_with_fake_redis.get_daily_summary()

        entry = summary.per_provider["openrouter"]
        assert entry["input_tokens"] == 1500
        assert entry["output_tokens"] == 300
        assert entry["call_count"] == 2

    @pytest.mark.asyncio
    async def test_record_raises_budget_exceeded(self, tracker_with_fake_redis):
        with patch("cost_tracking.tracker.get_settings") as mock_s:
            mock_s.return_value.daily_budget_usd = 0.000001
            mock_s.return_value.redis_url = "redis://localhost"

            with pytest.raises(BudgetExceededError):
                await tracker_with_fake_redis.record(
                    "openrouter",
                    "qwen/qwen-max",
                    input_tokens=100000,
                    output_tokens=20000,
                    check_budget=True,
                )

    @pytest.mark.asyncio
    async def test_different_providers_tracked_separately(self, tracker_with_fake_redis):
        with patch("cost_tracking.tracker.get_settings") as mock_s:
            mock_s.return_value.daily_budget_usd = 100.0
            mock_s.return_value.redis_url = "redis://localhost"

            await tracker_with_fake_redis.record("openrouter", "qwen/qwen-max", 500, 100, check_budget=False)
            await tracker_with_fake_redis.record("gemini", "gemini-2.0-flash", 500, 100, check_budget=False)

            summary = await tracker_with_fake_redis.get_daily_summary()

        assert summary.per_provider["openrouter"]["call_count"] == 1
        assert summary.per_provider["gemini"]["call_count"] == 1


class TestCostTrackerSummary:
    @pytest.mark.asyncio
    async def test_summary_budget_fields_correct(self, tracker_with_fake_redis):
        with patch("cost_tracking.tracker.get_settings") as mock_s:
            mock_s.return_value.daily_budget_usd = 5.0
            mock_s.return_value.redis_url = "redis://localhost"

            summary = await tracker_with_fake_redis.get_daily_summary()

        assert summary.budget_limit_usd == 5.0
        assert summary.over_budget is False
        assert 0.0 <= summary.budget_used_pct <= 100.0

    @pytest.mark.asyncio
    async def test_summary_returns_zero_for_fresh_redis(self, tracker_with_fake_redis):
        with patch("cost_tracking.tracker.get_settings") as mock_s:
            mock_s.return_value.daily_budget_usd = 5.0
            mock_s.return_value.redis_url = "redis://localhost"

            summary = await tracker_with_fake_redis.get_daily_summary()

        assert summary.total_cost_usd == 0.0
        assert summary.total_tokens == 0
