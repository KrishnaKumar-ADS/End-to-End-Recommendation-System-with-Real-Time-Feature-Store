"""Redis-backed per-day token and cost tracking."""

from __future__ import annotations

from datetime import datetime, timezone

import redis

from core.config import get_settings
from core.logging import get_logger
from cost_tracking.models import BudgetExceededError, DailyCostSummary
from cost_tracking.rates import estimate_cost_usd

logger = get_logger(__name__)

_PROVIDER_NAMES = ("openrouter", "gemini", "deepseek")


class CostTracker:
    def __init__(self) -> None:
        settings = get_settings()
        self._redis = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    @staticmethod
    def _today_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _provider_hash_key(day_key: str, provider: str) -> str:
        return f"cost:{day_key}:provider:{provider}"

    @staticmethod
    def _total_cost_key(day_key: str) -> str:
        return f"cost:{day_key}:total_micro_usd"

    async def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        check_budget: bool = True,
    ) -> float:
        normalized_provider = (provider or "unknown").strip().lower()
        normalized_model = (model or "unknown").strip()

        safe_input = max(0, int(input_tokens))
        safe_output = max(0, int(output_tokens))
        total_tokens = safe_input + safe_output

        cost_usd = estimate_cost_usd(
            provider=normalized_provider,
            model=normalized_model,
            input_tokens=safe_input,
            output_tokens=safe_output,
        )
        cost_micro_usd = int(round(cost_usd * 1_000_000))

        day_key = self._today_key()
        provider_key = self._provider_hash_key(day_key, normalized_provider)
        total_key = self._total_cost_key(day_key)

        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.hincrby(provider_key, "input_tokens", safe_input)
            pipe.hincrby(provider_key, "output_tokens", safe_output)
            pipe.hincrby(provider_key, "total_tokens", total_tokens)
            pipe.hincrby(provider_key, "call_count", 1)
            pipe.hincrby(provider_key, "cost_micro_usd", cost_micro_usd)
            pipe.hset(provider_key, "last_model", normalized_model)
            pipe.incrby(total_key, cost_micro_usd)
            pipe.expire(provider_key, 60 * 60 * 48)
            pipe.expire(total_key, 60 * 60 * 48)
            pipe.execute()
        except Exception as exc:
            logger.warning("cost_tracker_record_failed", error=str(exc))

        if check_budget:
            summary = await self.get_daily_summary()
            if summary.over_budget:
                raise BudgetExceededError(
                    daily_limit_usd=summary.budget_limit_usd,
                    used_usd=summary.total_cost_usd,
                )

        return cost_usd

    async def get_daily_summary(self) -> DailyCostSummary:
        settings = get_settings()
        budget_limit = float(settings.daily_budget_usd)
        day_key = self._today_key()

        per_provider: dict[str, dict[str, int | float]] = {}
        total_tokens = 0
        accumulated_micro = 0

        for provider in _PROVIDER_NAMES:
            provider_key = self._provider_hash_key(day_key, provider)
            try:
                raw = self._redis.hgetall(provider_key)
            except Exception:
                raw = {}

            input_tokens = int(raw.get("input_tokens", 0) or 0)
            output_tokens = int(raw.get("output_tokens", 0) or 0)
            provider_total_tokens = int(raw.get("total_tokens", input_tokens + output_tokens) or 0)
            call_count = int(raw.get("call_count", 0) or 0)
            cost_micro = int(raw.get("cost_micro_usd", 0) or 0)
            cost_usd = round(cost_micro / 1_000_000, 6)

            per_provider[provider] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": provider_total_tokens,
                "call_count": call_count,
                "cost_today_usd": cost_usd,
            }

            total_tokens += provider_total_tokens
            accumulated_micro += cost_micro

        if accumulated_micro == 0:
            try:
                accumulated_micro = int(self._redis.get(self._total_cost_key(day_key)) or 0)
            except Exception:
                accumulated_micro = 0

        total_cost_usd = round(accumulated_micro / 1_000_000, 6)

        if budget_limit <= 0:
            budget_used_pct = 0.0
            remaining_usd = 0.0
            over_budget = False
        else:
            budget_used_pct = min(100.0, round((total_cost_usd / budget_limit) * 100, 3))
            remaining_usd = round(max(0.0, budget_limit - total_cost_usd), 6)
            over_budget = total_cost_usd >= budget_limit

        return DailyCostSummary(
            date=day_key,
            total_cost_usd=total_cost_usd,
            total_tokens=total_tokens,
            budget_limit_usd=budget_limit,
            budget_used_pct=budget_used_pct,
            remaining_usd=remaining_usd,
            over_budget=over_budget,
            per_provider=per_provider,
        )
