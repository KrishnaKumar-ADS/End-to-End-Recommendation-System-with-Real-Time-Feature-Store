"""Models and exceptions for cost tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    def __init__(self, daily_limit_usd: float, used_usd: float) -> None:
        super().__init__(
            f"Daily budget exceeded: limit=${daily_limit_usd:.4f}, used=${used_usd:.4f}"
        )
        self.daily_limit_usd = float(daily_limit_usd)
        self.used_usd = float(used_usd)


@dataclass
class DailyCostSummary:
    date: str
    total_cost_usd: float
    total_tokens: int
    budget_limit_usd: float
    budget_used_pct: float
    remaining_usd: float
    over_budget: bool
    per_provider: dict[str, dict[str, int | float]] = field(default_factory=dict)
