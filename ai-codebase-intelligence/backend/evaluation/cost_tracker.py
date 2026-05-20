"""Week 15 evaluation cost tracker with token counting and budget guards."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


class BudgetExceededError(RuntimeError):
    """Raised when evaluation token cost exceeds configured budget."""


@dataclass(slots=True)
class ModelPricing:
    input_per_1m_usd: float
    output_per_1m_usd: float


@dataclass(slots=True)
class CostRecord:
    repo_name: str
    question_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


PRICING_TABLE: dict[str, ModelPricing] = {
    "qwen/qwen-2.5-coder-32b-instruct": ModelPricing(0.18, 0.18),
    "qwen/qwen-max": ModelPricing(1.60, 6.40),
    "gemini-2.0-flash": ModelPricing(0.075, 0.30),
    "models/text-embedding-004": ModelPricing(0.0, 0.0),
    "models/gemini-embedding-001": ModelPricing(0.0, 0.0),
}


class CostTracker:
    """Track session, per-repo, and per-question token/cost totals."""

    def __init__(
        self,
        daily_budget_usd: float = 10.0,
        pricing_table: dict[str, ModelPricing] | None = None,
        openrouter_fee_pct: float = 0.05,
    ) -> None:
        self.daily_budget_usd = float(daily_budget_usd)
        self.pricing_table = pricing_table or PRICING_TABLE
        self.openrouter_fee_pct = float(openrouter_fee_pct)

        self.records: list[CostRecord] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

        self._encoding = None
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None

    def count_tokens(self, text: str) -> int:
        normalized = (text or "").strip()
        if not normalized:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(normalized))
        return max(1, len(normalized) // 4)

    def _resolve_pricing(self, model: str) -> ModelPricing:
        normalized = (model or "").strip().lower()
        for key, pricing in self.pricing_table.items():
            if key.lower() == normalized:
                return pricing
        # Fallback pricing if unknown model appears in runtime responses.
        return ModelPricing(0.25, 1.00)

    def estimate_cost_usd(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        safe_prompt = max(0, int(prompt_tokens))
        safe_completion = max(0, int(completion_tokens))

        pricing = self._resolve_pricing(model)
        input_cost = (safe_prompt / 1_000_000) * pricing.input_per_1m_usd
        output_cost = (safe_completion / 1_000_000) * pricing.output_per_1m_usd
        cost = input_cost + output_cost

        if (provider or "").strip().lower() in {"openrouter", "qwen-coder", "qwen-max"}:
            cost *= 1.0 + self.openrouter_fee_pct

        return round(cost, 8)

    def record_call(
        self,
        repo_name: str,
        question_id: str,
        provider: str,
        model: str,
        prompt_text: str | None = None,
        completion_text: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> CostRecord:
        actual_prompt_tokens = (
            self.count_tokens(prompt_text or "")
            if prompt_tokens is None
            else max(0, int(prompt_tokens))
        )
        actual_completion_tokens = (
            self.count_tokens(completion_text or "")
            if completion_tokens is None
            else max(0, int(completion_tokens))
        )

        cost_usd = self.estimate_cost_usd(
            provider=provider,
            model=model,
            prompt_tokens=actual_prompt_tokens,
            completion_tokens=actual_completion_tokens,
        )

        record = CostRecord(
            repo_name=repo_name,
            question_id=question_id,
            provider=(provider or "unknown").strip(),
            model=(model or "unknown").strip(),
            prompt_tokens=actual_prompt_tokens,
            completion_tokens=actual_completion_tokens,
            total_tokens=actual_prompt_tokens + actual_completion_tokens,
            cost_usd=cost_usd,
        )

        self.records.append(record)
        self.total_prompt_tokens += actual_prompt_tokens
        self.total_completion_tokens += actual_completion_tokens
        self.total_cost_usd = round(self.total_cost_usd + cost_usd, 8)

        if self.total_cost_usd > self.daily_budget_usd:
            raise BudgetExceededError(
                f"Evaluation budget exceeded: ${self.total_cost_usd:.4f} > ${self.daily_budget_usd:.4f}"
            )

        return record

    def per_repo_totals(self) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for record in self.records:
            repo = summary.setdefault(
                record.repo_name,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "question_count": 0,
                },
            )
            repo["prompt_tokens"] = int(repo["prompt_tokens"]) + record.prompt_tokens
            repo["completion_tokens"] = int(repo["completion_tokens"]) + record.completion_tokens
            repo["total_tokens"] = int(repo["total_tokens"]) + record.total_tokens
            repo["total_cost_usd"] = round(float(repo["total_cost_usd"]) + record.cost_usd, 8)
            repo["question_count"] = int(repo["question_count"]) + 1
        return summary

    def to_report(self) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "daily_budget_usd": self.daily_budget_usd,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "records": [asdict(record) for record in self.records],
            "per_repo": self.per_repo_totals(),
        }

    def save_report(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_report()
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
