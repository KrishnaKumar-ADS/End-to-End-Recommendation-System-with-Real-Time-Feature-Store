"""Pydantic models for LLM-judge quality scoring."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class QualityScore(BaseModel):
    faithfulness: float = Field(default=0.0)
    relevance: float = Field(default=0.0)
    completeness: float = Field(default=0.0)
    overall: float = Field(default=0.0)
    critique: str = ""

    skipped: bool = False
    skip_reason: str | None = None
    judge_model: str = "qwen/qwen-max"

    @model_validator(mode="after")
    def _normalize(self) -> "QualityScore":
        if self.skipped:
            self.faithfulness = 0.0
            self.relevance = 0.0
            self.completeness = 0.0
            self.overall = 0.0
            return self

        self.faithfulness = _clamp_score(self.faithfulness)
        self.relevance = _clamp_score(self.relevance)
        self.completeness = _clamp_score(self.completeness)
        self.overall = round(
            self.faithfulness * 0.5 + self.relevance * 0.3 + self.completeness * 0.2,
            6,
        )
        return self

    @classmethod
    def skipped_score(cls, reason: str) -> "QualityScore":
        return cls(
            skipped=True,
            skip_reason=reason,
            critique=reason,
            judge_model="",
        )
