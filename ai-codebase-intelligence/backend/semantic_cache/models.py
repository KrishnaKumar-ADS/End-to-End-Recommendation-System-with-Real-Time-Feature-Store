"""Data models for semantic answer caching."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CachedAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    quality_score: dict | None = None
    repo_id: str = ""
    provider_used: str = ""
    model_used: str = ""
    cached_at: int = 0
    expires_at: int = 0


@dataclass
class CacheLookupResult:
    found: bool = False
    similarity: float = 0.0
    cached_answer: CachedAnswer | None = None
    reason: str = ""
