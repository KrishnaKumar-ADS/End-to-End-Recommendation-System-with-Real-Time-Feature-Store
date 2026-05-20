"""Quality evaluator that uses an LLM judge and safe fallbacks."""

from __future__ import annotations

import json
import re

import redis

from core.config import get_settings
from core.logging import get_logger
from evaluation.models import QualityScore
from evaluation.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_message

logger = get_logger(__name__)


class QualityEvaluator:
    def __init__(self, judge_model: str = "qwen/qwen-max") -> None:
        self._judge_model = judge_model
        self._client = None

        try:
            from reasoning.openrouter_client import OpenRouterClient

            self._client = OpenRouterClient(model=judge_model)
        except Exception as exc:
            logger.warning(
                "quality_evaluator_client_unavailable",
                error=str(exc),
                judge_model=judge_model,
            )

    async def score(
        self,
        question: str,
        answer: str,
        context_chunks: list[str],
    ) -> QualityScore:
        normalized_answer = (answer or "").strip()
        if len(normalized_answer) < 40:
            return QualityScore.skipped_score("Answer too short for quality scoring.")

        if not context_chunks:
            return QualityScore.skipped_score("No context chunks available for quality scoring.")

        if self._client is None:
            return QualityScore.skipped_score("Quality judge client unavailable.")

        prompt = build_judge_user_message(
            question=question,
            answer=normalized_answer,
            context_chunks=context_chunks,
        )

        try:
            raw = await self._client.complete(
                prompt=prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                model=self._judge_model,
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as exc:
            logger.warning("quality_scoring_failed", error=str(exc))
            return QualityScore.skipped_score(f"Quality scoring failed: {exc}")

        parsed = self._parse_response(raw)
        if parsed is None:
            return QualityScore.skipped_score("Judge returned invalid JSON.")

        return QualityScore(
            faithfulness=parsed["faithfulness"],
            relevance=parsed["relevance"],
            completeness=parsed["completeness"],
            critique=parsed.get("critique", ""),
            judge_model=self._judge_model,
        )

    @staticmethod
    def _parse_response(raw: str) -> dict | None:
        text = (raw or "").strip()
        if not text:
            return None

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\\s*```$", "", text).strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            text = text[start : end + 1]

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None

        required = ("faithfulness", "relevance", "completeness")
        if not all(key in payload for key in required):
            return None

        parsed: dict[str, float | str] = {}
        for key in required:
            try:
                parsed[key] = float(payload[key])
            except (TypeError, ValueError):
                return None

        parsed["critique"] = str(payload.get("critique", "")).strip()
        return parsed


async def update_running_averages(score: QualityScore) -> None:
    """Update Redis EMA scores used by the metrics endpoint."""
    if score.skipped:
        return

    alpha = 0.1

    try:
        settings = get_settings()
        redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        pipe = redis_client.pipeline(transaction=True)

        for field_name, new_value in (
            ("faithfulness", score.faithfulness),
            ("relevance", score.relevance),
            ("completeness", score.completeness),
            ("overall", score.overall),
        ):
            key = f"eval:avg:{field_name}"
            old_value = float(redis_client.get(key) or new_value)
            ema = round(alpha * new_value + (1 - alpha) * old_value, 6)
            pipe.set(key, ema)

        pipe.incr("eval:count")
        pipe.execute()
    except Exception:
        # Metrics updates are best-effort and should never fail the request path.
        return
