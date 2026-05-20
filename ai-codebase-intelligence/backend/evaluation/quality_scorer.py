"""Week 15 quality scorer using LLM-as-judge with strict JSON parsing."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from evaluation.prompts import (
    QUALITY_JUDGE_SYSTEM_PROMPT,
    build_quality_judge_user_message,
)


@dataclass(slots=True)
class QualityScore:
    question_id: str
    repo_name: str
    faithfulness: int | None
    relevance: int | None
    completeness: int | None
    composite_score: float | None
    critique: str
    judge_model: str
    retries_used: int
    raw_response: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QualityScorer:
    """Score answers on faithfulness/relevance/completeness using a judge model."""

    def __init__(
        self,
        judge_model: str = "qwen/qwen-max",
        openrouter_api_key: str | None = None,
        timeout_seconds: int = 60,
        max_retries: int = 2,
        judge_callable: Callable[[str, str], str] | None = None,
    ) -> None:
        self.judge_model = judge_model
        self.openrouter_api_key = openrouter_api_key
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)
        self.judge_callable = judge_callable

    def _parse_json_payload(self, text: str) -> dict[str, int | str] | None:
        candidate = (text or "").strip()
        if not candidate:
            return None

        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate).strip()

        if not candidate.startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            candidate = candidate[start : end + 1]

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        required = ("faithfulness", "relevance", "completeness", "critique")
        if any(key not in payload for key in required):
            return None

        parsed: dict[str, int | str] = {}
        for key in ("faithfulness", "relevance", "completeness"):
            try:
                value = int(payload[key])
            except (TypeError, ValueError):
                return None
            if value < 1 or value > 5:
                return None
            parsed[key] = value

        parsed["critique"] = str(payload.get("critique", "")).strip()
        return parsed

    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for quality scoring.")

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.judge_model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("Judge response had no choices.")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Judge response content is empty.")
            return content.strip()

    def _judge(self, system_prompt: str, user_prompt: str) -> str:
        if self.judge_callable is not None:
            return self.judge_callable(system_prompt, user_prompt)
        return self._call_openrouter(system_prompt, user_prompt)

    def score_answer(
        self,
        question_id: str,
        repo_name: str,
        question: str,
        retrieved_context: str | list[str],
        generated_answer: str,
    ) -> QualityScore:
        user_prompt = build_quality_judge_user_message(
            question=question,
            generated_answer=generated_answer,
            retrieved_context=retrieved_context,
        )

        last_raw = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._judge(QUALITY_JUDGE_SYSTEM_PROMPT, user_prompt)
            except Exception as exc:
                last_raw = str(exc)
                continue

            last_raw = raw
            parsed = self._parse_json_payload(raw)
            if parsed is None:
                continue

            faithfulness = int(parsed["faithfulness"])
            relevance = int(parsed["relevance"])
            completeness = int(parsed["completeness"])
            composite = round((faithfulness + relevance + completeness) / 3 / 5, 6)

            return QualityScore(
                question_id=question_id,
                repo_name=repo_name,
                faithfulness=faithfulness,
                relevance=relevance,
                completeness=completeness,
                composite_score=composite,
                critique=str(parsed["critique"]),
                judge_model=self.judge_model,
                retries_used=attempt,
                raw_response=raw,
            )

        return QualityScore(
            question_id=question_id,
            repo_name=repo_name,
            faithfulness=None,
            relevance=None,
            completeness=None,
            composite_score=None,
            critique="Judge response invalid after retries.",
            judge_model=self.judge_model,
            retries_used=self.max_retries,
            raw_response=last_raw,
        )

    def score_many(self, items: list[dict]) -> list[QualityScore]:
        scores: list[QualityScore] = []
        for item in items:
            scores.append(
                self.score_answer(
                    question_id=str(item.get("question_id", "unknown")),
                    repo_name=str(item.get("repo_name", "unknown")),
                    question=str(item.get("question", "")),
                    retrieved_context=item.get("retrieved_context", ""),
                    generated_answer=str(item.get("generated_answer", "")),
                )
            )
        return scores

    @staticmethod
    def save_scores(scores: list[QualityScore], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scores": [asdict(score) for score in scores],
            "avg_composite": round(
                sum(score.composite_score for score in scores if score.composite_score is not None)
                / max(1, len([score for score in scores if score.composite_score is not None])),
                6,
            ),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
