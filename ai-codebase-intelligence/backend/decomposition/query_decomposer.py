"""LLM-driven question decomposer for complex multi-intent asks."""

from __future__ import annotations

import json

from core.logging import get_logger
from reasoning.openrouter_client import OpenRouterClient

logger = get_logger(__name__)

_DECOMPOSER_SYSTEM_PROMPT = """Split a repository question into focused sub-questions.
Rules:
- Return JSON array of strings only.
- If already focused, return array with exactly one item.
- Keep each sub-question self-contained.
- Return at most 4 items.
"""


class QueryDecomposer:
    """Decompose compound questions into sub-questions."""

    MODEL = "qwen/qwen-max"
    MAX_SUB_QUESTIONS = 4

    def __init__(self, llm_client: OpenRouterClient | None = None) -> None:
        self._llm = llm_client or OpenRouterClient(model=self.MODEL)

    async def decompose(self, question: str) -> list[str]:
        try:
            response = await self._llm.complete(
                prompt=f"Question: {question}",
                system_prompt=_DECOMPOSER_SYSTEM_PROMPT,
                model=self.MODEL,
                temperature=0.0,
                max_tokens=300,
            )
            parsed = self._parse_response(response or "")
            if parsed:
                return parsed[: self.MAX_SUB_QUESTIONS]
            return [question]
        except Exception as exc:
            logger.warning("query_decompose_failed", error=str(exc))
            return [question]

    @staticmethod
    def _parse_response(raw: str) -> list[str]:
        text = raw.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                items = [str(item).strip() for item in data if str(item).strip()]
                return items
        except json.JSONDecodeError:
            return [text] if text else []

        return []
