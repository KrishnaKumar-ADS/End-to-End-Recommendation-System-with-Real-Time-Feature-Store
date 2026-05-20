"""LLM-powered intent classifier for question routing."""

from __future__ import annotations

from typing import Literal

from core.logging import get_logger
from reasoning.openrouter_client import OpenRouterClient

logger = get_logger(__name__)

IntentType = Literal[
    "code_explanation",
    "bug_trace",
    "architecture",
    "security",
    "general",
]

_VALID_INTENTS = {
    "code_explanation",
    "bug_trace",
    "architecture",
    "security",
    "general",
}

_CLASSIFIER_SYSTEM_PROMPT = """You classify repository questions into one label only.
Valid labels: code_explanation, bug_trace, architecture, security, general.
Return only one label with no punctuation or explanation.
If unsure return general.
"""


class IntentClassifier:
    """Classify user questions into intent categories for prompt/model routing."""

    CLASSIFIER_MODEL = "qwen/qwen-max"

    def __init__(self, llm_client: OpenRouterClient | None = None) -> None:
        self._llm = llm_client or OpenRouterClient(model=self.CLASSIFIER_MODEL)

    async def classify(self, question: str) -> IntentType:
        try:
            response = await self._llm.complete(
                prompt=f"Question: {question}",
                system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
                model=self.CLASSIFIER_MODEL,
                temperature=0.0,
                max_tokens=16,
            )
            token = (response or "").strip().lower().split()[0] if response else "general"
            if token in _VALID_INTENTS:
                return token  # type: ignore[return-value]
            logger.warning("intent_classifier_invalid_label", label=token)
            return "general"
        except Exception as exc:
            logger.warning("intent_classifier_failed", error=str(exc))
            return "general"
