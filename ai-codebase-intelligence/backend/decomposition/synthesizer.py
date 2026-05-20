"""Merge partial sub-query answers into one coherent response."""

from __future__ import annotations

from core.logging import get_logger
from reasoning.openrouter_client import OpenRouterClient

logger = get_logger(__name__)

_SYNTH_SYSTEM_PROMPT = """You synthesize multiple partial technical answers.
Return one coherent final answer that preserves concrete evidence and avoids repetition.
Do not introduce facts not present in partial answers.
"""


class AnswerSynthesizer:
    """Synthesize partial answers from deep ask workflow."""

    MODEL = "qwen/qwen-max"

    def __init__(self, llm_client: OpenRouterClient | None = None) -> None:
        self._llm = llm_client or OpenRouterClient(model=self.MODEL)

    async def synthesize(
        self,
        original_question: str,
        sub_questions: list[str],
        partial_answers: list[str],
    ) -> str:
        if len(partial_answers) <= 1:
            return partial_answers[0] if partial_answers else ""

        sections: list[str] = [f"Original question: {original_question}"]
        for idx, (sub_question, answer) in enumerate(zip(sub_questions, partial_answers), start=1):
            sections.append(f"Sub-question {idx}: {sub_question}\nPartial answer {idx}:\n{answer}")

        prompt = "\n\n---\n\n".join(sections)

        try:
            return await self._llm.complete(
                prompt=prompt,
                system_prompt=_SYNTH_SYSTEM_PROMPT,
                model=self.MODEL,
                temperature=0.2,
                max_tokens=2200,
            )
        except Exception as exc:
            logger.warning("answer_synthesis_failed", error=str(exc))
            fallback: list[str] = []
            for sub_question, answer in zip(sub_questions, partial_answers):
                fallback.append(f"{sub_question}\n{answer}")
            return "\n\n".join(fallback)
