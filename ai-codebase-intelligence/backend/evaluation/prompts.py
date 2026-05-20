"""Prompt helpers for quality evaluation."""

from __future__ import annotations

from typing import Iterable

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator for retrieval-augmented code answers.
Score the answer from 0.0 to 1.0 on these dimensions:
- faithfulness: Is the answer supported by the provided context?
- relevance: Does the answer directly address the user's question?
- completeness: Does the answer cover the key parts needed?

Return ONLY valid JSON with keys:
{
  "faithfulness": <float 0.0-1.0>,
  "relevance": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "critique": "<short explanation>"
}
""".strip()


# Week 15 judge prompt (1-5 scale) for automated benchmark scoring.
QUALITY_JUDGE_SYSTEM_PROMPT = """
You are an objective evaluator for retrieval-augmented code answers.
Given a question, retrieved context, and model answer, score:
- faithfulness (1-5): every claim must be supported by provided context
- relevance (1-5): answer must directly address the question
- completeness (1-5): answer should cover important requested aspects

Rules:
- Return ONLY strict JSON.
- Integers only for scores.
- If context is insufficient, lower faithfulness and completeness.

JSON schema:
{
  "faithfulness": 1,
  "relevance": 1,
  "completeness": 1,
  "critique": "one sentence describing the main weakness"
}
""".strip()


QUALITY_JUDGE_USER_TEMPLATE = """
Question:
{question}

Retrieved Context:
{retrieved_context}

Generated Answer:
{generated_answer}

Return strict JSON only.
""".strip()


EVAL_QUESTION_CATEGORIES: dict[str, list[str]] = {
    "architecture": [
        "Give a high-level architecture overview for this repository.",
        "Explain the request lifecycle from entrypoint to response.",
        "Describe module boundaries and data flow between core components.",
        "Identify the main extensibility points and why they exist.",
        "Explain how dependencies are layered and where coupling is highest.",
    ],
    "code_explanation": [
        "Explain what this function/class does and its key invariants.",
        "Trace this method call path and explain each major step.",
        "What are the key edge cases handled in this implementation?",
        "Explain how configuration values are loaded and validated.",
        "Summarize how error handling works in this component.",
    ],
    "bug_trace": [
        "Trace where this error would be raised and propagated.",
        "Identify likely root causes for an intermittent timeout here.",
        "Explain why this API call might return stale data.",
        "Find where a missing null-check could cause a runtime crash.",
        "Trace a failure path from request input to failing dependency.",
    ],
    "security": [
        "Identify potential injection risks in this code path.",
        "Assess credential handling and possible secret exposure risks.",
        "Point out missing validation that could allow abuse.",
        "Analyze access control checks for possible bypasses.",
        "List top security hardening improvements for this repository.",
    ],
}


def _truncate_context(text: str, max_context_chars: int) -> str:
    if len(text) <= max_context_chars:
        return text
    return text[:max_context_chars] + "\n[truncated]"


def build_judge_user_message(
    question: str,
    answer: str,
    context_chunks: list[str],
    max_context_chars: int = 6000,
) -> str:
    normalized_question = (question or "").strip()
    normalized_answer = (answer or "").strip()

    if len(normalized_answer) > 3000:
        normalized_answer = normalized_answer[:3000] + "... [truncated]"

    if context_chunks:
        labeled_chunks = [f"[Chunk {index}]\n{chunk}" for index, chunk in enumerate(context_chunks, start=1)]
        context_block = "\n\n".join(labeled_chunks)
        context_block = _truncate_context(context_block, max_context_chars)
    else:
        context_block = "[No context retrieved]"

    return (
        "Question:\n"
        f"{normalized_question}\n\n"
        "Answer:\n"
        f"{normalized_answer}\n\n"
        "Retrieved Context:\n"
        f"{context_block}\n\n"
        "Return strict JSON only."
    )


def build_quality_judge_user_message(
    question: str,
    generated_answer: str,
    retrieved_context: str | Iterable[str],
    max_context_chars: int = 8000,
) -> str:
    """Build a deterministic Week 15 judge prompt payload."""
    if isinstance(retrieved_context, str):
        context_text = retrieved_context
    else:
        context_text = "\n\n".join(str(chunk) for chunk in retrieved_context)

    context_text = _truncate_context(context_text.strip(), max_context_chars)

    return QUALITY_JUDGE_USER_TEMPLATE.format(
        question=(question or "").strip(),
        retrieved_context=context_text or "[No context provided]",
        generated_answer=(generated_answer or "").strip(),
    )
