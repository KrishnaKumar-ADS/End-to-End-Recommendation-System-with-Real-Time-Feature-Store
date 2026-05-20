"""Greedy context window manager that packs top chunks under token budget."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.logging import get_logger
from context_manager.token_counter import estimate_tokens, estimate_tokens_for_chunk, get_context_budget

logger = get_logger(__name__)


@dataclass
class PackedContext:
    """Result payload for one context-packing operation."""

    selected_chunks: list[dict] = field(default_factory=list)
    dropped_chunks: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = 0
    prompt: str = ""


class ContextWindowManager:
    """Fit prompt chunks into the selected model token budget."""

    def __init__(self, model: str = "qwen/qwen-2.5-coder-32b-instruct") -> None:
        self.model = model

    def pack(
        self,
        system_prompt: str,
        history_block: str,
        code_chunks: list[dict],
        question: str,
    ) -> PackedContext:
        """Select highest priority chunks that fit and assemble full prompt text."""
        history_tokens = estimate_tokens(history_block)
        budget_tokens = get_context_budget(model=self.model, history_tokens=history_tokens)

        ranked_chunks = sorted(
            code_chunks,
            key=lambda item: float(item.get("importance_score", item.get("score", 0.0)) or 0.0),
            reverse=True,
        )

        selected: list[dict] = []
        dropped: list[dict] = []
        used = 0

        for chunk in ranked_chunks:
            tokens = estimate_tokens_for_chunk(chunk)
            if used + tokens <= budget_tokens:
                selected.append(chunk)
                used += tokens
            else:
                dropped.append(chunk)

        prompt = self._assemble_prompt(
            system_prompt=system_prompt,
            history_block=history_block,
            selected_chunks=selected,
            question=question,
        )

        total_tokens = (
            estimate_tokens(system_prompt)
            + history_tokens
            + used
            + estimate_tokens(question)
        )

        logger.info(
            "context_window_packed",
            model=self.model,
            selected=len(selected),
            dropped=len(dropped),
            used_tokens=used,
            budget_tokens=budget_tokens,
            total_tokens=total_tokens,
        )

        return PackedContext(
            selected_chunks=selected,
            dropped_chunks=dropped,
            total_tokens=total_tokens,
            budget_tokens=budget_tokens,
            prompt=prompt,
        )

    @staticmethod
    def _assemble_prompt(
        system_prompt: str,
        history_block: str,
        selected_chunks: list[dict],
        question: str,
    ) -> str:
        parts: list[str] = [system_prompt.strip()]

        if history_block.strip():
            parts.append("<conversation_history>\n" + history_block.strip() + "\n</conversation_history>")

        if selected_chunks:
            blocks: list[str] = []
            for chunk in selected_chunks:
                file_path = str(chunk.get("file_path", "unknown"))
                function_name = str(chunk.get("name") or chunk.get("display_name") or chunk.get("function_name") or "unknown")
                start_line = chunk.get("start_line", "?")
                end_line = chunk.get("end_line", "?")
                content = str(chunk.get("content") or chunk.get("code_snippet") or "")
                hop_distance = chunk.get("hop_distance", 0)
                score = float(chunk.get("importance_score", chunk.get("score", 0.0)) or 0.0)

                lang = ""
                if "." in file_path:
                    lang = file_path.rsplit(".", 1)[-1]

                header = (
                    f"### File: {file_path} | Function: {function_name} | "
                    f"Lines: {start_line}-{end_line} | hop={hop_distance} | score={score:.2f}"
                )
                blocks.append(f"{header}\n```{lang}\n{content}\n```")

            parts.append("<code_context>\n" + "\n\n".join(blocks) + "\n</code_context>")

        parts.append(f"<question>\n{question.strip()}\n</question>")
        return "\n\n".join(part for part in parts if part)
