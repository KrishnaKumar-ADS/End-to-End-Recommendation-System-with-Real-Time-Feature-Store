"""Conversation history prompt block builder."""

from __future__ import annotations


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _format_single_turn(turn: dict) -> str:
    role = (turn.get("role") or "unknown").strip()
    timestamp = (turn.get("timestamp") or "").strip()
    content = (turn.get("content") or "").strip()

    attrs = [f'role="{role}"']
    if timestamp:
        attrs.append(f'timestamp="{timestamp}"')

    lines = [f"<turn {' '.join(attrs)}>", content]

    if role == "assistant":
        sources = turn.get("sources") or []
        if sources:
            src_bits: list[str] = []
            for source in sources[:5]:
                file_name = source.get("file") or source.get("file_path") or ""
                function_name = source.get("function") or source.get("function_name") or ""
                if file_name and function_name:
                    src_bits.append(f"{file_name}:{function_name}")
                elif file_name:
                    src_bits.append(file_name)
            if src_bits:
                lines.append(f"[Sources: {', '.join(src_bits)}]")

    lines.append("</turn>")
    return "\n".join(lines)


def build_history_block(turns: list[dict], max_tokens: int = 1200) -> str:
    """Build XML-like conversation history, keeping the most recent turns under budget."""
    if not turns:
        return ""

    selected: list[dict] = []
    used_tokens = 0

    for turn in reversed(turns):
        formatted = _format_single_turn(turn)
        turn_tokens = _estimate_tokens(formatted)
        if selected and used_tokens + turn_tokens > max_tokens:
            break
        selected.append(turn)
        used_tokens += turn_tokens

    selected.reverse()

    blocks = [_format_single_turn(turn) for turn in selected]
    return "<conversation_history>\n" + "\n\n".join(blocks) + "\n</conversation_history>"
