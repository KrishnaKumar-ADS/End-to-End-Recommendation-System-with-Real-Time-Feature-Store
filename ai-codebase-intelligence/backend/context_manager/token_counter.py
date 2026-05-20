"""Fast token estimation and model-specific context budgets."""

from __future__ import annotations

_CHARS_PER_TOKEN_TEXT = 4.0
_CHARS_PER_TOKEN_PYTHON = 3.5
_CHARS_PER_TOKEN_OTHER_CODE = 3.8

_EXTENSION_MAP: dict[str, float] = {
    ".py": _CHARS_PER_TOKEN_PYTHON,
    ".js": _CHARS_PER_TOKEN_OTHER_CODE,
    ".ts": _CHARS_PER_TOKEN_OTHER_CODE,
    ".go": _CHARS_PER_TOKEN_OTHER_CODE,
    ".java": _CHARS_PER_TOKEN_OTHER_CODE,
    ".rs": _CHARS_PER_TOKEN_OTHER_CODE,
    ".cpp": _CHARS_PER_TOKEN_OTHER_CODE,
    ".c": _CHARS_PER_TOKEN_OTHER_CODE,
    ".rb": _CHARS_PER_TOKEN_OTHER_CODE,
    ".php": _CHARS_PER_TOKEN_OTHER_CODE,
}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "qwen/qwen-2.5-coder-32b-instruct": 32_768,
    "qwen/qwen-max": 128_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 1_000_000,
    "deepseek-coder": 128_000,
    "deepseek-chat": 65_536,
    "deepseek-reasoner": 65_536,
    "openrouter/free": 65_536,
    "default": 8_192,
}

RESERVED_FOR_SYSTEM_PROMPT = 600
RESERVED_FOR_OUTPUT = 2_048
RESERVED_FOR_QUESTION = 256


def estimate_tokens(text: str, file_extension: str = "") -> int:
    """Estimate token count for text."""
    if not text:
        return 0

    chars_per_token = _EXTENSION_MAP.get(file_extension.lower(), _CHARS_PER_TOKEN_TEXT)
    estimated = int(len(text) / chars_per_token) + 1
    return max(estimated, 1)


def estimate_tokens_for_chunk(chunk: dict) -> int:
    """Estimate token count for one context chunk dict."""
    content = chunk.get("content") or chunk.get("code_snippet") or ""
    file_path = chunk.get("file_path") or ""

    ext = ""
    if "." in file_path:
        ext = "." + file_path.rsplit(".", 1)[-1]

    count = estimate_tokens(content, ext)
    return max(count, 1)


def get_context_budget(model: str, history_tokens: int = 0) -> int:
    """Return available token budget for code context."""
    window = MODEL_CONTEXT_WINDOWS.get(model, MODEL_CONTEXT_WINDOWS["default"])
    overhead = (
        RESERVED_FOR_SYSTEM_PROMPT
        + RESERVED_FOR_OUTPUT
        + RESERVED_FOR_QUESTION
        + max(history_tokens, 0)
    )

    budget = int((window - overhead) * 0.90)
    return max(budget, 1024)
