"""Qwen-powered query expansion with OpenRouter or direct Qwen fallback."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI

from core.config import get_settings

settings = get_settings()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen-max"
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-max"

_SYSTEM_PROMPT = """You expand software code-search queries.
Return ONLY valid JSON array of short strings.
No markdown or explanation.
"""


def _build_client() -> tuple[AsyncOpenAI | None, str]:
    if settings.openrouter_api_key:
        return (
            AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "https://github.com/ai-codebase-intelligence",
                    "X-Title": "AI Codebase Intelligence",
                },
                timeout=30.0,
            ),
            OPENROUTER_MODEL,
        )

    if settings.qwen_api_key:
        return (
            AsyncOpenAI(
                api_key=settings.qwen_api_key,
                base_url=QWEN_BASE_URL,
                timeout=30.0,
            ),
            QWEN_MODEL,
        )

    return None, ""


async def expand_query(query: str, n_expansions: int = 2) -> list[str]:
    client, model = _build_client()
    if client is None:
        return [query]

    prompt = (
        f"Generate {n_expansions + 1} code-search phrasings for: {query}. "
        "Return JSON array only."
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=180,
            temperature=0.3,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return [query]

        out: list[str] = [query]
        for item in parsed:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
            if len(out) >= n_expansions + 1:
                break
        return out
    except Exception:
        return [query]


def expand_query_sync(query: str, n_expansions: int = 2) -> list[str]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, expand_query(query, n_expansions))
                return future.result(timeout=15)
        return loop.run_until_complete(expand_query(query, n_expansions))
    except Exception:
        return [query]
