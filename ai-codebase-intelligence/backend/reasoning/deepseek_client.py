"""DeepSeek async client for code and reasoning tasks."""

from __future__ import annotations

from typing import AsyncIterator

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.exceptions import LLMProviderError
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


DEEPSEEK_CODER = "deepseek-coder"
DEEPSEEK_CHAT = "deepseek-chat"
DEEPSEEK_REASONER = "deepseek-reasoner"
DEFAULT_MODEL = DEEPSEEK_CODER

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096

CONTEXT_WINDOW_TOKENS = {
    DEEPSEEK_CODER: 128_000,
    DEEPSEEK_CHAT: 65_536,
    DEEPSEEK_REASONER: 65_536,
}


class DeepSeekClient:
    """Async DeepSeek chat-completions client with retry support."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if not settings.deepseek_api_key:
            raise LLMProviderError("DEEPSEEK_API_KEY is not set.")

        self._model = model
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=120.0,
            max_retries=0,
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        target_model = model or self._model
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = await self._complete_with_retry(
                messages=messages,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            return text
        except LLMProviderError:
            raise
        except Exception as exc:
            logger.error("deepseek_complete_failed", error=str(exc), model=target_model)
            raise LLMProviderError(f"DeepSeek generation failed: {exc}") from exc

    async def stream_complete(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> AsyncIterator[str]:
        target_model = model or self._model
        messages = self._build_messages(prompt, system_prompt)

        try:
            stream = await self._client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (APIStatusError, APIConnectionError, RateLimitError) as exc:
            raise LLMProviderError(f"DeepSeek streaming failed: {exc}") from exc
        except Exception as exc:
            raise LLMProviderError(f"DeepSeek streaming failed: {exc}") from exc

    def get_context_window(self, model: str | None = None) -> int:
        return CONTEXT_WINDOW_TOKENS.get(model or self._model, 65_536)

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((APIStatusError, APIConnectionError, RateLimitError)),
        reraise=True,
    )
    async def _complete_with_retry(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages