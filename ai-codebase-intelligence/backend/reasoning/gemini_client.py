"""Gemini generation client used by the Week 6 reasoning pipeline."""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator

import google.generativeai as genai
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.exceptions import LLMProviderError
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_PRO = "gemini-1.5-pro"
DEFAULT_MODEL = GEMINI_FLASH

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096


class GeminiChatClient:
    """Async-friendly wrapper around the synchronous Gemini SDK."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if not settings.gemini_api_key:
            raise LLMProviderError("GEMINI_API_KEY is not set.")

        genai.configure(api_key=settings.gemini_api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate a complete response as a single string."""
        try:
            return await asyncio.to_thread(
                self._complete_with_retry,
                prompt,
                system_prompt,
                temperature,
                max_tokens,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            logger.error("gemini_complete_failed", error=str(exc), model=self._model_name)
            raise LLMProviderError(f"Gemini generation failed: {exc}") from exc

    async def stream_complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> AsyncIterator[str]:
        """Yield incremental response chunks from Gemini."""
        queue: asyncio.Queue = asyncio.Queue()
        done = object()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            try:
                contents = self._build_contents(prompt, system_prompt)
                config = self._build_generation_config(temperature, max_tokens)

                stream = self._model.generate_content(
                    contents=contents,
                    generation_config=config,
                    stream=True,
                )
                for chunk in stream:
                    text = getattr(chunk, "text", None)
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, LLMProviderError(str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, Exception):
                raise LLMProviderError(f"Gemini streaming failed: {item}")
            yield item

    def estimate_tokens(self, text: str) -> int:
        """Best-effort token estimate for context budgeting."""
        try:
            import tiktoken

            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            return len(text) // 4

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def context_window_tokens(self) -> int:
        limits = {
            GEMINI_FLASH: 1_048_576,
            GEMINI_PRO: 2_097_152,
        }
        return limits.get(self._model_name, 1_048_576)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(lambda exc: not isinstance(exc, LLMProviderError)),
    )
    def _complete_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        contents = self._build_contents(prompt, system_prompt)
        config = self._build_generation_config(temperature, max_tokens)
        response = self._model.generate_content(
            contents=contents,
            generation_config=config,
        )

        text = getattr(response, "text", "")
        if not text:
            raise LLMProviderError("Gemini returned an empty response.")
        return text

    @staticmethod
    def _build_contents(prompt: str, system_prompt: str) -> list[dict]:
        if system_prompt:
            return [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood."}]},
                {"role": "user", "parts": [{"text": prompt}]},
            ]
        return [{"role": "user", "parts": [{"text": prompt}]}]

    @staticmethod
    def _build_generation_config(
        temperature: float,
        max_tokens: int,
    ) -> genai.types.GenerationConfig:
        return genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            candidate_count=1,
        )