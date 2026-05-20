"""OpenRouter async client used as a fallback LLM provider."""

from __future__ import annotations

import time
from typing import AsyncIterator

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.exceptions import LLMProviderError
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/your-username/ai-codebase-intelligence",
    "X-Title": "AI Codebase Intelligence Platform",
}

DEFAULT_MODEL = "openrouter/free"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
MAX_FALLBACK_MODELS = 6


class OpenRouterClient:
    """Async OpenRouter client with retry on transient failures."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if not settings.openrouter_api_key:
            raise LLMProviderError("OPENROUTER_API_KEY is not set.")

        self._model = model
        self._free_models_cache: list[str] = []
        self._free_models_cache_at: float = 0.0
        self._free_models_ttl_sec: float = 300.0
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
            timeout=90.0,
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
            return response.choices[0].message.content or ""
        except (APIStatusError, APIConnectionError, RateLimitError) as exc:
            fallback_models = await self._get_fallback_models(target_model)
            last_error: Exception = exc
            for fallback_model in fallback_models:
                try:
                    logger.warning(
                        "openrouter_model_fallback_try",
                        from_model=target_model,
                        to_model=fallback_model,
                    )
                    response = await self._complete_with_retry(
                        messages=messages,
                        model=fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = response.choices[0].message.content or ""
                    if text:
                        logger.info(
                            "openrouter_model_fallback_success",
                            from_model=target_model,
                            to_model=fallback_model,
                        )
                        return text
                except (APIStatusError, APIConnectionError, RateLimitError) as fallback_exc:
                    last_error = fallback_exc
                    continue
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    continue

            raise LLMProviderError(f"OpenRouter generation failed: {last_error}") from last_error
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"OpenRouter generation failed: {exc}") from exc

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

        models_to_try = [target_model]
        attempted_fallbacks = False

        while models_to_try:
            current_model = models_to_try.pop(0)
            emitted_any = False
            try:
                stream = await self._client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        emitted_any = True
                        yield delta

                if current_model != target_model:
                    logger.info(
                        "openrouter_stream_model_fallback_success",
                        from_model=target_model,
                        to_model=current_model,
                    )
                return
            except (APIStatusError, APIConnectionError, RateLimitError) as exc:
                if emitted_any:
                    raise LLMProviderError(f"OpenRouter streaming failed: {exc}") from exc

                if current_model == target_model and not attempted_fallbacks:
                    fallback_models = await self._get_fallback_models(target_model)
                    if fallback_models:
                        attempted_fallbacks = True
                        models_to_try.extend(fallback_models)
                        logger.warning(
                            "openrouter_stream_model_fallback_try",
                            from_model=target_model,
                            candidates=fallback_models,
                        )
                        continue

                raise LLMProviderError(f"OpenRouter streaming failed: {exc}") from exc
            except Exception as exc:
                if emitted_any:
                    raise LLMProviderError(f"OpenRouter streaming failed: {exc}") from exc
                raise LLMProviderError(f"OpenRouter streaming failed: {exc}") from exc

        raise LLMProviderError("OpenRouter streaming failed: no models available.")

    async def list_free_models(self) -> list[str]:
        """Return currently free OpenRouter models, with safe fallback."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{OPENROUTER_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                )
                response.raise_for_status()
                models = response.json().get("data", [])
                free: list[str] = []
                for model in models:
                    model_id = str(model.get("id") or "").strip()
                    if not model_id:
                        continue

                    if model.get("pricing", {}).get("prompt", "1") != "0":
                        continue

                    # Keep only chat-safe text output models.
                    architecture = model.get("architecture", {}) or {}
                    output_modalities = architecture.get("output_modalities") or []
                    if isinstance(output_modalities, list) and output_modalities:
                        outputs = {str(item).lower() for item in output_modalities}
                        if "text" not in outputs or "audio" in outputs:
                            continue

                    # Avoid non-free aliases that may include non-chat preview models.
                    if model_id != DEFAULT_MODEL and not (
                        model_id.endswith(":free") or "free" in model_id.lower()
                    ):
                        continue

                    free.append(model_id)

                deduped_sorted = sorted(list(dict.fromkeys(free)))
                return deduped_sorted
        except Exception as exc:
            logger.warning("openrouter_list_models_failed", error=str(exc))
            return [DEFAULT_MODEL]

    @property
    def model_name(self) -> str:
        return self._model

    async def _get_fallback_models(self, target_model: str) -> list[str]:
        if target_model != DEFAULT_MODEL:
            return []

        free_models = await self._get_cached_free_models()
        candidates = [model for model in free_models if model and model != target_model]
        return candidates[:MAX_FALLBACK_MODELS]

    async def _get_cached_free_models(self) -> list[str]:
        now = time.time()
        if self._free_models_cache and (now - self._free_models_cache_at) < self._free_models_ttl_sec:
            return self._free_models_cache

        models = await self.list_free_models()
        self._free_models_cache = models or [DEFAULT_MODEL]
        self._free_models_cache_at = now
        return self._free_models_cache

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
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