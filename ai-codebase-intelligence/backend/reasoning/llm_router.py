"""LLM router with task-aware provider selection and failover."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import AsyncIterator

from core.config import get_settings
from core.exceptions import LLMProviderError
from core.logging import get_logger
from reasoning.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)
settings = get_settings()
_circuit_breaker = CircuitBreaker()


class Provider(str, Enum):
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


class TaskType(str, Enum):
    CODE_QA = "code_qa"
    REASONING = "reasoning"
    SECURITY = "security"
    SUMMARIZE = "summarize"
    ARCHITECTURE = "architecture"


_ROUTING_TABLE: dict[TaskType, list[Provider]] = {
    TaskType.CODE_QA: [Provider.DEEPSEEK, Provider.GEMINI, Provider.OPENROUTER],
    TaskType.REASONING: [Provider.GEMINI, Provider.OPENROUTER, Provider.DEEPSEEK],
    TaskType.SECURITY: [Provider.DEEPSEEK, Provider.GEMINI, Provider.OPENROUTER],
    TaskType.SUMMARIZE: [Provider.GEMINI, Provider.DEEPSEEK, Provider.OPENROUTER],
    TaskType.ARCHITECTURE: [Provider.GEMINI, Provider.OPENROUTER, Provider.DEEPSEEK],
}

_REQUEST_TIMEOUT_SEC = 90.0
_STREAM_FIRST_TOKEN_TIMEOUT_SEC = 30.0
_STREAM_CHUNK_TIMEOUT_SEC = 60.0

_MODEL_TABLE: dict[TaskType, dict[Provider, str]] = {
    TaskType.CODE_QA: {
        Provider.DEEPSEEK: "deepseek-coder",
        Provider.GEMINI: "gemini-2.0-flash",
        Provider.OPENROUTER: "openrouter/free",
    },
    TaskType.REASONING: {
        Provider.GEMINI: "gemini-2.0-flash",
        Provider.DEEPSEEK: "deepseek-chat",
        Provider.OPENROUTER: "openrouter/free",
    },
    TaskType.SECURITY: {
        Provider.DEEPSEEK: "deepseek-reasoner",
        Provider.GEMINI: "gemini-2.0-flash",
        Provider.OPENROUTER: "openrouter/free",
    },
    TaskType.SUMMARIZE: {
        Provider.GEMINI: "gemini-2.0-flash",
        Provider.DEEPSEEK: "deepseek-chat",
        Provider.OPENROUTER: "openrouter/free",
    },
    TaskType.ARCHITECTURE: {
        Provider.GEMINI: "gemini-2.0-flash",
        Provider.DEEPSEEK: "deepseek-chat",
        Provider.OPENROUTER: "openrouter/free",
    },
}


def _get_client(provider: Provider, model: str):
    if provider == Provider.GEMINI:
        from reasoning.gemini_client import GeminiChatClient

        return GeminiChatClient(model=model)
    if provider == Provider.DEEPSEEK:
        from reasoning.deepseek_client import DeepSeekClient

        return DeepSeekClient(model=model)
    if provider == Provider.OPENROUTER:
        from reasoning.openrouter_client import OpenRouterClient

        return OpenRouterClient(model=model)
    raise ValueError(f"Unknown provider: {provider}")


def _is_provider_available(provider: Provider) -> bool:
    if provider == Provider.GEMINI:
        return bool(settings.gemini_api_key)
    if provider == Provider.DEEPSEEK:
        return bool(settings.deepseek_api_key)
    if provider == Provider.OPENROUTER:
        return bool(settings.openrouter_api_key)
    return False


def route(task_type: TaskType) -> Provider:
    """Return the preferred currently-available provider for a task type."""
    providers = _ROUTING_TABLE.get(task_type, [Provider.GEMINI])
    for provider in providers:
        if _is_provider_available(provider):
            return provider
    raise RuntimeError(
        f"No LLM provider available for task type '{task_type.value}'. "
        "Set GEMINI_API_KEY, DEEPSEEK_API_KEY, or OPENROUTER_API_KEY."
    )


def get_available_providers() -> list[Provider]:
    return [provider for provider in Provider if _is_provider_available(provider)]


async def ask(
    task_type: TaskType,
    prompt: str,
    system_prompt: str = "",
    temperature: float | None = None,
    max_tokens: int = 1200,
) -> tuple[str, Provider, str]:
    """Generate a response with automatic provider failover."""
    providers = _ROUTING_TABLE.get(task_type, [Provider.GEMINI])
    model_table = _MODEL_TABLE.get(task_type, {})
    last_error: Exception | None = None

    if temperature is None:
        temperature = _default_temperature(task_type)

    for provider in providers:
        if not _is_provider_available(provider):
            continue

        if not _circuit_breaker.is_available(provider.value):
            logger.info(
                "llm_router_provider_skipped_circuit_open",
                provider=provider.value,
                task_type=task_type.value,
            )
            continue

        model = model_table.get(provider, "")
        try:
            client = _get_client(provider, model)
            try:
                answer = await asyncio.wait_for(
                    client.complete(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=_REQUEST_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError as exc:
                raise LLMProviderError(
                    f"{provider.value} timed out after {_REQUEST_TIMEOUT_SEC:.0f}s"
                ) from exc
            _circuit_breaker.record_success(provider.value)
            return answer, provider, model
        except LLMProviderError as exc:
            _circuit_breaker.record_failure(provider.value)
            logger.warning(
                "llm_router_provider_failed",
                provider=provider.value,
                task_type=task_type.value,
                error=str(exc),
            )
            last_error = exc
            continue
        except Exception as exc:
            _circuit_breaker.record_failure(provider.value)
            logger.warning(
                "llm_router_provider_failed",
                provider=provider.value,
                task_type=task_type.value,
                error=str(exc),
            )
            last_error = exc
            continue

    raise LLMProviderError(
        f"All LLM providers failed for task type '{task_type.value}'. Last error: {last_error}"
    )


async def stream_ask(
    task_type: TaskType,
    prompt: str,
    system_prompt: str = "",
    temperature: float | None = None,
    max_tokens: int = 1200,
) -> AsyncIterator[tuple[str, Provider | None, str]]:
    """Stream response chunks, including provider/model metadata in first chunk."""
    providers = _ROUTING_TABLE.get(task_type, [Provider.GEMINI])
    model_table = _MODEL_TABLE.get(task_type, {})
    last_error: Exception | None = None

    if temperature is None:
        temperature = _default_temperature(task_type)

    for provider in providers:
        if not _is_provider_available(provider):
            continue

        if not _circuit_breaker.is_available(provider.value):
            logger.info(
                "llm_router_provider_skipped_circuit_open",
                provider=provider.value,
                task_type=task_type.value,
            )
            continue

        model = model_table.get(provider, "")
        try:
            client = _get_client(provider, model)
            stream_iter = client.stream_complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            try:
                first_chunk = await asyncio.wait_for(
                    anext(stream_iter),
                    timeout=_STREAM_FIRST_TOKEN_TIMEOUT_SEC,
                )
            except StopAsyncIteration as exc:
                raise LLMProviderError(
                    f"{provider.value} returned an empty stream"
                ) from exc
            except asyncio.TimeoutError as exc:
                raise LLMProviderError(
                    f"{provider.value} timed out waiting for first token "
                    f"after {_STREAM_FIRST_TOKEN_TIMEOUT_SEC:.0f}s"
                ) from exc

            yield first_chunk, provider, model

            while True:
                try:
                    chunk = await asyncio.wait_for(
                        anext(stream_iter),
                        timeout=_STREAM_CHUNK_TIMEOUT_SEC,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise LLMProviderError(
                        f"{provider.value} stream stalled for "
                        f"{_STREAM_CHUNK_TIMEOUT_SEC:.0f}s"
                    ) from exc

                yield chunk, None, ""

            _circuit_breaker.record_success(provider.value)
            return
        except LLMProviderError as exc:
            _circuit_breaker.record_failure(provider.value)
            logger.warning(
                "llm_router_stream_provider_failed",
                provider=provider.value,
                task_type=task_type.value,
                error=str(exc),
            )
            last_error = exc
            continue
        except Exception as exc:
            _circuit_breaker.record_failure(provider.value)
            logger.warning(
                "llm_router_stream_provider_failed",
                provider=provider.value,
                task_type=task_type.value,
                error=str(exc),
            )
            last_error = exc
            continue

    raise LLMProviderError(
        f"All streaming providers failed for task type '{task_type.value}'. Last error: {last_error}"
    )


def _default_temperature(task_type: TaskType) -> float:
    temperatures = {
        TaskType.CODE_QA: 0.1,
        TaskType.SECURITY: 0.1,
        TaskType.REASONING: 0.3,
        TaskType.SUMMARIZE: 0.2,
        TaskType.ARCHITECTURE: 0.3,
    }
    return temperatures.get(task_type, 0.2)