"""Static pricing table and helpers for token-cost estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRate:
    input_per_1m: float
    output_per_1m: float


PROVIDER_RATES: dict[str, ModelRate] = {
    # OpenRouter examples
    "openrouter:qwen/qwen-max": ModelRate(input_per_1m=1.60, output_per_1m=6.40),
    "openrouter:qwen/qwen-2.5-coder-32b-instruct": ModelRate(input_per_1m=0.30, output_per_1m=1.20),
    "openrouter:openrouter/free": ModelRate(input_per_1m=0.0, output_per_1m=0.0),
    "openrouter:default": ModelRate(input_per_1m=0.80, output_per_1m=3.20),
    # DeepSeek examples
    "deepseek:deepseek-coder": ModelRate(input_per_1m=0.14, output_per_1m=0.28),
    "deepseek:deepseek-chat": ModelRate(input_per_1m=0.14, output_per_1m=0.28),
    "deepseek:deepseek-reasoner": ModelRate(input_per_1m=0.55, output_per_1m=2.19),
    "deepseek:default": ModelRate(input_per_1m=0.30, output_per_1m=0.60),
    # Gemini (chat examples)
    "gemini:gemini-2.0-flash": ModelRate(input_per_1m=0.0, output_per_1m=0.0),
    "gemini:gemini-1.5-pro": ModelRate(input_per_1m=1.25, output_per_1m=5.00),
    "gemini:text-embedding-004": ModelRate(input_per_1m=0.0, output_per_1m=0.0),
    "gemini:models/gemini-embedding-001": ModelRate(input_per_1m=0.0, output_per_1m=0.0),
    "gemini:default": ModelRate(input_per_1m=0.0, output_per_1m=0.0),
}


def get_rate(provider: str, model: str) -> ModelRate:
    normalized_provider = (provider or "").strip().lower()
    normalized_model = (model or "").strip().lower()

    exact_key = f"{normalized_provider}:{normalized_model}"
    if exact_key in PROVIDER_RATES:
        return PROVIDER_RATES[exact_key]

    default_key = f"{normalized_provider}:default"
    if default_key in PROVIDER_RATES:
        return PROVIDER_RATES[default_key]

    return PROVIDER_RATES["openrouter:default"]


def estimate_cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    safe_input = max(0, int(input_tokens))
    safe_output = max(0, int(output_tokens))

    normalized_provider = (provider or "").strip().lower()
    normalized_model = (model or "").strip().lower()

    if normalized_provider == "gemini" and "embedding" in normalized_model:
        return 0.0

    rate = get_rate(normalized_provider, normalized_model)
    input_cost = (safe_input / 1_000_000) * rate.input_per_1m
    output_cost = (safe_output / 1_000_000) * rate.output_per_1m
    return round(input_cost + output_cost, 8)
