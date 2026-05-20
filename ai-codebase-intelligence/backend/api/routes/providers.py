"""Provider operational health endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from core.config import get_settings
from reasoning.circuit_breaker import (
    FAILURE_THRESHOLD,
    RECOVERY_TIMEOUT_SEC,
    CircuitBreaker,
    CircuitState,
)

settings = get_settings()
router = APIRouter(prefix="/api/v1/providers", tags=["providers"])


@router.get("/health", summary="LLM provider health status")
async def get_providers_health() -> dict:
    cb = CircuitBreaker()
    statuses = cb.get_all_statuses()
    now = time.time()

    result: dict[str, dict] = {}
    for provider_name, status in statuses.items():
        entry = {
            "state": status.state.value,
            "consecutive_failures": status.failure_count,
            "api_key_configured": bool(
                (provider_name == "gemini" and settings.gemini_api_key)
                or (provider_name == "deepseek" and settings.deepseek_api_key)
                or (provider_name == "openrouter" and settings.openrouter_api_key)
            ),
        }

        if status.state == CircuitState.OPEN and status.last_failure_time > 0:
            elapsed = now - status.last_failure_time
            entry["recovery_in_seconds"] = round(max(0.0, RECOVERY_TIMEOUT_SEC - elapsed), 1)
            entry["last_failure_ago_seconds"] = round(elapsed, 1)

        if status.last_success_time > 0:
            entry["last_success_ago_seconds"] = round(now - status.last_success_time, 1)

        result[provider_name] = entry

    return {
        "providers": result,
        "circuit_breaker": {
            "failure_threshold": FAILURE_THRESHOLD,
            "recovery_timeout_sec": RECOVERY_TIMEOUT_SEC,
        },
    }


@router.post("/{provider_name}/reset", summary="Reset provider circuit breaker")
async def reset_provider_circuit(provider_name: str) -> dict:
    valid_providers = {"gemini", "deepseek", "openrouter"}
    if provider_name not in valid_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{provider_name}'. Valid: {sorted(valid_providers)}",
        )

    cb = CircuitBreaker()
    cb.reset(provider_name)

    return {
        "provider": provider_name,
        "circuit": "reset_to_closed",
        "message": f"Provider '{provider_name}' circuit breaker manually reset.",
    }
