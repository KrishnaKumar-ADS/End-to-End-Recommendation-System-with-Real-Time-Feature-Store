"""Per-intent model and prompt selection."""

from __future__ import annotations

from intent.classifier import IntentType

INTENT_MODEL_MAP: dict[IntentType, str] = {
    "code_explanation": "qwen/qwen-2.5-coder-32b-instruct",
    "bug_trace": "qwen/qwen-2.5-coder-32b-instruct",
    "architecture": "qwen/qwen-max",
    "security": "qwen/qwen-max",
    "general": "qwen/qwen-max",
}

INTENT_TEMPERATURE_MAP: dict[IntentType, float] = {
    "code_explanation": 0.2,
    "bug_trace": 0.1,
    "architecture": 0.3,
    "security": 0.1,
    "general": 0.4,
}

_SYSTEM_PROMPTS: dict[IntentType, str] = {
    "code_explanation": (
        "You are an expert software engineer. "
        "Explain what the relevant code does, covering parameters, return values, and side effects. "
        "Cite concrete files and lines from provided context."
    ),
    "bug_trace": (
        "You are a debugging specialist. "
        "Trace likely root cause step-by-step, include call sequence, and propose concrete fixes. "
        "Do not invent unavailable evidence."
    ),
    "architecture": (
        "You are a principal architect. "
        "Describe module responsibilities and data flow from entry points to storage/services. "
        "Focus on relationships and boundaries."
    ),
    "security": (
        "You are a security engineer. "
        "Report only evidence-backed vulnerabilities with severity, impact, and remediation. "
        "Avoid speculation."
    ),
    "general": (
        "You are a helpful codebase assistant. "
        "Answer clearly using provided context and mention uncertainty when evidence is missing."
    ),
}


def get_system_prompt(intent: IntentType) -> str:
    return _SYSTEM_PROMPTS.get(intent, _SYSTEM_PROMPTS["general"])


def get_model_for_intent(intent: IntentType) -> str:
    return INTENT_MODEL_MAP.get(intent, INTENT_MODEL_MAP["general"])


def get_temperature_for_intent(intent: IntentType) -> float:
    return INTENT_TEMPERATURE_MAP.get(intent, INTENT_TEMPERATURE_MAP["general"])
