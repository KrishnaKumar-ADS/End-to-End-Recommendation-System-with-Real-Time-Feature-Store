"""Unit tests for production configuration validation."""

from unittest.mock import MagicMock

import pytest

from core.config import ConfigurationError, validate_production_config


def make_settings(**overrides):
    """Build a mock Settings object with all required fields populated."""
    defaults = {
        "gemini_api_key": "test-gemini-key",
        "openrouter_api_key": "sk-or-test-key",
        "deepseek_api_key": "test-deepseek-key",
        "postgres_host": "localhost",
        "postgres_db": "codebase_intelligence",
        "postgres_user": "postgres",
        "postgres_password": "postgres",
        "redis_url": "redis://localhost:6379/0",
        "qdrant_host": "localhost",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "neo4j_password",
    }
    defaults.update(overrides)

    settings = MagicMock()
    for key, value in defaults.items():
        setattr(settings, key, value)
    return settings


def test_validate_passes_with_all_fields():
    settings = make_settings()
    validate_production_config(settings)


def test_validate_raises_on_missing_gemini_key():
    settings = make_settings(gemini_api_key="")
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        validate_production_config(settings)


def test_validate_raises_on_missing_openrouter_key():
    settings = make_settings(openrouter_api_key="")
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        validate_production_config(settings)


def test_validate_raises_on_missing_postgres_host():
    settings = make_settings(postgres_host="")
    with pytest.raises(ConfigurationError, match="POSTGRES_HOST"):
        validate_production_config(settings)


def test_validate_raises_on_missing_redis_url():
    settings = make_settings(redis_url="")
    with pytest.raises(ConfigurationError, match="REDIS_URL"):
        validate_production_config(settings)


def test_validate_raises_on_missing_neo4j_uri():
    settings = make_settings(neo4j_uri="")
    with pytest.raises(ConfigurationError, match="NEO4J_URI"):
        validate_production_config(settings)


def test_validate_raises_on_missing_qdrant_host():
    settings = make_settings(qdrant_host="")
    with pytest.raises(ConfigurationError, match="QDRANT_HOST"):
        validate_production_config(settings)


def test_validate_error_message_includes_description():
    settings = make_settings(gemini_api_key="")
    with pytest.raises(ConfigurationError) as exc_info:
        validate_production_config(settings)

    assert "text-embedding-004" in str(exc_info.value)
    assert ".env.prod" in str(exc_info.value)


def test_validate_does_not_raise_on_missing_deepseek_key():
    settings = make_settings(deepseek_api_key="")
    validate_production_config(settings)


def test_validate_raises_on_whitespace_only_value():
    settings = make_settings(gemini_api_key="   ")
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        validate_production_config(settings)
