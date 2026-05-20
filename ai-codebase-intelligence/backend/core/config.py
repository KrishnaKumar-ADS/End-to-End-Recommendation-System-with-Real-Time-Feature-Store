from functools import lru_cache
import logging
from pydantic_settings import BaseSettings


class ConfigurationError(Exception):
    """Raised when a required configuration value is missing or invalid."""


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── LLM API Keys ─────────────────────────────────────
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    openrouter_api_key: str = ""
    qwen_api_key: str = ""

    # ── LLM Routing ──────────────────────────────────────
    llm_code_model: str = "deepseek-coder"
    llm_reasoning_model: str = "gemini-2.0-flash"
    llm_security_model: str = "deepseek-reasoner"
    llm_embedding_model: str = "models/gemini-embedding-001"
    embedding_vector_dim: int = 768
    embedding_allow_local_fallback: bool = True
    llm_fallback_model: str = "openrouter"

    # ── PostgreSQL ────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "codebase_intelligence"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # ── Redis ─────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    daily_budget_usd: float = 5.0

    # ── Qdrant ───────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ── Neo4j ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"

    # ── Webhook ──────────────────────────────────────────
    webhook_secret: str = ""

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def resolved_embedding_model(self) -> str:
        """
        Normalize legacy Gemini embedding model names to currently supported ones.
        """
        legacy_model_map = {
            "models/text-embedding-004": "models/gemini-embedding-001",
        }
        return legacy_model_map.get(self.llm_embedding_model, self.llm_embedding_model)

    def validate_api_keys(self) -> list[str]:
        """
        Returns a list of warning messages for missing API keys.
        Called at startup — missing keys disable that provider gracefully.
        """
        warnings = []
        if not self.gemini_api_key:
            warnings.append("GEMINI_API_KEY not set — embeddings and primary LLM will be unavailable.")
        if not self.deepseek_api_key:
            warnings.append("DEEPSEEK_API_KEY not set — code-specific reasoning will use Gemini fallback.")
        if not self.openrouter_api_key:
            warnings.append("OPENROUTER_API_KEY not set — fallback provider disabled.")
        if not self.qwen_api_key:
            warnings.append("QWEN_API_KEY not set — direct Qwen query expansion fallback disabled.")
        return warnings

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def validate_production_config(settings: "Settings") -> None:
    """
    Validate required configuration values for production deployment.

    Raises:
        ConfigurationError: if a required configuration value is missing.
    """
    required = {
        "gemini_api_key": "Gemini text-embedding-004 vector embeddings",
        "openrouter_api_key": "Qwen LLM via OpenRouter (qwen-2.5-coder-32b-instruct, qwen-max)",
        "postgres_host": "PostgreSQL metadata database",
        "postgres_db": "PostgreSQL database name",
        "postgres_user": "PostgreSQL authentication",
        "postgres_password": "PostgreSQL authentication",
        "redis_url": "Celery broker, session store, and caching layer",
        "qdrant_host": "Qdrant vector database for semantic search",
        "neo4j_uri": "Neo4j graph database for dependency graphs",
        "neo4j_user": "Neo4j authentication",
        "neo4j_password": "Neo4j authentication",
    }

    for field, description in required.items():
        value = getattr(settings, field, "")
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ConfigurationError(
                f"Required environment variable '{field.upper()}' is not set.\n"
                f"  This variable is required for: {description}\n"
                f"  Set it in your .env.prod file and run: make prod-up"
            )

    if not settings.deepseek_api_key:
        logging.getLogger(__name__).warning(
            "DEEPSEEK_API_KEY is not set - DeepSeek direct fallback unavailable. "
            "OpenRouter (Qwen) remains available as the primary provider."
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()