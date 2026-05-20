"""Synchronous SQLAlchemy session factory used by background workers."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.config import get_settings

settings = get_settings()

_sync_url = str(settings.database_url).replace("postgresql+asyncpg", "postgresql+psycopg2")
_sync_engine = create_engine(_sync_url, pool_pre_ping=True, pool_size=5, max_overflow=10)

SyncSessionLocal = sessionmaker(
    bind=_sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)