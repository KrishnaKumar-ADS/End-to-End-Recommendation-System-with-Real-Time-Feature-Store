import sys
from pathlib import Path

from celery import Celery

# Ensure backend root is importable even when worker starts from another CWD.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "codebase_intelligence",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["tasks.ingest_task", "tasks.embed_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    broker_connection_retry_on_startup=True,  # ← fixes the warning
)