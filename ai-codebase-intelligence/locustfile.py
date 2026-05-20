"""Locust entrypoint for Week 15 load testing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.locust_tasks import CodebasePlatformUser  # noqa: E402,F401
