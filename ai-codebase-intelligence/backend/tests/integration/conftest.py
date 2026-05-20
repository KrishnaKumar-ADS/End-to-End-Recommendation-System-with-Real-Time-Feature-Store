import os

import httpx
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring a live Docker production stack",
    )


def pytest_collection_modifyitems(config, items):
    base_url = os.getenv("INTEGRATION_BASE_URL", "https://localhost")
    verify_ssl = os.getenv("INTEGRATION_VERIFY_SSL", "0") == "1"

    stack_reachable = False
    try:
        response = httpx.get(f"{base_url}/health", verify=verify_ssl, timeout=5.0)
        stack_reachable = response.status_code == 200
    except Exception:
        stack_reachable = False

    if stack_reachable:
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "Production Docker stack is not reachable. "
            "Run make prod-up before running integration tests."
        )
    )
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_marker)
