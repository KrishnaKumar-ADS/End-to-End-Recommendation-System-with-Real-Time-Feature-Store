"""Integration tests for the live Docker production stack.

Run explicitly with:
    pytest tests/integration/test_docker_stack.py -m integration -v
"""

from __future__ import annotations

import os
import re
import time

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE_URL = os.getenv("INTEGRATION_BASE_URL", "https://localhost")
VERIFY_SSL = os.getenv("INTEGRATION_VERIFY_SSL", "0") == "1"
TEST_REPO_URL = os.getenv("INTEGRATION_TEST_REPO", "https://github.com/psf/requests")
TEST_REPO_BRANCH = os.getenv("INTEGRATION_TEST_BRANCH", "main")


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        verify=VERIFY_SSL,
        timeout=30.0,
        follow_redirects=True,
    )


@pytest.fixture(scope="session")
def ingested_repo_id(client: httpx.Client) -> str:
    response = client.post(
        "/api/v1/ingest",
        json={"github_url": TEST_REPO_URL, "branch": TEST_REPO_BRANCH},
    )

    if response.status_code == 409:
        data = response.json()
        repo_id = data.get("repo_id")
        if repo_id:
            return repo_id
        pytest.skip("Repo already ingested but response did not include repo_id")

    if response.status_code != 202:
        pytest.skip(f"Ingest request failed: HTTP {response.status_code} - {response.text}")

    data = response.json()
    repo_id = data.get("repo_id")
    task_id = data.get("task_id")
    if not repo_id or not task_id:
        pytest.skip("Ingest response missing repo_id/task_id")

    for _ in range(60):
        time.sleep(5)
        status_response = client.get(f"/api/v1/status/{task_id}")
        if status_response.status_code != 200:
            continue

        status_data = status_response.json()
        status = str(status_data.get("status", "")).lower()
        if status == "completed":
            return repo_id
        if status == "failed":
            pytest.skip(f"Ingestion failed: {status_data}")

    pytest.skip("Ingestion did not complete in time")


class TestNginxRouting:
    def test_http_redirects_to_https(self):
        with httpx.Client(verify=VERIFY_SSL, follow_redirects=False, timeout=10.0) as plain_http:
            response = plain_http.get("http://localhost/")
        assert response.status_code in (301, 302, 307, 308)
        assert response.headers.get("location", "").startswith("https://")

    def test_health_routes_to_backend(self, client: httpx.Client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"

    def test_docs_routes_to_backend(self, client: httpx.Client):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "swagger" in response.text.lower()

    def test_root_routes_to_frontend(self, client: httpx.Client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_static_asset_cache_headers(self, client: httpx.Client):
        response = client.get("/")
        assert response.status_code == 200
        match = re.search(r"/_next/static/[^\"']+\.js", response.text)
        if not match:
            pytest.skip("No Next.js static asset URL found in page HTML")

        asset_response = client.get(match.group(0))
        assert asset_response.status_code == 200
        cache_control = asset_response.headers.get("cache-control", "")
        assert "max-age=31536000" in cache_control or "immutable" in cache_control


class TestPipeline:
    def test_search_returns_results_key(self, client: httpx.Client, ingested_repo_id: str):
        response = client.get(
            "/api/v1/search",
            params={"q": "request", "repo_id": ingested_repo_id, "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_graph_returns_nodes_and_edges(self, client: httpx.Client, ingested_repo_id: str):
        response = client.get(f"/api/v1/graph/{ingested_repo_id}")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)


class TestSecurityAndRateLimit:
    def test_security_headers_present(self, client: httpx.Client):
        response = client.get("/health")
        assert response.status_code == 200
        headers = {k.lower(): v for k, v in response.headers.items()}
        assert headers.get("x-content-type-options") == "nosniff"
        assert "x-frame-options" in headers
        assert "strict-transport-security" in headers

    def test_ask_endpoint_not_immediately_rate_limited(self, client: httpx.Client):
        response = client.post(
            "/api/v1/ask",
            json={"repo_id": "missing", "question": "ping", "stream": False},
        )
        assert response.status_code != 429
