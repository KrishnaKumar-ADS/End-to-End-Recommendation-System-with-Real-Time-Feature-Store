"""Week 15 Locust load test tasks for production-stack benchmarking."""

from __future__ import annotations

import os
import random
import time

from locust import HttpUser, between, task


DEFAULT_REPO_URL = os.getenv("EVAL_LOCUST_REPO", "https://github.com/psf/requests")
DEFAULT_REPO_BRANCH = os.getenv("EVAL_LOCUST_BRANCH", "main")


class CodebasePlatformUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self) -> None:
        self.repo_id = os.getenv("EVAL_REPO_ID", "").strip()
        self.task_id = os.getenv("EVAL_TASK_ID", "").strip()

        if self.repo_id:
            return

        payload = {
            "github_url": DEFAULT_REPO_URL,
            "branch": DEFAULT_REPO_BRANCH,
        }
        response = self.client.post(
            "/api/v1/ingest",
            json=payload,
            name="POST /api/v1/ingest",
            verify=False,
        )

        if response.status_code not in {202, 409}:
            return

        try:
            body = response.json()
        except Exception:
            body = {}

        self.repo_id = str(body.get("repo_id", "")).strip()
        self.task_id = str(body.get("task_id", "")).strip()

        if self.task_id and not self.repo_id:
            self._resolve_repo_id_from_status(self.task_id)

    def _resolve_repo_id_from_status(self, task_id: str) -> None:
        deadline = time.time() + 120
        while time.time() < deadline:
            response = self.client.get(
                f"/api/v1/status/{task_id}",
                name="GET /api/v1/status/{task_id}",
                verify=False,
            )
            if response.status_code != 200:
                time.sleep(2)
                continue
            payload = response.json()
            repo_id = payload.get("repo_id")
            if repo_id:
                self.repo_id = str(repo_id)
                return
            time.sleep(2)

    @task(5)
    def search_task(self) -> None:
        if not self.repo_id:
            return
        query = random.choice(
            [
                "authentication",
                "database",
                "middleware",
                "session",
                "error handling",
                "rate limiting",
            ]
        )
        self.client.get(
            "/api/v1/search",
            params={"q": query, "repo_id": self.repo_id, "top_k": 5},
            name="GET /api/v1/search",
            verify=False,
        )

    @task(3)
    def ask_task(self) -> None:
        if not self.repo_id:
            return
        question = random.choice(
            [
                "Summarize the architecture.",
                "Where is auth implemented?",
                "Explain error handling.",
                "Identify security risks.",
            ]
        )
        self.client.post(
            "/api/v1/ask",
            json={"repo_id": self.repo_id, "question": question, "stream": False},
            name="POST /api/v1/ask",
            verify=False,
        )

    @task(1)
    def graph_task(self) -> None:
        if not self.repo_id:
            return
        self.client.get(
            f"/api/v1/graph/{self.repo_id}",
            name="GET /api/v1/graph/{repo_id}",
            verify=False,
        )

    @task(1)
    def status_task(self) -> None:
        if not self.task_id:
            return
        self.client.get(
            f"/api/v1/status/{self.task_id}",
            name="GET /api/v1/status/{task_id}",
            verify=False,
        )
