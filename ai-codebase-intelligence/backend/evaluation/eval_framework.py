"""Week 15 end-to-end evaluation orchestration framework."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from evaluation.cost_tracker import CostTracker
from evaluation.repos import EvalQuestion, EvalRepo, EVAL_REPOS, get_repo_by_name


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class EvalConfig:
    repos_to_eval: list[str] | None = None
    questions_per_repo: int = 20
    providers_to_compare: list[str] = field(default_factory=lambda: ["qwen-coder", "qwen-max"])
    timeout_per_question: int = 120
    skip_already_ingested: bool = True
    base_url: str = "https://localhost"
    skip_tls_verify: bool = True
    ingestion_poll_interval: int = 5
    max_ingestion_wait_seconds: int = 1200
    output_dir: Path = field(default_factory=lambda: _project_root() / "data" / "benchmarks" / "results")


@dataclass(slots=True)
class QuestionRunResult:
    question_id: str
    repo_name: str
    question: str
    category: str
    provider_used: str
    model_used: str
    answer: str
    sources: list[dict[str, Any]]
    graph_path: list[str]
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    timestamp: str = field(default_factory=_utc_now)
    error: str | None = None


@dataclass(slots=True)
class RepoRunResult:
    repo_name: str
    github_url: str
    branch: str
    language: str
    repo_id: str | None
    ingestion_task_id: str | None
    ingestion_time_seconds: float
    questions: list[QuestionRunResult] = field(default_factory=list)


@dataclass(slots=True)
class EvalRunResult:
    run_id: str
    started_at: str
    finished_at: str | None
    config: dict[str, Any]
    repos: list[RepoRunResult]


class EvaluationRunner:
    """Coordinates ingestion, question execution, and result persistence."""

    def __init__(
        self,
        config: EvalConfig,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self.config = config
        self.client = httpx.Client(
            base_url=config.base_url,
            verify=not config.skip_tls_verify,
            timeout=config.timeout_per_question,
            follow_redirects=True,
        )
        self.cost_tracker = cost_tracker or CostTracker()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "EvaluationRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _selected_repos(self) -> list[EvalRepo]:
        if not self.config.repos_to_eval:
            return list(EVAL_REPOS)

        selected: list[EvalRepo] = []
        for repo_name in self.config.repos_to_eval:
            repo = get_repo_by_name(repo_name)
            if repo is not None:
                selected.append(repo)
        return selected

    def _question_subset(self, repo: EvalRepo) -> list[EvalQuestion]:
        if self.config.questions_per_repo <= 0:
            return []
        return repo.questions[: self.config.questions_per_repo]

    def _list_existing_repositories(self) -> list[dict[str, Any]]:
        response = self.client.get("/api/v1/repos")
        if response.status_code != 200:
            return []
        payload = response.json()
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("repos"), list):
            return payload["repos"]
        return []

    def _find_existing_repo_id(self, github_url: str) -> str | None:
        normalized = github_url.rstrip("/")
        for repo in self._list_existing_repositories():
            if str(repo.get("github_url", "")).rstrip("/") == normalized:
                status = str(repo.get("status", "")).lower()
                if status == "completed":
                    return str(repo.get("id") or repo.get("repo_id") or "") or None
        return None

    def _poll_status(self, task_id: str) -> tuple[bool, str | None]:
        deadline = time.time() + self.config.max_ingestion_wait_seconds
        while time.time() < deadline:
            response = self.client.get(f"/api/v1/status/{task_id}")
            if response.status_code != 200:
                time.sleep(self.config.ingestion_poll_interval)
                continue

            payload = response.json()
            status_value = str(payload.get("status", "")).lower()
            repo_id = payload.get("repo_id")

            if status_value == "completed":
                return True, str(repo_id) if repo_id else None
            if status_value == "failed":
                return False, str(repo_id) if repo_id else None

            time.sleep(self.config.ingestion_poll_interval)

        return False, None

    def ensure_repo_ingested(self, repo: EvalRepo) -> tuple[str | None, str | None, float, str | None]:
        """Ensure repository is indexed and return repo_id/task_id/ingestion_time/error."""
        if self.config.skip_already_ingested:
            existing_id = self._find_existing_repo_id(repo.github_url)
            if existing_id:
                return existing_id, None, 0.0, None

        started = time.perf_counter()
        response = self.client.post(
            "/api/v1/ingest",
            json={"github_url": repo.github_url, "branch": repo.branch},
        )

        if response.status_code == 409:
            existing_id = self._find_existing_repo_id(repo.github_url)
            if existing_id:
                return existing_id, None, 0.0, None
            return None, None, 0.0, "Repository already exists but could not resolve repo_id."

        if response.status_code != 202:
            return None, None, 0.0, f"Ingest failed with HTTP {response.status_code}: {response.text[:200]}"

        payload = response.json()
        task_id = str(payload.get("task_id", "")) or None
        repo_id = str(payload.get("repo_id", "")) or None
        if not task_id:
            return repo_id, None, 0.0, "Ingest response missing task_id."

        completed, polled_repo_id = self._poll_status(task_id)
        elapsed = round(time.perf_counter() - started, 3)
        resolved_repo_id = polled_repo_id or repo_id

        if not completed:
            return resolved_repo_id, task_id, elapsed, "Ingestion did not complete successfully."

        return resolved_repo_id, task_id, elapsed, None

    def _task_type_for_category(self, category: str) -> str:
        if category == "architecture":
            return "reasoning"
        if category == "security":
            return "security"
        if category == "code_explanation":
            return "code_qa"
        if category == "bug_trace":
            return "code_qa"
        return "reasoning"

    def _ask_once(
        self,
        repo_id: str,
        repo_name: str,
        question_index: int,
        question: EvalQuestion,
    ) -> QuestionRunResult:
        question_id = f"{repo_name.replace('/', '_')}-q{question_index + 1:02d}"
        started = time.perf_counter()

        try:
            response = self.client.post(
                "/api/v1/ask",
                json={
                    "repo_id": repo_id,
                    "question": question.text,
                    "stream": False,
                    "task_type": self._task_type_for_category(question.category),
                },
                timeout=self.config.timeout_per_question,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return QuestionRunResult(
                question_id=question_id,
                repo_name=repo_name,
                question=question.text,
                category=question.category,
                provider_used="",
                model_used="",
                answer="",
                sources=[],
                graph_path=[],
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_cost_usd=0.0,
                error=str(exc),
            )

        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code != 200:
            return QuestionRunResult(
                question_id=question_id,
                repo_name=repo_name,
                question=question.text,
                category=question.category,
                provider_used="",
                model_used="",
                answer="",
                sources=[],
                graph_path=[],
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_cost_usd=0.0,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        payload = response.json()
        answer = str(payload.get("answer", ""))
        provider_used = str(payload.get("provider_used", "unknown"))
        model_used = str(payload.get("model_used", "unknown"))

        prompt_tokens = self.cost_tracker.count_tokens(question.text)
        completion_tokens = self.cost_tracker.count_tokens(answer)

        try:
            cost_record = self.cost_tracker.record_call(
                repo_name=repo_name,
                question_id=question_id,
                provider=provider_used,
                model=model_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            cost_usd = cost_record.cost_usd
        except Exception as exc:
            cost_usd = 0.0
            return QuestionRunResult(
                question_id=question_id,
                repo_name=repo_name,
                question=question.text,
                category=question.category,
                provider_used=provider_used,
                model_used=model_used,
                answer=answer,
                sources=list(payload.get("sources") or []),
                graph_path=list(payload.get("graph_path") or []),
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_cost_usd=cost_usd,
                error=str(exc),
            )

        return QuestionRunResult(
            question_id=question_id,
            repo_name=repo_name,
            question=question.text,
            category=question.category,
            provider_used=provider_used,
            model_used=model_used,
            answer=answer,
            sources=list(payload.get("sources") or []),
            graph_path=list(payload.get("graph_path") or []),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=cost_usd,
        )

    def run_repo(self, repo: EvalRepo) -> RepoRunResult:
        repo_id, task_id, ingest_seconds, ingest_error = self.ensure_repo_ingested(repo)
        result = RepoRunResult(
            repo_name=repo.name,
            github_url=repo.github_url,
            branch=repo.branch,
            language=repo.language,
            repo_id=repo_id,
            ingestion_task_id=task_id,
            ingestion_time_seconds=ingest_seconds,
            questions=[],
        )

        if ingest_error:
            result.questions.append(
                QuestionRunResult(
                    question_id=f"{repo.name.replace('/', '_')}-ingest",
                    repo_name=repo.name,
                    question="<ingestion>",
                    category="ingestion",
                    provider_used="",
                    model_used="",
                    answer="",
                    sources=[],
                    graph_path=[],
                    latency_ms=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_cost_usd=0.0,
                    error=ingest_error,
                )
            )
            return result

        if not repo_id:
            result.questions.append(
                QuestionRunResult(
                    question_id=f"{repo.name.replace('/', '_')}-repoid",
                    repo_name=repo.name,
                    question="<repo_id>",
                    category="ingestion",
                    provider_used="",
                    model_used="",
                    answer="",
                    sources=[],
                    graph_path=[],
                    latency_ms=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_cost_usd=0.0,
                    error="Repository ID unavailable after ingestion.",
                )
            )
            return result

        questions = self._question_subset(repo)
        for idx, question in enumerate(questions):
            result.questions.append(self._ask_once(repo_id, repo.name, idx, question))
        return result

    def _save_repo_result(self, repo_result: RepoRunResult) -> Path:
        repo_dir = self.config.output_dir / repo_result.repo_name.replace("/", "_")
        repo_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = repo_dir / f"{timestamp}.json"
        output_path.write_text(json.dumps(asdict(repo_result), indent=2), encoding="utf-8")
        return output_path

    def _serialize_config(self) -> dict[str, Any]:
        return {
            "repos_to_eval": self.config.repos_to_eval,
            "questions_per_repo": self.config.questions_per_repo,
            "providers_to_compare": self.config.providers_to_compare,
            "timeout_per_question": self.config.timeout_per_question,
            "skip_already_ingested": self.config.skip_already_ingested,
            "base_url": self.config.base_url,
            "skip_tls_verify": self.config.skip_tls_verify,
            "ingestion_poll_interval": self.config.ingestion_poll_interval,
            "max_ingestion_wait_seconds": self.config.max_ingestion_wait_seconds,
            "output_dir": str(self.config.output_dir),
        }

    def run(self) -> EvalRunResult:
        run_result = EvalRunResult(
            run_id=str(uuid.uuid4())[:8],
            started_at=_utc_now(),
            finished_at=None,
            config=self._serialize_config(),
            repos=[],
        )

        for repo in self._selected_repos():
            repo_result = self.run_repo(repo)
            run_result.repos.append(repo_result)
            self._save_repo_result(repo_result)

        run_result.finished_at = _utc_now()

        summary_path = self.config.output_dir / f"eval_summary_{run_result.run_id}.json"
        summary_path.write_text(json.dumps(asdict(run_result), indent=2), encoding="utf-8")

        return run_result
