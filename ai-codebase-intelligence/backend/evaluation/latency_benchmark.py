"""Week 15 latency benchmark: p50/p95/p99 profiling for public API endpoints."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class EndpointStats:
    endpoint: str
    method: str
    sample_count: int
    success_count: int
    error_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    std_dev_ms: float
    cache_hits: int = 0
    cache_misses: int = 0
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None


@dataclass(slots=True)
class LatencyRunResult:
    run_id: str
    created_at: str
    base_url: str
    samples_per_endpoint: int
    stats: list[EndpointStats] = field(default_factory=list)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _stats(
    endpoint: str,
    method: str,
    durations_ms: list[float],
    sample_count: int,
    error_count: int,
    cache_hits: int = 0,
    cache_misses: int = 0,
    ttft_samples: list[float] | None = None,
) -> EndpointStats:
    safe = durations_ms or [0.0]
    ttft = ttft_samples or []
    success_count = max(0, sample_count - error_count)

    return EndpointStats(
        endpoint=endpoint,
        method=method,
        sample_count=sample_count,
        success_count=success_count,
        error_count=error_count,
        p50_ms=round(_percentile(safe, 0.50), 3),
        p95_ms=round(_percentile(safe, 0.95), 3),
        p99_ms=round(_percentile(safe, 0.99), 3),
        min_ms=round(min(safe), 3),
        max_ms=round(max(safe), 3),
        mean_ms=round(statistics.mean(safe), 3),
        std_dev_ms=round(statistics.pstdev(safe), 3),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        ttft_p50_ms=round(_percentile(ttft, 0.50), 3) if ttft else None,
        ttft_p95_ms=round(_percentile(ttft, 0.95), 3) if ttft else None,
        ttft_p99_ms=round(_percentile(ttft, 0.99), 3) if ttft else None,
    )


class LatencyBenchmark:
    """Benchmark ingest/status/search/ask/graph/health endpoint latency."""

    def __init__(
        self,
        base_url: str = "https://localhost",
        samples_per_endpoint: int = 50,
        skip_tls_verify: bool = True,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url
        self.samples_per_endpoint = int(samples_per_endpoint)
        self.timeout_seconds = int(timeout_seconds)
        self.client = httpx.Client(
            base_url=base_url,
            verify=not skip_tls_verify,
            timeout=timeout_seconds,
            follow_redirects=True,
        )

        self.output_dir = _project_root() / "data" / "benchmarks" / "latency"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "LatencyBenchmark":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _bench_simple(
        self,
        endpoint: str,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
        success_statuses: set[int] | None = None,
        detect_cache: bool = False,
    ) -> EndpointStats:
        success = success_statuses or {200}
        durations: list[float] = []
        errors = 0
        cache_hits = 0
        cache_misses = 0

        for _ in range(self.samples_per_endpoint):
            start = time.perf_counter()
            try:
                response = self.client.request(
                    method=method,
                    url=path,
                    params=params,
                    json=json_payload,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                if response.status_code in success:
                    durations.append(elapsed_ms)
                    if detect_cache:
                        try:
                            payload = response.json()
                            is_cached = bool(payload.get("cached"))
                        except Exception:
                            is_cached = elapsed_ms < 60
                        if is_cached:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

        return _stats(
            endpoint=endpoint,
            method=method,
            durations_ms=durations,
            sample_count=self.samples_per_endpoint,
            error_count=errors,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def _bench_streaming_ask(self, repo_id: str, question: str) -> EndpointStats:
        n = max(3, self.samples_per_endpoint // 10)
        durations: list[float] = []
        ttft_samples: list[float] = []
        errors = 0

        for _ in range(n):
            start = time.perf_counter()
            first_token_at = None
            try:
                with self.client.stream(
                    "POST",
                    "/api/v1/ask",
                    json={
                        "repo_id": repo_id,
                        "question": question,
                        "stream": True,
                        "task_type": "code_qa",
                    },
                    timeout=max(60, self.timeout_seconds),
                ) as response:
                    if response.status_code != 200:
                        errors += 1
                        continue

                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_text = line.decode("utf-8", errors="ignore") if isinstance(line, bytes) else str(line)
                        if first_token_at is None and line_text.startswith("data:"):
                            payload = line_text[5:].strip()
                            if payload and payload != "[DONE]":
                                first_token_at = time.perf_counter()
                                ttft_samples.append((first_token_at - start) * 1000)
                        if line_text.strip() == "data: [DONE]":
                            durations.append((time.perf_counter() - start) * 1000)
                            break
            except Exception:
                errors += 1

        return _stats(
            endpoint="POST /api/v1/ask (stream)",
            method="POST",
            durations_ms=durations,
            sample_count=n,
            error_count=errors,
            ttft_samples=ttft_samples,
        )

    def run(
        self,
        repo_id: str,
        task_id: str,
        ingest_payload: dict[str, Any],
        ask_question: str = "Explain the repository architecture.",
    ) -> LatencyRunResult:
        run = LatencyRunResult(
            run_id=f"latency_{int(time.time())}",
            created_at=datetime.now(timezone.utc).isoformat(),
            base_url=self.base_url,
            samples_per_endpoint=self.samples_per_endpoint,
            stats=[],
        )

        run.stats.append(
            self._bench_simple(
                endpoint="POST /api/v1/ingest",
                method="POST",
                path="/api/v1/ingest",
                json_payload=ingest_payload,
                success_statuses={202, 409},
            )
        )
        run.stats.append(
            self._bench_simple(
                endpoint="GET /api/v1/status/{task_id}",
                method="GET",
                path=f"/api/v1/status/{task_id}",
                success_statuses={200},
            )
        )
        run.stats.append(
            self._bench_simple(
                endpoint="GET /api/v1/search",
                method="GET",
                path="/api/v1/search",
                params={"repo_id": repo_id, "q": "authentication", "top_k": 5},
                success_statuses={200},
                detect_cache=True,
            )
        )
        run.stats.append(
            self._bench_simple(
                endpoint="POST /api/v1/ask (no stream)",
                method="POST",
                path="/api/v1/ask",
                json_payload={
                    "repo_id": repo_id,
                    "question": ask_question,
                    "stream": False,
                    "task_type": "reasoning",
                },
                success_statuses={200},
                detect_cache=True,
            )
        )
        run.stats.append(self._bench_streaming_ask(repo_id=repo_id, question=ask_question))
        run.stats.append(
            self._bench_simple(
                endpoint="GET /api/v1/graph/{repo_id}",
                method="GET",
                path=f"/api/v1/graph/{repo_id}",
                success_statuses={200},
            )
        )
        run.stats.append(
            self._bench_simple(
                endpoint="GET /health",
                method="GET",
                path="/health",
                success_statuses={200},
            )
        )

        self.save(run)
        self.save_markdown_report(run)
        return run

    def save(self, result: LatencyRunResult) -> Path:
        output_path = self.output_dir / "latency_results.json"
        output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return output_path

    def save_markdown_report(self, result: LatencyRunResult) -> Path:
        rows = [
            "# Latency Report",
            "",
            f"Generated: {result.created_at}",
            "",
            "| Endpoint | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Errors |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        for stat in result.stats:
            rows.append(
                f"| {stat.endpoint} | {stat.p50_ms:.1f} | {stat.p95_ms:.1f} | {stat.p99_ms:.1f} | {stat.mean_ms:.1f} | {stat.error_count}/{stat.sample_count} |"
            )
            if stat.ttft_p50_ms is not None:
                rows.append(
                    f"| TTFT ({stat.endpoint}) | {stat.ttft_p50_ms:.1f} | {stat.ttft_p95_ms:.1f} | {stat.ttft_p99_ms:.1f} | - | - |"
                )

        report_path = self.output_dir / "latency_report.md"
        report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return report_path
