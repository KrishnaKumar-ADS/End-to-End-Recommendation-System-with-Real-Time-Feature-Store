"""Unit tests for evaluation.latency_benchmark."""

from __future__ import annotations

from evaluation.latency_benchmark import _percentile, _stats


def test_percentile_basic_ordering() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50 = _percentile(values, 0.50)
    p95 = _percentile(values, 0.95)
    p99 = _percentile(values, 0.99)
    assert p99 >= p95 >= p50


def test_percentile_single_value() -> None:
    assert _percentile([42.0], 0.95) == 42.0


def test_stats_tracks_cache_hits_and_misses() -> None:
    stat = _stats(
        endpoint="GET /api/v1/search",
        method="GET",
        durations_ms=[10.0, 12.0, 8.0],
        sample_count=3,
        error_count=0,
        cache_hits=2,
        cache_misses=1,
    )
    assert stat.cache_hits == 2
    assert stat.cache_misses == 1
    assert stat.error_count == 0


def test_stats_handles_empty_samples() -> None:
    stat = _stats(
        endpoint="GET /health",
        method="GET",
        durations_ms=[],
        sample_count=0,
        error_count=0,
    )
    assert stat.p50_ms == 0.0
    assert stat.p95_ms == 0.0
    assert stat.p99_ms == 0.0


def test_stats_includes_ttft_when_present() -> None:
    stat = _stats(
        endpoint="POST /api/v1/ask (stream)",
        method="POST",
        durations_ms=[1000.0, 1200.0, 1400.0],
        sample_count=3,
        error_count=0,
        ttft_samples=[300.0, 350.0, 400.0],
    )
    assert stat.ttft_p50_ms is not None
    assert stat.ttft_p95_ms is not None
    assert stat.ttft_p99_ms is not None
