"""Unit tests for evaluation.report_generator."""

from __future__ import annotations

import json

from evaluation.report_generator import ReportGenerator, ReportPaths


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_markdown_contains_required_sections(tmp_path) -> None:
    generator = ReportGenerator(output_dir=tmp_path)

    eval_summary = tmp_path / "results" / "eval_summary_x.json"
    latency = tmp_path / "latency" / "latency_results.json"
    quality = tmp_path / "quality" / "quality_scores.json"
    cost = tmp_path / "costs" / "cost_report.json"

    _write_json(eval_summary, {"repos": []})
    _write_json(latency, {"stats": []})
    _write_json(quality, {"scores": [], "avg_composite": 0.0})
    _write_json(cost, {"records": [], "total_cost_usd": 0.0})

    markdown = generator.generate_markdown(
        ReportPaths(
            eval_summary=eval_summary,
            latency_results=latency,
            quality_scores=quality,
            cost_report=cost,
            load_csv=None,
        )
    )

    assert "## Executive Summary" in markdown
    assert "## Latency Benchmarks" in markdown
    assert "## Cost Breakdown" in markdown


def test_markdown_table_row_for_latency_stat(tmp_path) -> None:
    generator = ReportGenerator(output_dir=tmp_path)

    eval_summary = tmp_path / "results" / "eval_summary_x.json"
    latency = tmp_path / "latency" / "latency_results.json"
    quality = tmp_path / "quality" / "quality_scores.json"
    cost = tmp_path / "costs" / "cost_report.json"

    _write_json(eval_summary, {"repos": []})
    _write_json(latency, {"stats": [{"endpoint": "GET /health", "p50_ms": 4, "p95_ms": 9, "p99_ms": 14, "mean_ms": 5, "error_count": 0, "sample_count": 50}]})
    _write_json(quality, {"scores": [], "avg_composite": 0.0})
    _write_json(cost, {"records": [], "total_cost_usd": 0.0})

    markdown = generator.generate_markdown(
        ReportPaths(eval_summary=eval_summary, latency_results=latency, quality_scores=quality, cost_report=cost)
    )

    assert "| GET /health |" in markdown


def test_score_badge_thresholds() -> None:
    assert ReportGenerator._score_badge(0.90).startswith("GREEN")
    assert ReportGenerator._score_badge(0.75).startswith("YELLOW")
    assert ReportGenerator._score_badge(0.60).startswith("RED")


def test_generate_html_returns_html_document() -> None:
    generator = ReportGenerator()
    html = generator.generate_html("# Title\n\nBody")
    assert "<html>" in html.lower()
    assert "title" in html.lower()


def test_write_reports_outputs_both_formats(tmp_path) -> None:
    generator = ReportGenerator(output_dir=tmp_path)

    eval_summary = tmp_path / "results" / "eval_summary_x.json"
    latency = tmp_path / "latency" / "latency_results.json"
    quality = tmp_path / "quality" / "quality_scores.json"
    cost = tmp_path / "costs" / "cost_report.json"

    _write_json(eval_summary, {"repos": []})
    _write_json(latency, {"stats": []})
    _write_json(quality, {"scores": [], "avg_composite": 0.0})
    _write_json(cost, {"records": [], "total_cost_usd": 0.0})

    markdown_path, html_path = generator.write_reports(
        ReportPaths(eval_summary=eval_summary, latency_results=latency, quality_scores=quality, cost_report=cost)
    )

    assert markdown_path.exists()
    assert html_path.exists()


def test_missing_data_handled_without_crash(tmp_path) -> None:
    generator = ReportGenerator(output_dir=tmp_path)
    missing = tmp_path / "missing.json"

    markdown = generator.generate_markdown(
        ReportPaths(
            eval_summary=missing,
            latency_results=missing,
            quality_scores=missing,
            cost_report=missing,
        )
    )

    assert "Benchmark Report" in markdown
