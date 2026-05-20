"""Week 15 benchmark report generator (Markdown + HTML)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from markdown import markdown


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass(slots=True)
class ReportPaths:
    eval_summary: Path
    latency_results: Path
    quality_scores: Path
    cost_report: Path
    load_csv: Path | None = None


class ReportGenerator:
    """Generate benchmark summary report from evaluation artifacts."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or (_project_root() / "data" / "benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _score_badge(value: float | None) -> str:
        if value is None:
            return "N/A"
        if value >= 0.85:
            return f"GREEN {value:.2f}"
        if value >= 0.70:
            return f"YELLOW {value:.2f}"
        return f"RED {value:.2f}"

    @staticmethod
    def _load_load_metrics(load_csv: Path | None) -> dict[str, float | str]:
        if not load_csv or not load_csv.exists():
            return {
                "peak_rps": "N/A",
                "error_rate": "N/A",
                "median_response_ms": "N/A",
            }

        rows: list[dict[str, str]] = []
        with load_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)

        if not rows:
            return {
                "peak_rps": "N/A",
                "error_rate": "N/A",
                "median_response_ms": "N/A",
            }

        rps_values = [float(row.get("Requests/s", 0.0) or 0.0) for row in rows]
        median_values = [float(row.get("Median Response Time", 0.0) or 0.0) for row in rows]

        failures = sum(float(row.get("Failure Count", 0.0) or 0.0) for row in rows)
        requests = sum(float(row.get("Request Count", 0.0) or 0.0) for row in rows)
        error_rate = (failures / requests) if requests else 0.0

        return {
            "peak_rps": round(max(rps_values) if rps_values else 0.0, 3),
            "error_rate": round(error_rate, 6),
            "median_response_ms": round(sum(median_values) / max(1, len(median_values)), 3),
        }

    @staticmethod
    def _flatten_questions(eval_summary: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for repo in eval_summary.get("repos", []):
            repo_name = repo.get("repo_name", "unknown")
            for question in repo.get("questions", []):
                row = dict(question)
                row["repo_name"] = repo_name
                rows.append(row)
        return rows

    @staticmethod
    def _avg(values: list[float]) -> float:
        return round(sum(values) / max(1, len(values)), 6)

    def generate_markdown(self, paths: ReportPaths) -> str:
        eval_summary = _load_json(paths.eval_summary)
        latency = _load_json(paths.latency_results)
        quality = _load_json(paths.quality_scores)
        cost = _load_json(paths.cost_report)
        load = self._load_load_metrics(paths.load_csv)

        questions = self._flatten_questions(eval_summary)
        successful_questions = [q for q in questions if not q.get("error")]

        avg_quality = quality.get("avg_composite")
        total_cost = float(cost.get("total_cost_usd", 0.0) or 0.0)
        total_questions = len(successful_questions)
        avg_cost = total_cost / max(1, total_questions)

        generated_at = datetime.now(timezone.utc).isoformat()

        lines: list[str] = [
            "# AI Codebase Intelligence Platform - Benchmark Report",
            "",
            f"Generated: {generated_at}",
            "",
            "## Executive Summary",
            (
                "The evaluation pipeline executed ingestion, retrieval, reasoning, quality scoring, and cost accounting "
                f"for {len(eval_summary.get('repos', []))} repositories and {total_questions} answered questions. "
                f"Average quality score was {avg_quality if avg_quality is not None else 'N/A'} and average cost per answered query was ${avg_cost:.5f}."
            ),
            "",
            "## Ingestion Summary",
            "| Repository | Language | Files | Chunks | Ingestion Time (s) |",
            "|---|---|---:|---:|---:|",
        ]

        for repo in eval_summary.get("repos", []):
            lines.append(
                "| {repo} | {lang} | {files} | {chunks} | {time_s:.2f} |".format(
                    repo=repo.get("repo_name", "unknown"),
                    lang=repo.get("language", "unknown"),
                    files=repo.get("total_files", "-"),
                    chunks=repo.get("total_chunks", "-"),
                    time_s=float(repo.get("ingestion_time_seconds", 0.0) or 0.0),
                )
            )

        lines.extend(
            [
                "",
                "## Latency Benchmarks",
                "| Endpoint | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Errors |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )

        for stat in latency.get("stats", []):
            lines.append(
                "| {endpoint} | {p50:.1f} | {p95:.1f} | {p99:.1f} | {mean:.1f} | {errors}/{samples} |".format(
                    endpoint=stat.get("endpoint", "unknown"),
                    p50=float(stat.get("p50_ms", 0.0) or 0.0),
                    p95=float(stat.get("p95_ms", 0.0) or 0.0),
                    p99=float(stat.get("p99_ms", 0.0) or 0.0),
                    mean=float(stat.get("mean_ms", 0.0) or 0.0),
                    errors=int(stat.get("error_count", 0) or 0),
                    samples=int(stat.get("sample_count", 0) or 0),
                )
            )

        lines.extend(
            [
                "",
                "## TTFT Benchmarks",
                "| Endpoint | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) |",
                "|---|---:|---:|---:|",
            ]
        )

        for stat in latency.get("stats", []):
            if stat.get("ttft_p50_ms") is None:
                continue
            lines.append(
                "| {endpoint} | {p50:.1f} | {p95:.1f} | {p99:.1f} |".format(
                    endpoint=stat.get("endpoint", "unknown"),
                    p50=float(stat.get("ttft_p50_ms", 0.0) or 0.0),
                    p95=float(stat.get("ttft_p95_ms", 0.0) or 0.0),
                    p99=float(stat.get("ttft_p99_ms", 0.0) or 0.0),
                )
            )

        lines.extend(
            [
                "",
                "## Cost Breakdown",
                "| Provider | Model | Avg Cost / Query (USD) | Queries | Total Cost (USD) |",
                "|---|---|---:|---:|---:|",
            ]
        )

        provider_model_totals: dict[tuple[str, str], dict[str, float]] = {}
        for record in cost.get("records", []):
            key = (str(record.get("provider", "unknown")), str(record.get("model", "unknown")))
            entry = provider_model_totals.setdefault(key, {"queries": 0.0, "cost": 0.0})
            entry["queries"] += 1
            entry["cost"] += float(record.get("cost_usd", 0.0) or 0.0)

        for (provider, model), totals in sorted(provider_model_totals.items()):
            queries = int(totals["queries"])
            total = float(totals["cost"])
            avg = total / max(1, queries)
            lines.append(f"| {provider} | {model} | {avg:.6f} | {queries} | {total:.6f} |")

        lines.extend(
            [
                "",
                "## Quality Scores",
                "| Repo | Faithfulness | Relevance | Completeness | Composite |",
                "|---|---:|---:|---:|---|",
            ]
        )

        by_repo: dict[str, list[dict[str, Any]]] = {}
        for score in quality.get("scores", []):
            by_repo.setdefault(str(score.get("repo_name", "unknown")), []).append(score)

        for repo_name, repo_scores in sorted(by_repo.items()):
            faithfulness_values = [float(s["faithfulness"]) for s in repo_scores if s.get("faithfulness") is not None]
            relevance_values = [float(s["relevance"]) for s in repo_scores if s.get("relevance") is not None]
            completeness_values = [float(s["completeness"]) for s in repo_scores if s.get("completeness") is not None]
            composite_values = [float(s["composite_score"]) for s in repo_scores if s.get("composite_score") is not None]

            composite = self._avg(composite_values) if composite_values else None
            lines.append(
                "| {repo} | {f:.2f} | {r:.2f} | {c:.2f} | {badge} |".format(
                    repo=repo_name,
                    f=self._avg(faithfulness_values) if faithfulness_values else 0.0,
                    r=self._avg(relevance_values) if relevance_values else 0.0,
                    c=self._avg(completeness_values) if completeness_values else 0.0,
                    badge=self._score_badge(composite),
                )
            )

        lines.extend(
            [
                "",
                "## Model Comparison",
                "| Model | Avg Latency (ms) | Avg Cost (USD) | Answers |",
                "|---|---:|---:|---:|",
            ]
        )

        model_rows: dict[str, dict[str, float]] = {}
        for question in successful_questions:
            model = str(question.get("model_used", "unknown"))
            bucket = model_rows.setdefault(model, {"latency": 0.0, "cost": 0.0, "count": 0.0})
            bucket["latency"] += float(question.get("latency_ms", 0.0) or 0.0)
            bucket["cost"] += float(question.get("total_cost_usd", 0.0) or 0.0)
            bucket["count"] += 1

        for model, bucket in sorted(model_rows.items()):
            count = int(bucket["count"])
            lines.append(
                f"| {model} | {bucket['latency'] / max(1, count):.2f} | {bucket['cost'] / max(1, count):.6f} | {count} |"
            )

        lines.extend(
            [
                "",
                "## Load Test Summary",
                f"- Peak RPS: {load['peak_rps']}",
                f"- Error rate: {load['error_rate']}",
                f"- Median response (ms): {load['median_response_ms']}",
                "",
                "## Recommendations",
                "1. Use qwen/qwen-2.5-coder-32b-instruct for high-volume code explanation workloads.",
                "2. Route architecture/security evaluations to qwen/qwen-max when quality is prioritized over latency.",
                "3. Investigate highest p99 endpoints first, especially streaming ask TTFT and graph queries.",
            ]
        )

        return "\n".join(lines) + "\n"

    def generate_html(self, markdown_text: str) -> str:
        body = markdown(markdown_text, extensions=["tables", "fenced_code"])
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Benchmark Report</title>"
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.5;}"
            "table{border-collapse:collapse;width:100%;margin:12px 0;}"
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}"
            "th{background:#f4f4f4;}code{background:#f7f7f7;padding:2px 4px;border-radius:4px;}"
            "</style></head><body>"
            f"{body}</body></html>"
        )

    def write_reports(self, paths: ReportPaths) -> tuple[Path, Path]:
        markdown_text = self.generate_markdown(paths)
        html_text = self.generate_html(markdown_text)

        markdown_path = self.output_dir / "BENCHMARK_REPORT.md"
        html_path = self.output_dir / "BENCHMARK_REPORT.html"

        markdown_path.write_text(markdown_text, encoding="utf-8")
        html_path.write_text(html_text, encoding="utf-8")

        # Keep timestamped copies for archival.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (self.output_dir / f"BENCHMARK_REPORT_{stamp}.md").write_text(markdown_text, encoding="utf-8")
        (self.output_dir / f"BENCHMARK_REPORT_{stamp}.html").write_text(html_text, encoding="utf-8")

        return markdown_path, html_path
