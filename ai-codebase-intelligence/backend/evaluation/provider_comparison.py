"""Week 16 provider comparison report generator.

Creates:
- data/benchmarks/PROVIDER_COMPARISON.md
- data/benchmarks/PROVIDER_COMPARISON.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_mean(values: list[float]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(mean(cleaned))


def _score_emoji(score: float | None) -> str:
    if score is None:
        return "N/A"
    if score >= 0.80:
        return f"GREEN {score:.2f}"
    if score >= 0.60:
        return f"YELLOW {score:.2f}"
    return f"RED {score:.2f}"


def _ms_str(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}ms"


def _usd(value: float) -> str:
    return f"${value:.6f}" if value < 0.01 else f"${value:.3f}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_json(path: Path, glob: str) -> Path | None:
    files = sorted(path.glob(glob))
    if not files:
        return None
    return files[-1]


@dataclass(slots=True)
class ModelMetrics:
    model: str
    provider: str
    input_per_1m: float
    output_per_1m: float
    query_count: int = 0
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    ttft_p50_ms: float | None = None
    avg_cost_per_query_usd: float = 0.0
    total_cost_usd: float = 0.0
    faithfulness: float | None = None
    relevance: float | None = None
    completeness: float | None = None
    composite: float | None = None
    routed_task_types: list[str] | None = None


PRICING_TABLE: dict[str, tuple[str, float, float]] = {
    "qwen/qwen-2.5-coder-32b-instruct": ("OpenRouter", 0.18, 0.18),
    "qwen/qwen-max": ("OpenRouter", 1.60, 6.40),
    "gemini-2.0-flash": ("Gemini", 0.075, 0.30),
    "models/text-embedding-004": ("Gemini", 0.0, 0.0),
    "models/gemini-embedding-001": ("Gemini", 0.0, 0.0),
    "deepseek-coder-v2": ("DeepSeek", 0.14, 0.28),
    "deepseek-chat": ("DeepSeek", 0.07, 1.10),
    "deepseek-reasoner": ("DeepSeek", 0.55, 2.19),
}


def _build_seed_metrics() -> dict[str, ModelMetrics]:
    metrics: dict[str, ModelMetrics] = {}
    for model, (provider, input_rate, output_rate) in PRICING_TABLE.items():
        metrics[model] = ModelMetrics(
            model=model,
            provider=provider,
            input_per_1m=input_rate,
            output_per_1m=output_rate,
            routed_task_types=[],
        )
    return metrics


def _collect_question_rows(eval_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repo in eval_summary.get("repos", []):
        for question in repo.get("questions", []):
            row = dict(question)
            row["repo_name"] = repo.get("repo_name", "unknown")
            rows.append(row)
    return rows


def aggregate_metrics(
    eval_summary: dict[str, Any],
    latency_results: dict[str, Any],
    quality_scores: dict[str, Any],
    cost_report: dict[str, Any],
) -> list[ModelMetrics]:
    metrics = _build_seed_metrics()

    question_rows = _collect_question_rows(eval_summary)

    per_model_latencies: dict[str, list[float]] = {}
    per_model_tasks: dict[str, set[str]] = {}
    for row in question_rows:
        model = str(row.get("model_used") or "").strip()
        if not model:
            continue
        if model not in metrics:
            metrics[model] = ModelMetrics(
                model=model,
                provider=str(row.get("provider_used") or "unknown").title(),
                input_per_1m=0.25,
                output_per_1m=1.00,
                routed_task_types=[],
            )
        latency_ms = row.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            per_model_latencies.setdefault(model, []).append(float(latency_ms))
        per_model_tasks.setdefault(model, set()).add(str(row.get("category") or "unknown"))
        metrics[model].query_count += 1

    for model, values in per_model_latencies.items():
        values_sorted = sorted(values)
        count = len(values_sorted)
        metrics[model].avg_latency_ms = _safe_mean(values_sorted)
        if count:
            metrics[model].p50_latency_ms = values_sorted[int((count - 1) * 0.50)]
            metrics[model].p95_latency_ms = values_sorted[int((count - 1) * 0.95)]
            metrics[model].p99_latency_ms = values_sorted[int((count - 1) * 0.99)]

    for model, tasks in per_model_tasks.items():
        metrics[model].routed_task_types = sorted(tasks)

    # Global TTFT from latency benchmark for streaming ask endpoint.
    for stat in latency_results.get("stats", []):
        endpoint = str(stat.get("endpoint", ""))
        if "ask" in endpoint.lower() and stat.get("ttft_p50_ms") is not None:
            ttft = float(stat.get("ttft_p50_ms"))
            for item in metrics.values():
                if item.query_count > 0:
                    item.ttft_p50_ms = ttft
            break

    per_model_costs: dict[str, list[float]] = {}
    for record in cost_report.get("records", []):
        model = str(record.get("model") or "").strip()
        if not model:
            continue
        per_model_costs.setdefault(model, []).append(float(record.get("cost_usd") or 0.0))

    for model, costs in per_model_costs.items():
        if model not in metrics:
            metrics[model] = ModelMetrics(
                model=model,
                provider=str(cost_report.get("provider") or "unknown").title(),
                input_per_1m=0.25,
                output_per_1m=1.00,
                routed_task_types=[],
            )
        total = float(sum(costs))
        metrics[model].total_cost_usd = total
        metrics[model].avg_cost_per_query_usd = total / max(1, len(costs))
        metrics[model].query_count = max(metrics[model].query_count, len(costs))

    quality_rows = quality_scores.get("scores", [])
    per_model_f: dict[str, list[float]] = {}
    per_model_r: dict[str, list[float]] = {}
    per_model_c: dict[str, list[float]] = {}
    per_model_comp: dict[str, list[float]] = {}

    for row in quality_rows:
        model = str(row.get("model_used") or "").strip()
        if not model:
            continue
        if row.get("faithfulness") is not None:
            per_model_f.setdefault(model, []).append(float(row["faithfulness"]) / 5.0)
        if row.get("relevance") is not None:
            per_model_r.setdefault(model, []).append(float(row["relevance"]) / 5.0)
        if row.get("completeness") is not None:
            per_model_c.setdefault(model, []).append(float(row["completeness"]) / 5.0)
        if row.get("composite_score") is not None:
            per_model_comp.setdefault(model, []).append(float(row["composite_score"]))

    for model in set(list(per_model_f.keys()) + list(per_model_r.keys()) + list(per_model_c.keys()) + list(per_model_comp.keys())):
        if model not in metrics:
            metrics[model] = ModelMetrics(
                model=model,
                provider="Unknown",
                input_per_1m=0.25,
                output_per_1m=1.00,
                routed_task_types=[],
            )
        metrics[model].faithfulness = _safe_mean(per_model_f.get(model, []))
        metrics[model].relevance = _safe_mean(per_model_r.get(model, []))
        metrics[model].completeness = _safe_mean(per_model_c.get(model, []))
        metrics[model].composite = _safe_mean(per_model_comp.get(model, []))

    results = [item for item in metrics.values() if item.query_count > 0 or item.model in PRICING_TABLE]
    results.sort(key=lambda item: (item.query_count, item.model), reverse=True)
    return results


def generate_markdown(metrics: list[ModelMetrics]) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    used = [item for item in metrics if item.query_count > 0]

    lines = [
        "# LLM Provider Comparison",
        "",
        f"Generated: {timestamp}",
        "",
        "## Summary",
        "",
        "| Model | Provider | Queries | Avg Latency | TTFT p50 | Avg Cost/Query | Composite Quality |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for item in used:
        lines.append(
            "| {model} | {provider} | {count} | {lat} | {ttft} | {cost} | {quality} |".format(
                model=item.model,
                provider=item.provider,
                count=item.query_count,
                lat=_ms_str(item.avg_latency_ms),
                ttft=_ms_str(item.ttft_p50_ms),
                cost=_usd(item.avg_cost_per_query_usd),
                quality=_score_emoji(item.composite),
            )
        )

    lines.extend(
        [
            "",
            "## Pricing Table",
            "",
            "| Model | Provider | Input $/1M | Output $/1M |",
            "|---|---|---:|---:|",
        ]
    )

    for item in metrics:
        lines.append(
            f"| {item.model} | {item.provider} | {item.input_per_1m:.3f} | {item.output_per_1m:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Routing Recommendation",
            "",
            "- Use qwen/qwen-2.5-coder-32b-instruct for high-volume code explanation and bug tracing.",
            "- Use qwen/qwen-max for architecture and security deep reasoning tasks.",
            "- Use Gemini embedding models for semantic retrieval; embedding cost remains near zero.",
        ]
    )

    return "\n".join(lines) + "\n"


def generate_html(metrics: list[ModelMetrics], markdown_text: str) -> str:
    rows = []
    for item in metrics:
        if item.query_count <= 0:
            continue
        rows.append(
            "<tr><td>{model}</td><td>{provider}</td><td>{count}</td><td>{lat}</td><td>{cost}</td><td>{quality}</td></tr>".format(
                model=item.model,
                provider=item.provider,
                count=item.query_count,
                lat=_ms_str(item.avg_latency_ms),
                cost=_usd(item.avg_cost_per_query_usd),
                quality=_score_emoji(item.composite),
            )
        )

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Provider Comparison</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:24px;}"
        "table{border-collapse:collapse;width:100%;margin-top:16px;}"
        "th,td{border:1px solid #334155;padding:8px;text-align:left;}"
        "th{background:#1e293b;}tr:nth-child(even){background:#111827;}"
        "code{background:#1e293b;padding:2px 4px;border-radius:4px;}</style></head><body>"
        "<h1>LLM Provider Comparison</h1>"
        "<p>Week 16 generated report from benchmark artifacts.</p>"
        "<table><thead><tr><th>Model</th><th>Provider</th><th>Queries</th><th>Avg Latency</th><th>Avg Cost</th><th>Quality</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<h2>Markdown Source</h2><pre>"
        + markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre></body></html>"
    )


def load_inputs(benchmark_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    results_dir = benchmark_dir / "results"
    latency_dir = benchmark_dir / "latency"
    quality_dir = benchmark_dir / "quality"
    costs_dir = benchmark_dir / "costs"

    eval_path = _latest_json(results_dir, "eval_summary_*.json")
    latency_path = _latest_json(latency_dir, "latency*.json")
    quality_path = _latest_json(quality_dir, "quality*.json")
    cost_path = _latest_json(costs_dir, "cost*.json")

    return (
        _read_json(eval_path) if eval_path else {},
        _read_json(latency_path) if latency_path else {},
        _read_json(quality_path) if quality_path else {},
        _read_json(cost_path) if cost_path else {},
    )


def write_reports(output_dir: Path, markdown_text: str, html_text: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "PROVIDER_COMPARISON.md"
    html_path = output_dir / "PROVIDER_COMPARISON.html"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return markdown_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Week 16 LLM provider comparison report.")
    parser.add_argument(
        "--output-dir",
        default=str(_project_root() / "data" / "benchmarks"),
        help="Benchmark directory containing results/latency/quality/costs folders.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Only generate markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    eval_summary, latency_results, quality_scores, cost_report = load_inputs(output_dir)

    metrics = aggregate_metrics(
        eval_summary=eval_summary,
        latency_results=latency_results,
        quality_scores=quality_scores,
        cost_report=cost_report,
    )
    markdown_text = generate_markdown(metrics)
    html_text = "" if args.no_html else generate_html(metrics, markdown_text)

    markdown_path, html_path = write_reports(output_dir, markdown_text, html_text or "")

    print("=== PROVIDER COMPARISON COMPLETE ===")
    print(f"Markdown: {markdown_path}")
    if args.no_html:
        print("HTML: skipped (--no-html)")
    else:
        print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
