"""Unit tests for evaluation.provider_comparison."""

from __future__ import annotations

from evaluation.provider_comparison import (
    _ms_str,
    _score_emoji,
    aggregate_metrics,
    generate_markdown,
)


def test_ms_str_formats_milliseconds_and_seconds() -> None:
    assert _ms_str(850) == "850ms"
    assert _ms_str(1500) == "1.5s"


def test_score_emoji_thresholds() -> None:
    assert _score_emoji(0.85).startswith("GREEN")
    assert _score_emoji(0.65).startswith("YELLOW")
    assert _score_emoji(0.45).startswith("RED")


def test_aggregate_metrics_collects_query_counts() -> None:
    eval_summary = {
        "repos": [
            {
                "repo_name": "repo-a",
                "questions": [
                    {
                        "model_used": "qwen/qwen-2.5-coder-32b-instruct",
                        "provider_used": "openrouter",
                        "latency_ms": 1200,
                        "category": "code_explanation",
                    },
                    {
                        "model_used": "qwen/qwen-max",
                        "provider_used": "openrouter",
                        "latency_ms": 2400,
                        "category": "architecture",
                    },
                ],
            }
        ]
    }
    metrics = aggregate_metrics(
        eval_summary=eval_summary,
        latency_results={},
        quality_scores={},
        cost_report={},
    )

    by_model = {item.model: item for item in metrics}
    assert by_model["qwen/qwen-2.5-coder-32b-instruct"].query_count == 1
    assert by_model["qwen/qwen-max"].query_count == 1


def test_generate_markdown_contains_summary_table() -> None:
    metrics = aggregate_metrics(
        eval_summary={
            "repos": [
                {
                    "repo_name": "repo-a",
                    "questions": [
                        {
                            "model_used": "qwen/qwen-2.5-coder-32b-instruct",
                            "provider_used": "openrouter",
                            "latency_ms": 1000,
                            "category": "code_qa",
                        }
                    ],
                }
            ]
        },
        latency_results={},
        quality_scores={},
        cost_report={"records": []},
    )
    markdown = generate_markdown(metrics)

    assert "# LLM Provider Comparison" in markdown
    assert "## Summary" in markdown
    assert "qwen/qwen-2.5-coder-32b-instruct" in markdown
