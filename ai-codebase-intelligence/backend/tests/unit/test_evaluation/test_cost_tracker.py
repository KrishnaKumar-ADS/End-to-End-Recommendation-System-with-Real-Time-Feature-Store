"""Unit tests for evaluation.cost_tracker."""

from __future__ import annotations

import json

import pytest

from evaluation.cost_tracker import BudgetExceededError, CostTracker, ModelPricing


def test_pricing_exact_lookup_for_known_model() -> None:
    tracker = CostTracker()
    pricing = tracker._resolve_pricing("qwen/qwen-max")
    assert isinstance(pricing, ModelPricing)
    assert pricing.input_per_1m_usd > 0
    assert pricing.output_per_1m_usd > pricing.input_per_1m_usd


def test_unknown_model_falls_back_to_default_pricing() -> None:
    tracker = CostTracker()
    pricing = tracker._resolve_pricing("unknown-model")
    assert pricing.input_per_1m_usd == 0.25
    assert pricing.output_per_1m_usd == 1.00


def test_token_counting_returns_positive_for_non_empty_text() -> None:
    tracker = CostTracker()
    assert tracker.count_tokens("hello world") > 0


def test_cost_increases_with_token_volume() -> None:
    tracker = CostTracker()
    low = tracker.estimate_cost_usd("openrouter", "qwen/qwen-max", 100, 50)
    high = tracker.estimate_cost_usd("openrouter", "qwen/qwen-max", 1000, 500)
    assert high > low


def test_openrouter_fee_is_applied() -> None:
    tracker = CostTracker(openrouter_fee_pct=0.05)
    with_fee = tracker.estimate_cost_usd("openrouter", "qwen/qwen-2.5-coder-32b-instruct", 100000, 100000)
    without_fee = CostTracker(openrouter_fee_pct=0.0).estimate_cost_usd(
        "openrouter", "qwen/qwen-2.5-coder-32b-instruct", 100000, 100000
    )
    assert with_fee > without_fee


def test_budget_exceeded_raises_exception() -> None:
    tracker = CostTracker(daily_budget_usd=0.0001)
    with pytest.raises(BudgetExceededError):
        tracker.record_call(
            repo_name="repo",
            question_id="q1",
            provider="openrouter",
            model="qwen/qwen-max",
            prompt_tokens=50000,
            completion_tokens=50000,
        )


def test_per_repo_totals_aggregate_records() -> None:
    tracker = CostTracker(daily_budget_usd=100.0)
    tracker.record_call("repo-a", "q1", "openrouter", "qwen/qwen-max", prompt_tokens=100, completion_tokens=50)
    tracker.record_call("repo-a", "q2", "openrouter", "qwen/qwen-max", prompt_tokens=200, completion_tokens=75)
    totals = tracker.per_repo_totals()
    assert totals["repo-a"]["question_count"] == 2
    assert totals["repo-a"]["prompt_tokens"] == 300
    assert totals["repo-a"]["completion_tokens"] == 125


def test_save_report_writes_valid_json(tmp_path) -> None:
    tracker = CostTracker(daily_budget_usd=10.0)
    tracker.record_call("repo-a", "q1", "openrouter", "qwen/qwen-max", prompt_tokens=100, completion_tokens=50)

    output = tracker.save_report(tmp_path / "cost_report.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["total_prompt_tokens"] == 100
    assert payload["total_completion_tokens"] == 50
    assert payload["records"][0]["question_id"] == "q1"
