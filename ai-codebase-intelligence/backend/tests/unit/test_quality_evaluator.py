"""Unit tests for evaluation.quality_evaluator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from evaluation.models import QualityScore
from evaluation.quality_evaluator import QualityEvaluator


class TestQualityScorePostInit:
    def test_overall_computed_from_components(self):
        score = QualityScore(faithfulness=1.0, relevance=1.0, completeness=1.0)
        assert score.overall == 1.0

    def test_overall_weighted_correctly(self):
        score = QualityScore(faithfulness=1.0, relevance=0.0, completeness=0.0)
        assert abs(score.overall - 0.5) < 1e-5

    def test_scores_clamped_above_one(self):
        score = QualityScore(faithfulness=1.5, relevance=2.0, completeness=1.1)
        assert score.faithfulness == 1.0
        assert score.relevance == 1.0
        assert score.completeness == 1.0

    def test_scores_clamped_below_zero(self):
        score = QualityScore(faithfulness=-0.1, relevance=-1.0, completeness=-0.5)
        assert score.faithfulness == 0.0
        assert score.relevance == 0.0
        assert score.completeness == 0.0

    def test_skipped_score_factory(self):
        score = QualityScore.skipped_score("Answer too short")
        assert score.skipped is True
        assert score.skip_reason == "Answer too short"
        assert score.overall == 0.0


class TestQualityEvaluatorParseResponse:
    def test_parse_clean_json(self):
        raw = '{"faithfulness": 0.9, "relevance": 0.8, "completeness": 0.7, "critique": "OK"}'
        result = QualityEvaluator._parse_response(raw)
        assert result["faithfulness"] == 0.9
        assert result["relevance"] == 0.8
        assert result["critique"] == "OK"

    def test_parse_json_with_markdown_fences(self):
        raw = "```json\n{\"faithfulness\": 0.85, \"relevance\": 0.75, \"completeness\": 0.65, \"critique\": \"test\"}\n```"
        result = QualityEvaluator._parse_response(raw)
        assert result is not None
        assert result["faithfulness"] == 0.85

    def test_parse_string_float_values(self):
        raw = '{"faithfulness": "0.9", "relevance": "0.8", "completeness": "0.7", "critique": "OK"}'
        result = QualityEvaluator._parse_response(raw)
        assert result["faithfulness"] == 0.9

    def test_parse_returns_none_for_missing_required_keys(self):
        raw = '{"faithfulness": 0.9, "relevance": 0.8}'
        result = QualityEvaluator._parse_response(raw)
        assert result is None

    def test_parse_returns_none_for_empty_string(self):
        assert QualityEvaluator._parse_response("") is None

    def test_parse_returns_none_for_plain_text(self):
        assert QualityEvaluator._parse_response("The answer is good.") is None


class TestQualityEvaluatorScore:
    @pytest.mark.asyncio
    async def test_returns_valid_score_on_success(self):
        mock_response = '{"faithfulness": 0.91, "relevance": 0.88, "completeness": 0.75, "critique": "Good"}'

        with patch("evaluation.quality_evaluator.QualityEvaluator.__init__", return_value=None):
            evaluator = QualityEvaluator.__new__(QualityEvaluator)
            evaluator._judge_model = "qwen/qwen-max"
            evaluator._client = AsyncMock()
            evaluator._client.complete = AsyncMock(return_value=mock_response)

            score = await evaluator.score(
                question="How does auth work?",
                answer="Authentication uses bcrypt to verify passwords in auth/service.py.",
                context_chunks=["def verify_password(plain, hashed): return bcrypt.checkpw(...)"],
            )

        assert score.skipped is False
        assert score.faithfulness == 0.91
        assert score.relevance == 0.88
        assert score.completeness == 0.75
        assert 0.0 < score.overall <= 1.0
        assert score.critique == "Good"

    @pytest.mark.asyncio
    async def test_returns_skipped_when_answer_too_short(self):
        with patch("evaluation.quality_evaluator.QualityEvaluator.__init__", return_value=None):
            evaluator = QualityEvaluator.__new__(QualityEvaluator)
            evaluator._judge_model = "qwen/qwen-max"
            evaluator._client = AsyncMock()

            score = await evaluator.score(
                question="How does X work?",
                answer="Yes.",
                context_chunks=["def foo(): pass"],
            )

        assert score.skipped is True
        assert "short" in (score.skip_reason or "").lower()

    @pytest.mark.asyncio
    async def test_returns_skipped_when_no_context(self):
        with patch("evaluation.quality_evaluator.QualityEvaluator.__init__", return_value=None):
            evaluator = QualityEvaluator.__new__(QualityEvaluator)
            evaluator._judge_model = "qwen/qwen-max"
            evaluator._client = AsyncMock()

            score = await evaluator.score(
                question="How does X work?",
                answer="X works by doing Y and Z which handles the primary flow of the system.",
                context_chunks=[],
            )

        assert score.skipped is True

    @pytest.mark.asyncio
    async def test_returns_skipped_on_llm_failure(self):
        with patch("evaluation.quality_evaluator.QualityEvaluator.__init__", return_value=None):
            evaluator = QualityEvaluator.__new__(QualityEvaluator)
            evaluator._judge_model = "qwen/qwen-max"
            evaluator._client = AsyncMock()
            evaluator._client.complete = AsyncMock(side_effect=Exception("OpenRouter connection refused"))

            score = await evaluator.score(
                question="How does auth work?",
                answer="Authentication uses bcrypt.checkpw() in the auth service to verify passwords.",
                context_chunks=["def verify_password(p, h): return bcrypt.checkpw(p, h)"],
            )

        assert score.skipped is True
        assert "failed" in (score.skip_reason or "").lower()

    @pytest.mark.asyncio
    async def test_returns_skipped_when_judge_returns_invalid_json(self):
        with patch("evaluation.quality_evaluator.QualityEvaluator.__init__", return_value=None):
            evaluator = QualityEvaluator.__new__(QualityEvaluator)
            evaluator._judge_model = "qwen/qwen-max"
            evaluator._client = AsyncMock()
            evaluator._client.complete = AsyncMock(return_value="The answer looks pretty good overall.")

            score = await evaluator.score(
                question="How does auth work?",
                answer="Authentication uses bcrypt.checkpw() in the auth service to verify passwords.",
                context_chunks=["def verify_password(p, h): return bcrypt.checkpw(p, h)"],
            )

        assert score.skipped is True
