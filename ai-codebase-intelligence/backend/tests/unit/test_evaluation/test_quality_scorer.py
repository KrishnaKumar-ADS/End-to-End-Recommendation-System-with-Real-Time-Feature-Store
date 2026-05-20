"""Unit tests for evaluation.quality_scorer."""

from __future__ import annotations

from evaluation.quality_scorer import QualityScorer


def test_parse_valid_json_payload() -> None:
    scorer = QualityScorer(judge_callable=lambda _sys, _usr: "")
    parsed = scorer._parse_json_payload(
        '{"faithfulness": 4, "relevance": 5, "completeness": 3, "critique": "Solid answer"}'
    )
    assert parsed is not None
    assert parsed["faithfulness"] == 4
    assert parsed["relevance"] == 5
    assert parsed["completeness"] == 3


def test_parse_json_with_markdown_fences() -> None:
    scorer = QualityScorer(judge_callable=lambda _sys, _usr: "")
    parsed = scorer._parse_json_payload(
        "```json\n{\"faithfulness\": 5, \"relevance\": 4, \"completeness\": 4, \"critique\": \"Good\"}\n```"
    )
    assert parsed is not None
    assert parsed["faithfulness"] == 5


def test_parse_rejects_missing_keys() -> None:
    scorer = QualityScorer(judge_callable=lambda _sys, _usr: "")
    parsed = scorer._parse_json_payload('{"faithfulness": 4, "relevance": 4}')
    assert parsed is None


def test_parse_rejects_out_of_range_scores() -> None:
    scorer = QualityScorer(judge_callable=lambda _sys, _usr: "")
    parsed = scorer._parse_json_payload(
        '{"faithfulness": 0, "relevance": 6, "completeness": 3, "critique": "bad"}'
    )
    assert parsed is None


def test_score_answer_computes_composite() -> None:
    scorer = QualityScorer(
        judge_callable=lambda _sys, _usr: '{"faithfulness": 5, "relevance": 4, "completeness": 3, "critique": "ok"}'
    )
    score = scorer.score_answer(
        question_id="q1",
        repo_name="repo",
        question="How does auth work?",
        retrieved_context=["Auth implementation snippet"],
        generated_answer="It validates bearer tokens and checks scopes.",
    )
    assert score.composite_score == (5 + 4 + 3) / 3 / 5


def test_score_answer_retries_on_invalid_json_then_succeeds() -> None:
    responses = iter([
        "not json",
        '{"faithfulness": 4, "relevance": 4, "completeness": 4, "critique": "ok"}',
    ])

    def judge(_sys: str, _usr: str) -> str:
        return next(responses)

    scorer = QualityScorer(judge_callable=judge, max_retries=2)
    score = scorer.score_answer(
        question_id="q1",
        repo_name="repo",
        question="Q",
        retrieved_context=["C"],
        generated_answer="A",
    )
    assert score.composite_score is not None
    assert score.retries_used == 1


def test_score_answer_returns_null_scores_after_retry_exhaustion() -> None:
    scorer = QualityScorer(judge_callable=lambda _sys, _usr: "invalid", max_retries=2)
    score = scorer.score_answer(
        question_id="q1",
        repo_name="repo",
        question="Q",
        retrieved_context=["C"],
        generated_answer="A",
    )
    assert score.faithfulness is None
    assert score.relevance is None
    assert score.completeness is None
    assert score.composite_score is None


def test_score_many_returns_one_result_per_item() -> None:
    scorer = QualityScorer(
        judge_callable=lambda _sys, _usr: '{"faithfulness": 3, "relevance": 3, "completeness": 3, "critique": "avg"}'
    )
    scores = scorer.score_many(
        [
            {
                "question_id": "q1",
                "repo_name": "repo",
                "question": "Q1",
                "retrieved_context": ["ctx"],
                "generated_answer": "ans",
            },
            {
                "question_id": "q2",
                "repo_name": "repo",
                "question": "Q2",
                "retrieved_context": ["ctx"],
                "generated_answer": "ans",
            },
        ]
    )
    assert len(scores) == 2
    assert scores[0].question_id == "q1"
    assert scores[1].question_id == "q2"
