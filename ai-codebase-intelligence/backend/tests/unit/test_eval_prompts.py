"""Unit tests for evaluation prompt builder."""

from evaluation.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_message


class TestJudgeSystemPrompt:
    def test_system_prompt_contains_three_dimensions(self):
        assert "faithfulness" in JUDGE_SYSTEM_PROMPT.lower()
        assert "relevance" in JUDGE_SYSTEM_PROMPT.lower()
        assert "completeness" in JUDGE_SYSTEM_PROMPT.lower()

    def test_system_prompt_specifies_json_output(self):
        assert "json" in JUDGE_SYSTEM_PROMPT.lower()

    def test_system_prompt_specifies_float_range(self):
        assert "0.0" in JUDGE_SYSTEM_PROMPT
        assert "1.0" in JUDGE_SYSTEM_PROMPT


class TestBuildJudgeUserMessage:
    def test_contains_question(self):
        msg = build_judge_user_message(
            question="How does auth work?",
            answer="Auth uses bcrypt.",
            context_chunks=["def verify_password(): ..."],
        )
        assert "How does auth work?" in msg

    def test_contains_answer(self):
        msg = build_judge_user_message(
            question="Q?",
            answer="Auth uses bcrypt to hash passwords.",
            context_chunks=["def verify_password(): ..."],
        )
        assert "Auth uses bcrypt" in msg

    def test_contains_context_chunks(self):
        msg = build_judge_user_message(
            question="Q?",
            answer="A long enough answer to satisfy the length constraint.",
            context_chunks=["def my_unique_function_abc(): pass"],
        )
        assert "my_unique_function_abc" in msg

    def test_handles_empty_context(self):
        msg = build_judge_user_message(
            question="Q?",
            answer="A long enough answer that definitely covers the requirement.",
            context_chunks=[],
        )
        assert "no context retrieved" in msg.lower()

    def test_truncates_long_context(self):
        big_chunk = "x" * 10000
        msg = build_judge_user_message(
            question="Q?",
            answer="A.",
            context_chunks=[big_chunk],
            max_context_chars=500,
        )
        assert "[truncated]" in msg
        assert len(msg) < 10000 + 500

    def test_multiple_chunks_labelled(self):
        msg = build_judge_user_message(
            question="Q?",
            answer="Answer here that is long enough to pass all checks.",
            context_chunks=["chunk one code", "chunk two code"],
        )
        assert "[Chunk 1]" in msg
        assert "[Chunk 2]" in msg
