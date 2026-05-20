"""Unit tests for analysis/bug_localizer.py."""

import json

from analysis.bug_localizer import (
    BugLocalizationResult,
    ErrorSignalParser,
    _build_bug_analysis_prompt,
    _parse_llm_bug_response,
)


class TestErrorSignalParser:
    def setup_method(self):
        self.parser = ErrorSignalParser()

    def test_parses_keyerror_with_file_and_line(self):
        result = self.parser.parse("KeyError: 'user_id' in verify_token at auth/middleware.py:52")
        assert result["exception_type"] == "KeyError"
        assert result["file_path"] == "auth/middleware.py"
        assert result["line_number"] == 52
        assert result["function_name"] == "verify_token"

    def test_parses_attribute_error(self):
        result = self.parser.parse("AttributeError: 'NoneType' has no attribute 'send' in notify_user")
        assert result["exception_type"] == "AttributeError"
        assert result["function_name"] == "notify_user"

    def test_parses_type_error_with_line_keyword(self):
        result = self.parser.parse("TypeError in process_payment (payments/processor.py line 88)")
        assert result["exception_type"] == "TypeError"
        assert result["file_path"] == "payments/processor.py"
        assert result["line_number"] == 88
        assert result["function_name"] == "process_payment"

    def test_free_text_returns_none_for_all_fields(self):
        result = self.parser.parse("users are getting logged out randomly")
        assert result["exception_type"] is None
        assert result["file_path"] is None
        assert result["line_number"] is None
        assert result["function_name"] is None

    def test_raw_field_always_present(self):
        text = "some error"
        result = self.parser.parse(text)
        assert result["raw"] == text

    def test_backslash_path_normalized(self):
        result = self.parser.parse("ValueError in func at auth\\service.py:10")
        assert result["file_path"] == "auth/service.py"


class TestParseLlmBugResponse:
    @staticmethod
    def _make_result():
        return BugLocalizationResult(error_signal="test error")

    def test_parses_valid_json(self):
        raw = json.dumps(
            {
                "root_cause_file": "auth/service.py",
                "root_cause_function": "verify_token",
                "root_cause_line": 52,
                "explanation": "The dict key is missing",
                "fix_suggestion": "Use .get() instead of []",
                "confidence": "high",
            }
        )
        result = _parse_llm_bug_response(raw, self._make_result(), {})
        assert result.root_cause_file == "auth/service.py"
        assert result.root_cause_function == "verify_token"
        assert result.root_cause_line == 52
        assert result.confidence == "high"
        assert "missing" in result.explanation

    def test_strips_markdown_code_fence(self):
        raw = (
            "```json\n"
            "{\"root_cause_file\": \"x.py\", \"root_cause_function\": \"f\", "
            "\"root_cause_line\": 1, \"explanation\": \"e\", \"fix_suggestion\": \"fix\", "
            "\"confidence\": \"low\"}\n```"
        )
        result = _parse_llm_bug_response(raw, self._make_result(), {})
        assert result.root_cause_file == "x.py"

    def test_gracefully_handles_invalid_json(self):
        result = _parse_llm_bug_response("not json at all", self._make_result(), {})
        assert result.confidence == "low"
        assert result.explanation != ""

    def test_falls_back_to_parsed_error_on_bad_json(self):
        parsed = {"file_path": "fallback.py", "function_name": "fallback_func", "line_number": 99}
        result = _parse_llm_bug_response("garbage", self._make_result(), parsed)
        assert result.root_cause_file == "fallback.py"
        assert result.root_cause_function == "fallback_func"
        assert result.root_cause_line == 99


class TestBuildBugPrompt:
    def test_error_signal_in_prompt(self):
        prompt = _build_bug_analysis_prompt(
            error_description="KeyError: test",
            parsed_error={},
            callers=[],
            callees=[],
            call_chain=[],
            code_by_function={},
        )
        assert "KeyError: test" in prompt

    def test_call_chain_in_prompt(self):
        prompt = _build_bug_analysis_prompt(
            error_description="err",
            parsed_error={},
            callers=["caller_a"],
            callees=["callee_b"],
            call_chain=["caller_a", "target", "callee_b"],
            code_by_function={},
        )
        assert "caller_a -> target -> callee_b" in prompt

    def test_code_block_in_prompt(self):
        prompt = _build_bug_analysis_prompt(
            error_description="err",
            parsed_error={},
            callers=[],
            callees=[],
            call_chain=["my_func"],
            code_by_function={"my_func": "def my_func(): pass"},
        )
        assert "def my_func(): pass" in prompt
        assert '<function name="my_func">' in prompt

    def test_json_schema_instruction_in_prompt(self):
        prompt = _build_bug_analysis_prompt(
            error_description="err",
            parsed_error={},
            callers=[],
            callees=[],
            call_chain=[],
            code_by_function={},
        )
        assert '"root_cause_file"' in prompt
        assert '"fix_suggestion"' in prompt
        assert '"confidence"' in prompt
