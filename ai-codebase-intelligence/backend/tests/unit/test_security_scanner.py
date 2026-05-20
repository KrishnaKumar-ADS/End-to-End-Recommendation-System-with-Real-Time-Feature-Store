"""Unit tests for analysis/security_scanner.py."""

import pytest

from analysis.security_scanner import SecurityScanner, Severity


@pytest.fixture
def scanner():
    return SecurityScanner()


def test_sql_injection_detected(scanner):
    findings = scanner.scan_chunk(
        content='query = f"SELECT * FROM users WHERE id = {user_id}"',
        file_path="db.py",
        function_name="get_user",
        chunk_offset=1,
    )
    categories = [finding.category for finding in findings]
    assert "SQL_INJECTION" in categories


def test_hardcoded_secret_detected(scanner):
    findings = scanner.scan_chunk(
        content="password = 'hardcoded_secret_123'",
        file_path="config.py",
        function_name="setup",
        chunk_offset=5,
    )
    assert any(finding.category == "HARDCODED_SECRET" for finding in findings)


def test_pickle_loads_detected(scanner):
    findings = scanner.scan_chunk(
        content="data = pickle.loads(user_data)",
        file_path="deserialize.py",
        function_name="load_data",
        chunk_offset=1,
    )
    assert any(finding.category == "UNSAFE_DESERIALIZATION" for finding in findings)


def test_command_injection_detected(scanner):
    findings = scanner.scan_chunk(
        content="subprocess.run(cmd, shell=True)",
        file_path="shell.py",
        function_name="run_command",
        chunk_offset=1,
    )
    assert any(finding.category == "COMMAND_INJECTION" for finding in findings)


def test_debug_mode_detected(scanner):
    findings = scanner.scan_chunk(
        content="app.run(debug=True, port=8080)",
        file_path="app.py",
        function_name="main",
        chunk_offset=1,
    )
    assert any(finding.category == "DEBUG_MODE" for finding in findings)


def test_weak_crypto_md5_detected(scanner):
    findings = scanner.scan_chunk(
        content="digest = hashlib.md5(content).hexdigest()",
        file_path="crypto.py",
        function_name="hash_content",
        chunk_offset=1,
    )
    assert any(finding.category == "WEAK_CRYPTOGRAPHY" for finding in findings)


def test_insecure_random_detected(scanner):
    findings = scanner.scan_chunk(
        content="token = random.randint(100000, 999999)",
        file_path="tokens.py",
        function_name="gen_token",
        chunk_offset=1,
    )
    assert any(finding.category == "INSECURE_RANDOM" for finding in findings)


def test_clean_code_returns_no_findings(scanner):
    findings = scanner.scan_chunk(
        content="def add(a: int, b: int) -> int:\n    return a + b",
        file_path="math.py",
        function_name="add",
        chunk_offset=1,
    )
    assert findings == []


def test_line_number_offset_applied(scanner):
    code = "x = 1\ny = 2\npassword = 'secret123'\nz = 3"
    findings = scanner.scan_chunk(
        content=code,
        file_path="file.py",
        function_name="func",
        chunk_offset=10,
    )
    secret_finding = next(finding for finding in findings if finding.category == "HARDCODED_SECRET")
    assert secret_finding.line_number == 12


def test_sql_injection_is_critical(scanner):
    findings = scanner.scan_chunk(
        content='execute(f"SELECT * WHERE id={user_id}")',
        file_path="q.py",
        function_name="query",
        chunk_offset=1,
    )
    assert findings[0].severity == Severity.CRITICAL


def test_insecure_random_is_medium(scanner):
    findings = scanner.scan_chunk(
        content="x = random.random()",
        file_path="r.py",
        function_name="rand",
        chunk_offset=1,
    )
    assert findings[0].severity == Severity.MEDIUM


def test_batch_scan_aggregates_findings(scanner):
    chunks = [
        {"content": "password = 'abc123'", "file_path": "a.py", "name": "a", "start_line": 1},
        {"content": "data = pickle.loads(x)", "file_path": "b.py", "name": "b", "start_line": 1},
        {"content": "def safe(): return 1", "file_path": "c.py", "name": "c", "start_line": 1},
    ]
    findings = scanner.scan_chunks_batch(chunks)
    assert len(findings) == 2


def test_batch_scan_sorted_by_severity(scanner):
    chunks = [
        {"content": "x = random.random()", "file_path": "r.py", "name": "r", "start_line": 1},
        {"content": "password = 'secret'", "file_path": "s.py", "name": "s", "start_line": 1},
        {"content": "app.run(debug=True)", "file_path": "a.py", "name": "a", "start_line": 1},
    ]
    findings = scanner.scan_chunks_batch(chunks)
    assert findings[0].severity.value in ("high", "critical")
    assert findings[-1].severity.value == "medium"


def test_get_rules_summary_returns_15_rules(scanner):
    summary = scanner.get_rules_summary()
    assert len(summary) == 15


def test_finding_has_cwe_id(scanner):
    findings = scanner.scan_chunk(
        content="execute(f'SELECT * WHERE id={uid}')",
        file_path="q.py",
        function_name="q",
        chunk_offset=1,
    )
    assert any(finding.cwe_id == "CWE-89" for finding in findings)


def test_matched_text_truncated_to_200_chars(scanner):
    long_line = "password = 'secret' " + ("x" * 300)
    findings = scanner.scan_chunk(
        content=long_line,
        file_path="f.py",
        function_name="f",
        chunk_offset=1,
    )
    if findings:
        assert len(findings[0].matched_text) <= 200
