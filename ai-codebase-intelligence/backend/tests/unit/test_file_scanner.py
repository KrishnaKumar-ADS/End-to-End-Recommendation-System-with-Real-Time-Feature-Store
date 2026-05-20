import pytest
from pathlib import Path
from ingestion.file_scanner import scan_repository
from ingestion.language_detector import detect_language, is_skipped_dir


def test_detect_python(): assert detect_language(Path("app.py")) == "python"
def test_detect_typescript(): assert detect_language(Path("index.tsx")) == "typescript"
def test_detect_go(): assert detect_language(Path("main.go")) == "go"
def test_detect_unsupported(): assert detect_language(Path("image.png")) is None
def test_detect_lockfile(): assert detect_language(Path("package-lock.json")) is None
def test_skip_node_modules(): assert is_skipped_dir("node_modules") is True
def test_skip_pycache(): assert is_skipped_dir("__pycache__") is True
def test_no_skip_src(): assert is_skipped_dir("src") is False


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "src" / "utils.py").write_text("import os\n")
    (tmp_path / "src" / "app.ts").write_text("const x = 1;\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "README.md").write_text("# Project\n")
    return tmp_path

def test_scan_finds_python(sample_repo):
    files = scan_repository(sample_repo)
    assert "python" in [f.language for f in files]

def test_scan_excludes_pycache(sample_repo):
    files = scan_repository(sample_repo)
    assert not any("__pycache__" in f.relative_path for f in files)

def test_scan_correct_file_count(sample_repo):
    files = scan_repository(sample_repo)
    assert len(files) == 4

def test_scan_line_count(sample_repo):
    files = scan_repository(sample_repo)
    main_py = next(f for f in files if "main.py" in f.relative_path)
    assert main_py.line_count == 3