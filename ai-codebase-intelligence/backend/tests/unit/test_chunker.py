from pathlib import Path
from ingestion.chunker import chunk_by_lines, CodeChunkData, chunk_file


def test_chunk_by_lines_basic(tmp_path):
    """Fallback chunker splits content into 50-line blocks."""
    lines = [f"line {i}" for i in range(120)]
    content = "\n".join(lines)
    f = tmp_path / "test.yaml"
    f.write_text(content)

    chunks = chunk_by_lines(content, "test.yaml", "yaml")
    assert len(chunks) == 3   # 50 + 50 + 20


def test_chunk_by_lines_skips_empty(tmp_path):
    content = "\n\n\n\n"
    chunks = chunk_by_lines(content, "empty.yaml", "yaml")
    assert len(chunks) == 0


def test_code_chunk_display_name_with_parent():
    c = CodeChunkData(
        name="my_method",
        chunk_type="method",
        content="def my_method(self): pass",
        start_line=10,
        end_line=12,
        language="python",
        file_path="app.py",
        parent_name="MyClass",
    )
    assert c.display_name == "MyClass.my_method"


def test_code_chunk_display_name_no_parent():
    c = CodeChunkData(
        name="my_func",
        chunk_type="function",
        content="def my_func(): pass\n    return 1",
        start_line=1,
        end_line=3,
        language="python",
        file_path="app.py",
    )
    assert c.display_name == "my_func"


def test_code_chunk_line_count():
    c = CodeChunkData(
        name="f",
        chunk_type="function",
        content="x",
        start_line=5,
        end_line=15,
        language="python",
        file_path="a.py",
    )
    assert c.line_count == 11


def test_chunk_file_returns_empty_for_blank_file(tmp_path):
    f = tmp_path / "blank.py"
    f.write_text("   \n\n\n")
    result = chunk_file(f, "blank.py", "python")
    assert result == []