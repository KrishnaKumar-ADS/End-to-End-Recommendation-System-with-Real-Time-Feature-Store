"""
Code Chunker — splits source files into logical code chunks.

A chunk is the smallest meaningful unit of code:
  - A function definition
  - A class definition
  - A method inside a class
  - A module-level code block (fallback)

Each chunk contains:
  - name        : function/class name
  - chunk_type  : "function", "class", "method", "module"
  - content     : the actual source code lines
  - start_line  : line number where chunk starts (1-indexed)
  - end_line    : line number where chunk ends
  - language    : python / javascript / etc.
  - docstring   : extracted docstring if available (Python only in Week 2)
"""

from dataclasses import dataclass, field
from pathlib import Path
from core.logging import get_logger

logger = get_logger(__name__)

# Maximum lines per fallback chunk (for non-parseable files)
FALLBACK_CHUNK_SIZE = 50

# Minimum lines for a chunk to be worth storing
# (skip one-liner functions like `def get_id(self): return self.id`)
MIN_CHUNK_LINES = 3


@dataclass
class CodeChunkData:
    """
    Represents a single extracted code chunk before DB storage.
    """
    name: str
    chunk_type: str          # function, class, method, module
    content: str             # raw source code of this chunk
    start_line: int          # 1-indexed
    end_line: int            # 1-indexed
    language: str
    file_path: str           # relative path within repo
    docstring: str = ""      # extracted docstring if available
    parent_name: str = ""    # if method, the class it belongs to
    decorators: list[str] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def display_name(self) -> str:
        if self.parent_name:
            return f"{self.parent_name}.{self.name}"
        return self.name

    def __repr__(self) -> str:
        return f"<CodeChunk {self.chunk_type}:{self.display_name} L{self.start_line}-{self.end_line}>"


def chunk_file(file_path: Path, relative_path: str, language: str) -> list[CodeChunkData]:
    """
    Main entry point. Given a file path and language, returns a list
    of CodeChunkData objects — one per meaningful code unit.

    Routing:
      python      → python AST chunker
      js/ts       → tree-sitter chunker (Week 2 Day 3)
      go/java     → tree-sitter chunker (Week 2 Day 3)
      everything  → fallback line chunker
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        logger.warning("chunker_read_failed", path=str(file_path), error=str(e))
        return []

    if not content.strip():
        return []

    if language == "python":
        from ingestion.chunkers.python_chunker import chunk_python
        chunks = chunk_python(content, relative_path)

    elif language in ("javascript", "typescript"):
        from ingestion.chunkers.treesitter_chunker import chunk_with_treesitter
        chunks = chunk_with_treesitter(content, relative_path, language)

    elif language in ("go", "java", "rust", "cpp", "c"):
        from ingestion.chunkers.treesitter_chunker import chunk_with_treesitter
        chunks = chunk_with_treesitter(content, relative_path, language)

    else:
        chunks = chunk_by_lines(content, relative_path, language)

    # Filter out tiny chunks that aren't worth embedding
    chunks = [c for c in chunks if c.line_count >= MIN_CHUNK_LINES]

    logger.debug(
        "chunked_file",
        path=relative_path,
        language=language,
        chunks=len(chunks),
    )
    return chunks


def chunk_by_lines(
    content: str,
    relative_path: str,
    language: str,
    chunk_size: int = FALLBACK_CHUNK_SIZE,
) -> list[CodeChunkData]:
    """
    Fallback chunker — splits file into fixed-size line blocks.
    Used for YAML, SQL, Markdown, and any unsupported language.
    """
    lines = content.splitlines()
    chunks = []

    for i in range(0, len(lines), chunk_size):
        block = lines[i : i + chunk_size]
        start_line = i + 1
        end_line = i + len(block)
        chunk_content = "\n".join(block)

        if not chunk_content.strip():
            continue

        chunks.append(CodeChunkData(
            name=f"block_{start_line}",
            chunk_type="module",
            content=chunk_content,
            start_line=start_line,
            end_line=end_line,
            language=language,
            file_path=relative_path,
        ))

    return chunks