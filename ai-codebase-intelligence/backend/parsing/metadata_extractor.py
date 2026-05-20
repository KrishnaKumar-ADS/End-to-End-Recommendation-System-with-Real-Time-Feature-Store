"""
Metadata Extractor — unified interface over all parsers.

This is the only file the ingestion pipeline calls.
It decides internally which parser to use based on language.

Usage:
    from parsing.metadata_extractor import extract_metadata

    chunks = extract_metadata(
        file_path=Path("src/auth.py"),
        relative_path="src/auth.py",
        language="python",
        content="def login(): ..."
    )
"""

from pathlib import Path
from ingestion.chunker import CodeChunkData, chunk_file
from core.logging import get_logger

logger = get_logger(__name__)


def extract_metadata(
    file_path: Path,
    relative_path: str,
    language: str,
    content: str | None = None,
) -> list[CodeChunkData]:
    """
    Extract all code chunks from a source file.

    Args:
        file_path:     absolute path to the file on disk
        relative_path: path relative to repo root (used for display)
        language:      detected language string (python, javascript, etc.)
        content:       optional pre-read file content (avoids double read)

    Returns:
        List of CodeChunkData objects, one per function/class/block.
    """
    if content is None:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.warning("metadata_read_failed", path=relative_path, error=str(e))
            return []

    chunks = chunk_file(file_path, relative_path, language)

    logger.info(
        "metadata_extracted",
        file=relative_path,
        language=language,
        chunks=len(chunks),
    )
    return chunks


def extract_metadata_batch(
    files: list[tuple[Path, str, str]],
) -> dict[str, list[CodeChunkData]]:
    """
    Extract metadata from multiple files.

    Args:
        files: list of (file_path, relative_path, language) tuples

    Returns:
        Dict mapping relative_path -> list of chunks
    """
    results = {}
    for file_path, relative_path, language in files:
        chunks = extract_metadata(file_path, relative_path, language)
        if chunks:
            results[relative_path] = chunks

    total_chunks = sum(len(v) for v in results.values())
    logger.info(
        "batch_extraction_complete",
        files=len(files),
        files_with_chunks=len(results),
        total_chunks=total_chunks,
    )
    return results