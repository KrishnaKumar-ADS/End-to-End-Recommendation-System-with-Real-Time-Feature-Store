from pathlib import Path
from dataclasses import dataclass
from ingestion.language_detector import detect_language, is_skipped_dir
from core.logging import get_logger

logger = get_logger(__name__)

MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB


@dataclass
class ScannedFile:
    path: Path
    relative_path: str
    language: str
    size_bytes: int
    line_count: int


def scan_repository(repo_path: Path) -> list[ScannedFile]:
    results: list[ScannedFile] = []

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue

        parts = file_path.relative_to(repo_path).parts
        if any(is_skipped_dir(part) for part in parts[:-1]):
            continue

        language = detect_language(file_path)
        if language is None:
            continue

        size_bytes = file_path.stat().st_size
        if size_bytes > MAX_FILE_SIZE_BYTES:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            line_count = content.count("\n") + 1
        except OSError:
            continue

        results.append(ScannedFile(
            path=file_path,
            relative_path=str(file_path.relative_to(repo_path)),
            language=language,
            size_bytes=size_bytes,
            line_count=line_count,
        ))

    logger.info("scan_complete", repo=str(repo_path), total_files=len(results))
    return results