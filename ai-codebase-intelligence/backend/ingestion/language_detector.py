from pathlib import Path

EXTENSION_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".go": "go", ".rs": "rust", ".cpp": "cpp", ".cc": "cpp",
    ".c": "c", ".h": "c", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".sh": "bash", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".toml": "toml", ".sql": "sql",
    ".md": "markdown", ".html": "html", ".css": "css",
}

SKIP_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".next", "target", "vendor",
    ".idea", ".vscode", "coverage", ".pytest_cache",
}

SKIP_FILES = {
    ".DS_Store", "package-lock.json", "yarn.lock",
    "poetry.lock", "Pipfile.lock", "Cargo.lock",
}


def detect_language(file_path: Path) -> str | None:
    if file_path.name in SKIP_FILES:
        return None
    return EXTENSION_MAP.get(file_path.suffix.lower())


def is_skipped_dir(dir_name: str) -> bool:
    return dir_name in SKIP_DIRS