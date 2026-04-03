"""Deploy DS19 backend to a Hugging Face Docker Space.

Usage:
    python script_deploy_hf_space.py --repo-id <username>/<space-name>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi, get_token  # pyright: ignore[reportMissingImports]
    from huggingface_hub.utils import HfHubHTTPError  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required. Install with: pip install huggingface_hub"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent

ALLOW_PATTERNS = [
    "Dockerfile",
    ".dockerignore",
    "README.md",
    "requirements.txt",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/index.html",
    "frontend/tsconfig.json",
    "frontend/vite.config.ts",
    "frontend/postcss.config.js",
    "frontend/tailwind.config.js",
    "frontend/public/**",
    "frontend/src/**",
    "backend/__init__.py",
    "backend/app/**",
    "mlops/__init__.py",
    "mlops/ab_testing/**",
    "mlops/bandits/**",
    "mlops/mlflow_setup/**",
    "models/__init__.py",
    "models/two_tower/**",
    "models/saved/two_tower_best.pt",
    "models/saved/lgbm_ranker.pkl",
    "retrieval/faiss_item.index",
    "retrieval/item_idx_map.npy",
    "data/processed/item2idx.json",
    "data/processed/idx2item.json",
    "data/processed/dataset_meta.json",
    "data/processed/user2idx.json",
    "data/features/user_features.parquet",
    "data/features/item_features.parquet",
    "data/sequences/sequences.pkl",
    "data/raw/movies.csv",
]

IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/venv/**",
    "**/.venv/**",
    "**/.pytest_cache/**",
    "**/node_modules/**",
    "frontend/dist/**",
]


def resolve_token(explicit_token: str | None) -> str:
    token = (
        explicit_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or get_token()
    )
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Run huggingface-cli login, set HF_TOKEN, or pass --token."
        )
    return token


def deploy(
    repo_id: str,
    api: HfApi,
    private: bool,
    commit_message: str,
    max_retries: int,
    clean: bool,
) -> None:

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=private,
        exist_ok=True,
    )

    for attempt in range(1, max_retries + 1):
        try:
            delete_patterns = ["*", "**/*"] if clean else None
            api.upload_folder(
                folder_path=str(PROJECT_ROOT),
                repo_id=repo_id,
                repo_type="space",
                commit_message=commit_message,
                allow_patterns=ALLOW_PATTERNS,
                ignore_patterns=IGNORE_PATTERNS,
                delete_patterns=delete_patterns,
            )
            return
        except HfHubHTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            is_retryable = status_code in {502, 503, 504}
            if is_retryable and attempt < max_retries:
                backoff_seconds = attempt * 5
                print(
                    f"Upload attempt {attempt}/{max_retries} failed with HTTP {status_code}. "
                    f"Retrying in {backoff_seconds}s..."
                )
                time.sleep(backoff_seconds)
                continue
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy DS19 to Hugging Face Spaces")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Target Space repo id, for example: your-username/ds19-recommender",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN or HUGGINGFACEHUB_API_TOKEN)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private",
    )
    parser.add_argument(
        "--commit-message",
        default="Deploy DS19 recommender Space",
        help="Commit message for uploaded files",
    )
    parser.add_argument(
        "--max-retries",
        default=3,
        type=int,
        help="Maximum upload retries for transient 5xx responses",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing Space repo files before upload (keeps only allow-listed files)",
    )
    args = parser.parse_args()

    token = resolve_token(args.token)
    api = HfApi(token=token)

    repo_id = args.repo_id
    if not repo_id:
        whoami = api.whoami()
        username = whoami.get("name")
        if not username:
            raise RuntimeError("Could not resolve Hugging Face username from token")
        repo_id = f"{username}/ds19-recommender-api"

    deploy(
        repo_id=repo_id,
        api=api,
        private=args.private,
        commit_message=args.commit_message,
        max_retries=max(args.max_retries, 1),
        clean=args.clean,
    )

    print(f"Deployment complete: https://huggingface.co/spaces/{repo_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
