import shutil
from pathlib import Path
import git
from core.config import get_settings
from core.logging import get_logger
from core.exceptions import RepoNotFoundError

logger = get_logger(__name__)
settings = get_settings()

REPOS_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
REPOS_DIR.mkdir(parents=True, exist_ok=True)


def clone_repository(github_url: str, repo_id: str, branch: str = "main") -> Path:
    dest = REPOS_DIR / repo_id

    if dest.exists():
        logger.info("repo_already_cloned", repo_id=repo_id)
        return dest

    logger.info("cloning_repo", url=github_url, dest=str(dest))

    try:
        git.Repo.clone_from(url=github_url, to_path=str(dest), branch=branch, depth=1)
        logger.info("clone_complete", repo_id=repo_id)
        return dest

    except git.exc.GitCommandError as e:
        if branch == "main":
            logger.warning("main_branch_failed_trying_master")
            try:
                git.Repo.clone_from(url=github_url, to_path=str(dest), branch="master", depth=1)
                return dest
            except git.exc.GitCommandError as e2:
                raise RepoNotFoundError(f"Could not clone {github_url}: {e2}") from e2
        raise RepoNotFoundError(f"Could not clone {github_url}: {e}") from e


def delete_repository(repo_id: str) -> None:
    dest = REPOS_DIR / repo_id
    if dest.exists():
        shutil.rmtree(dest)


def get_repo_path(repo_id: str) -> Path | None:
    dest = REPOS_DIR / repo_id
    return dest if dest.exists() else None