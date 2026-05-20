import git

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ingestion.repo_loader import clone_repository, get_repo_path, delete_repository
from core.exceptions import RepoNotFoundError


@pytest.fixture
def tmp_repos(tmp_path, monkeypatch):
    import ingestion.repo_loader as loader
    monkeypatch.setattr(loader, "REPOS_DIR", tmp_path)
    return tmp_path


def test_get_path_none_when_missing(tmp_repos):
    assert get_repo_path("nope") is None

def test_get_path_returns_when_exists(tmp_repos):
    (tmp_repos / "r1").mkdir()
    assert get_repo_path("r1") == tmp_repos / "r1"

def test_delete_removes_dir(tmp_repos):
    (tmp_repos / "r1").mkdir()
    delete_repository("r1")
    assert not (tmp_repos / "r1").exists()

def test_delete_no_error_missing(tmp_repos):
    delete_repository("doesnotexist")

@patch("ingestion.repo_loader.git.Repo.clone_from")
def test_clone_success(mock_clone, tmp_repos):
    mock_clone.return_value = MagicMock()
    result = clone_repository("https://github.com/user/repo", "id1", "main")
    assert result == tmp_repos / "id1"

@patch("ingestion.repo_loader.git.Repo.clone_from")
def test_clone_skips_if_exists(mock_clone, tmp_repos):
    (tmp_repos / "id1").mkdir()
    clone_repository("https://github.com/user/repo", "id1", "main")
    mock_clone.assert_not_called()

@patch("ingestion.repo_loader.git.Repo.clone_from", side_effect=git.exc.GitCommandError("clone", 128))
def test_clone_raises_repo_not_found(mock_clone, tmp_repos):
    with pytest.raises(RepoNotFoundError):
        clone_repository("https://github.com/bad/url", "id2", "main")