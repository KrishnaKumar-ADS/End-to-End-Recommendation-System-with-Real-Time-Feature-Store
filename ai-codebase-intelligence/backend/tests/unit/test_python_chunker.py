from ingestion.chunkers.python_chunker import (
    chunk_python,
    extract_imports,
    extract_function_calls,
)

# ── Sample Python code used across all tests ──────────────────

SAMPLE_CODE = '''
import os
import hashlib
from pathlib import Path

def verify_password(plain: str, hashed: str) -> bool:
    """Check if plain password matches the hash."""
    return hashlib.checkpw(plain.encode(), hashed.encode())

async def generate_token(user_id: int) -> str:
    """Generate a JWT token for a user."""
    payload = {"user_id": user_id}
    return encode(payload)

class AuthService:
    """Handles all authentication logic."""

    def __init__(self, db):
        """Initialize with database connection."""
        self.db = db

    def login(self, username: str, password: str):
        """Authenticate a user."""
        user = self.db.get_user(username)
        return verify_password(password, user.password_hash)

    @staticmethod
    def logout(token: str) -> bool:
        return invalidate(token)
'''

# ── chunk extraction tests ────────────────────────────────────

def test_extracts_top_level_functions():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    names = [c.name for c in chunks]
    assert "verify_password" in names
    assert "generate_token" in names


def test_extracts_class():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert len(classes) == 1
    assert classes[0].name == "AuthService"


def test_extracts_methods():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    methods = [c for c in chunks if c.chunk_type in ("method", "async_method")]
    method_names = [c.name for c in methods]
    assert "login" in method_names
    assert "__init__" in method_names
    assert "logout" in method_names


def test_method_has_correct_parent():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    login = next(c for c in chunks if c.name == "login")
    assert login.parent_name == "AuthService"
    assert login.display_name == "AuthService.login"


def test_async_function_detected():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    token_func = next(c for c in chunks if c.name == "generate_token")
    assert "async" in token_func.chunk_type


def test_docstrings_extracted():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    verify = next(c for c in chunks if c.name == "verify_password")
    assert "Check if plain password" in verify.docstring


def test_class_docstring_extracted():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    auth_class = next(c for c in chunks if c.chunk_type == "class")
    assert "authentication" in auth_class.docstring.lower()


def test_decorator_extracted():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    logout = next(c for c in chunks if c.name == "logout")
    assert "staticmethod" in logout.decorators


def test_line_numbers_correct():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    verify = next(c for c in chunks if c.name == "verify_password")
    assert verify.start_line > 0
    assert verify.end_line >= verify.start_line


def test_content_contains_def():
    chunks = chunk_python(SAMPLE_CODE, "auth.py")
    verify = next(c for c in chunks if c.name == "verify_password")
    assert "def verify_password" in verify.content


def test_syntax_error_returns_empty():
    bad_code = "def broken(:\n    pass"
    chunks = chunk_python(bad_code, "bad.py")
    assert chunks == []


def test_empty_file_returns_empty():
    chunks = chunk_python("", "empty.py")
    assert chunks == []


# ── import extraction tests ───────────────────────────────────

def test_extract_imports():
    imports = extract_imports(SAMPLE_CODE)
    assert "os" in imports
    assert "hashlib" in imports
    assert "pathlib" in imports


def test_extract_imports_from_style():
    code = "from fastapi import FastAPI, Depends\nimport uvicorn"
    imports = extract_imports(code)
    assert "fastapi" in imports
    assert "uvicorn" in imports


# ── function call extraction tests ───────────────────────────

def test_extract_function_calls():
    code = '''
def login():
    user = get_user("admin")
    result = verify_password(user, "pass")
    return generate_token(user.id)
'''
    calls = extract_function_calls(code)
    assert "get_user" in calls
    assert "verify_password" in calls
    assert "generate_token" in calls