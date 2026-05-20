import pytest
from ingestion.chunkers.treesitter_chunker import chunk_with_treesitter

# ── JavaScript tests ──────────────────────────────────────────

JS_CODE = """
function authenticateUser(username, password) {
    const user = findUser(username);
    if (!user) return null;
    return verifyPassword(password, user.hash);
}

const generateToken = (userId) => {
    return jwt.sign({ userId }, SECRET);
};

class AuthController {
    constructor(authService) {
        this.authService = authService;
    }

    async login(req, res) {
        const result = await this.authService.login(req.body);
        res.json(result);
    }
}
"""

def test_js_extracts_functions():
    chunks = chunk_with_treesitter(JS_CODE, "auth.js", "javascript")
    names = [c.name for c in chunks]
    assert "authenticateUser" in names

def test_js_extracts_class():
    chunks = chunk_with_treesitter(JS_CODE, "auth.js", "javascript")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert any(c.name == "AuthController" for c in classes)

def test_js_has_correct_language():
    chunks = chunk_with_treesitter(JS_CODE, "auth.js", "javascript")
    assert all(c.language == "javascript" for c in chunks)

def test_js_line_numbers_valid():
    chunks = chunk_with_treesitter(JS_CODE, "auth.js", "javascript")
    for c in chunks:
        assert c.start_line > 0
        assert c.end_line >= c.start_line

# ── TypeScript tests ──────────────────────────────────────────

TS_CODE = """
interface User {
    id: number;
    username: string;
}

function hashPassword(plain: string): string {
    return bcrypt.hash(plain, 10);
}

class UserService {
    private db: Database;

    constructor(db: Database) {
        this.db = db;
    }

    async getUser(id: number): Promise<User> {
        return this.db.findById(id);
    }
}
"""

def test_ts_extracts_functions():
    chunks = chunk_with_treesitter(TS_CODE, "user.ts", "typescript")
    names = [c.name for c in chunks]
    assert "hashPassword" in names

def test_ts_extracts_class():
    chunks = chunk_with_treesitter(TS_CODE, "user.ts", "typescript")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert any(c.name == "UserService" for c in classes)

# ── Go tests ─────────────────────────────────────────────────

GO_CODE = """
package main

import "fmt"

func authenticateUser(username string, password string) bool {
    user := findUser(username)
    return verifyPassword(password, user.Hash)
}

func generateToken(userID int) string {
    return signJWT(userID)
}
"""

def test_go_extracts_functions():
    chunks = chunk_with_treesitter(GO_CODE, "auth.go", "go")
    names = [c.name for c in chunks]
    assert "authenticateUser" in names
    assert "generateToken" in names

# ── Fallback tests ────────────────────────────────────────────

def test_unsupported_language_falls_back():
    code = "SELECT * FROM users WHERE id = 1;\nSELECT * FROM posts;\n" * 30
    chunks = chunk_with_treesitter(code, "query.sql", "sql")
    # Should fall back to line chunker and return blocks
    assert len(chunks) > 0