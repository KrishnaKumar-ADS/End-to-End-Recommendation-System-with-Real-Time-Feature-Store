"""Evaluation repository definitions and curated question bank for Week 15."""

from __future__ import annotations

from dataclasses import dataclass, field


ARCHITECTURE = "architecture"
CODE_EXPLANATION = "code_explanation"
BUG_TRACE = "bug_trace"
SECURITY = "security"


@dataclass(slots=True)
class EvalQuestion:
    text: str
    category: str
    expected_keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalRepo:
    name: str
    github_url: str
    branch: str
    language: str
    approx_files: int
    approx_chunks: int
    questions: list[EvalQuestion] = field(default_factory=list)


def _q(text: str, category: str, keywords: list[str]) -> EvalQuestion:
    return EvalQuestion(text=text, category=category, expected_keywords=keywords)


REQUESTS_QUESTIONS = [
    _q("Give a high-level overview of how requests handles a GET call from API entrypoint to final response.", ARCHITECTURE, ["Session", "PreparedRequest", "adapter", "Response"]),
    _q("Explain how connection pooling is implemented in requests.", ARCHITECTURE, ["HTTPAdapter", "urllib3", "pool", "PoolManager"]),
    _q("Describe the adapter architecture and the role of BaseAdapter vs HTTPAdapter.", ARCHITECTURE, ["BaseAdapter", "HTTPAdapter", "send", "mount"]),
    _q("How does Session differ from top-level requests.get helpers?", ARCHITECTURE, ["cookies", "headers", "state", "Session"]),
    _q("Explain redirect handling flow at architecture level.", ARCHITECTURE, ["redirect", "history", "resolve_redirects", "status_code"]),
    _q("What does PreparedRequest.prepare_url do and which edge cases does it handle?", CODE_EXPLANATION, ["scheme", "query", "requote", "url"]),
    _q("Explain Session.resolve_redirects behavior including redirect limits.", CODE_EXPLANATION, ["max_redirects", "history", "yield", "response"]),
    _q("What is Response.iter_content used for and when should it be preferred over Response.content?", CODE_EXPLANATION, ["stream", "chunk", "decode", "memory"]),
    _q("How is authentication prepared in PreparedRequest.prepare_auth?", CODE_EXPLANATION, ["auth", "AuthBase", "headers", "credentials"]),
    _q("Explain SSL verification handling in requests.", CODE_EXPLANATION, ["verify", "cert", "ssl", "urllib3"]),
    _q("Trace how an SSL handshake failure becomes a requests.ConnectionError or requests.SSLError.", BUG_TRACE, ["SSLError", "ConnectionError", "urllib3", "exception"]),
    _q("Trace where redirect loops are detected and converted into a terminal exception.", BUG_TRACE, ["TooManyRedirects", "history", "max_redirects", "raise"]),
    _q("A streamed response never releases connection back to pool. Trace responsible code path.", BUG_TRACE, ["stream", "close", "release_conn", "pool"]),
    _q("Trace chunked transfer decoding failure path to ChunkedEncodingError.", BUG_TRACE, ["chunked", "ProtocolError", "iter_content", "ChunkedEncodingError"]),
    _q("Trace how cookie state is updated across redirect responses.", BUG_TRACE, ["cookies", "redirect", "extract_cookies", "jar"]),
    _q("Can user-controlled values trigger header injection inside request-building flow?", SECURITY, ["header", "CRLF", "sanitize", "validation"]),
    _q("Analyze security impact of verify=False and where warnings are emitted.", SECURITY, ["verify", "MITM", "warning", "ssl"]),
    _q("Could redirect logic leak credentials across host boundaries?", SECURITY, ["redirect", "auth", "host", "rebuild_auth"]),
    _q("Identify places where URL credentials may be exposed in logs or errors.", SECURITY, ["url", "password", "log", "redact"]),
    _q("Assess protections against resource exhaustion via malicious redirect chains.", SECURITY, ["redirect", "limit", "history", "memory"]),
]


EXPRESS_QUESTIONS = [
    _q("Explain Express request routing from app.get registration to handler dispatch.", ARCHITECTURE, ["Router", "Layer", "dispatch", "stack"]),
    _q("Describe middleware chaining design and next() flow.", ARCHITECTURE, ["middleware", "next", "stack", "dispatch"]),
    _q("How do app.use and app.get differ in architecture and use-cases?", ARCHITECTURE, ["use", "get", "route", "middleware"]),
    _q("Explain Router mounting behavior and path prefixing semantics.", ARCHITECTURE, ["mount", "prefix", "router", "path"]),
    _q("How is 404 handling finalized when no route matches?", ARCHITECTURE, ["finalhandler", "404", "router", "fallthrough"]),
    _q("Explain res.json vs res.send behavior for object payloads.", CODE_EXPLANATION, ["json", "send", "Content-Type", "stringify"]),
    _q("How does express.json body parsing integrate into request handling?", CODE_EXPLANATION, ["body", "json", "middleware", "parser"]),
    _q("Explain error-handling middleware signature and invocation rules.", CODE_EXPLANATION, ["next", "error", "four arguments", "middleware"]),
    _q("How does express.static resolve and serve files?", CODE_EXPLANATION, ["static", "serve-static", "path", "cache"]),
    _q("Explain app.listen implementation path and host binding defaults.", CODE_EXPLANATION, ["listen", "server", "host", "port"]),
    _q("Trace why middleware mounted at /api might not execute for /api/users.", BUG_TRACE, ["mount", "path", "prefix", "router"]),
    _q("Trace res.redirect path for setting status code and Location header.", BUG_TRACE, ["redirect", "Location", "status", "header"]),
    _q("Trace synchronous route exception handling path to final 500 response.", BUG_TRACE, ["try", "catch", "Layer", "error"]),
    _q("Trace unhandled Promise rejection behavior in async route handlers.", BUG_TRACE, ["Promise", "async", "next", "error"]),
    _q("Trace route matching ambiguity when overlapping patterns are registered.", BUG_TRACE, ["route", "order", "path", "match"]),
    _q("Assess directory traversal protections in static file serving path.", SECURITY, ["traversal", "path", "normalize", "dotfiles"]),
    _q("Could malicious input trigger response splitting via header APIs?", SECURITY, ["header", "CRLF", "sanitize", "response splitting"]),
    _q("What security risks exist when req.params is used directly in DB queries?", SECURITY, ["injection", "params", "validation", "sanitize"]),
    _q("Assess open redirect risk pattern in res.redirect(req.query.url).", SECURITY, ["open redirect", "query", "whitelist", "validation"]),
    _q("Identify missing security defaults that production Express apps should add.", SECURITY, ["helmet", "rate limit", "cors", "headers"]),
]


GIN_QUESTIONS = [
    _q("Describe how Gin routes requests using trie/radix structures.", ARCHITECTURE, ["Engine", "tree", "node", "routing"]),
    _q("Explain HandlerChain execution model and middleware propagation.", ARCHITECTURE, ["HandlersChain", "Next", "Abort", "index"]),
    _q("How does Gin context pooling improve performance?", ARCHITECTURE, ["sync.Pool", "Context", "reuse", "allocation"]),
    _q("Compare gin.Default and gin.New design choices.", ARCHITECTURE, ["Default", "New", "Logger", "Recovery"]),
    _q("Explain panic recovery architecture in Gin.", ARCHITECTURE, ["Recovery", "panic", "recover", "middleware"]),
    _q("Explain ShouldBindJSON vs BindJSON behavior.", CODE_EXPLANATION, ["bind", "json", "error", "abort"]),
    _q("How are URL parameters extracted for routes like /users/:id?", CODE_EXPLANATION, ["params", "key", "value", "path"]),
    _q("What is gin.H and how is it used in JSON responses?", CODE_EXPLANATION, ["map", "json", "response", "helper"]),
    _q("Explain RouterGroup implementation and inherited middleware behavior.", CODE_EXPLANATION, ["RouterGroup", "basePath", "handlers", "group"]),
    _q("How do Context.Set and Context.Get pass values across middleware?", CODE_EXPLANATION, ["context", "keys", "set", "get"]),
    _q("Trace wildcard route conflict behavior causing unexpected matches.", BUG_TRACE, ["wildcard", "path", "node", "conflict"]),
    _q("Trace why writing response inside a goroutine can fail.", BUG_TRACE, ["goroutine", "writer", "race", "context"]),
    _q("Trace panic recovery when partial response has already been written.", BUG_TRACE, ["panic", "write", "header", "recovery"]),
    _q("Trace duplicated handler execution when Next is misused.", BUG_TRACE, ["Next", "index", "handlers", "middleware"]),
    _q("Trace request body mutation issues across middleware chain.", BUG_TRACE, ["body", "read", "reset", "handler"]),
    _q("What injection risks arise when binding unvalidated request structs?", SECURITY, ["bind", "validation", "input", "injection"]),
    _q("Which security headers are not enabled by default in Gin apps?", SECURITY, ["headers", "CSP", "X-Frame-Options", "middleware"]),
    _q("Assess file upload attack surface and default limits in Gin.", SECURITY, ["upload", "size", "memory", "multipart"]),
    _q("Can crafted URLs trigger pathological route matching overhead?", SECURITY, ["trie", "depth", "dos", "routing"]),
    _q("Assess sensitive query parameter exposure in logger middleware.", SECURITY, ["logger", "query", "redact", "privacy"]),
]


FASTAPI_QUESTIONS = [
    _q("Explain how FastAPI builds OpenAPI from route annotations and models.", ARCHITECTURE, ["openapi", "schema", "pydantic", "route"]),
    _q("Describe FastAPI dependency injection lifecycle with Depends.", ARCHITECTURE, ["Depends", "dependency", "resolve", "injection"]),
    _q("How does FastAPI execute sync handlers inside async server runtime?", ARCHITECTURE, ["async", "sync", "threadpool", "executor"]),
    _q("Explain request body validation flow with Pydantic v2.", ARCHITECTURE, ["pydantic", "validate", "model", "request"]),
    _q("Describe FastAPI WebSocket integration architecture.", ARCHITECTURE, ["websocket", "ASGI", "accept", "send"]),
    _q("What does startup lifespan do and where is it wired?", CODE_EXPLANATION, ["startup", "lifespan", "event", "app"]),
    _q("Explain BackgroundTasks execution timing relative to response.", CODE_EXPLANATION, ["background", "task", "response", "execution"]),
    _q("How do Path, Query, and Body differ in extraction/validation?", CODE_EXPLANATION, ["Path", "Query", "Body", "validation"]),
    _q("What does response_model enforce on returned payloads?", CODE_EXPLANATION, ["response_model", "filter", "serialization", "schema"]),
    _q("How is CORS middleware configured and applied in FastAPI?", CODE_EXPLANATION, ["CORS", "middleware", "origins", "headers"]),
    _q("Trace DB session leak risk in async dependency patterns.", BUG_TRACE, ["dependency", "yield", "session", "close"]),
    _q("Trace request validation error formatting path to 422 response.", BUG_TRACE, ["validation", "422", "handler", "error"]),
    _q("Trace background task exception handling after 200 response.", BUG_TRACE, ["background", "exception", "logging", "task"]),
    _q("Trace route ordering conflict between /users/me and /users/{id}.", BUG_TRACE, ["route", "order", "path", "regex"]),
    _q("Trace event loop lifecycle issues in async test client failures.", BUG_TRACE, ["event loop", "test client", "async", "runtime"]),
    _q("Assess API key auth timing-attack protection patterns.", SECURITY, ["api key", "timing", "compare_digest", "auth"]),
    _q("Assess file upload DoS risks and mitigation points.", SECURITY, ["upload", "size", "limit", "dos"]),
    _q("Analyze OAuth2 password grant security tradeoffs.", SECURITY, ["oauth2", "password", "token", "security"]),
    _q("Could path params enable traversal when used in file operations?", SECURITY, ["path", "traversal", "validation", "sanitize"]),
    _q("Identify missing request-size protections against JSON abuse.", SECURITY, ["body", "size", "limit", "middleware"]),
]


FULLSTACK_FASTAPI_TEMPLATE_QUESTIONS = [
    _q("Describe frontend-backend-database architecture and communication boundaries.", ARCHITECTURE, ["FastAPI", "frontend", "database", "docker"]),
    _q("Explain cross-stack authentication flow and token lifecycle.", ARCHITECTURE, ["JWT", "token", "auth", "backend"]),
    _q("Trace how frontend API client calls backend endpoints.", ARCHITECTURE, ["fetch", "client", "endpoint", "request"]),
    _q("Explain migration and schema version workflow in this template.", ARCHITECTURE, ["alembic", "migration", "revision", "schema"]),
    _q("Describe Docker Compose service orchestration for the template.", ARCHITECTURE, ["compose", "service", "backend", "frontend"]),
    _q("Trace user creation flow from route to DB write.", CODE_EXPLANATION, ["user", "create", "database", "crud"]),
    _q("Explain email verification mechanics and involved components.", CODE_EXPLANATION, ["email", "verification", "token", "task"]),
    _q("Explain item schema validation path before persistence.", CODE_EXPLANATION, ["schema", "validation", "item", "model"]),
    _q("How are DB sessions created and cleaned up?", CODE_EXPLANATION, ["session", "engine", "dependency", "pool"]),
    _q("How is auth state stored on frontend and attached to requests?", CODE_EXPLANATION, ["token", "storage", "authorization", "header"]),
    _q("Trace 401-after-login issue from frontend token to backend verifier.", BUG_TRACE, ["401", "token", "header", "verify"]),
    _q("Trace migration conflict error path for duplicate relation creation.", BUG_TRACE, ["alembic", "migration", "relation", "exists"]),
    _q("Trace missing verification email path and failure points.", BUG_TRACE, ["email", "task", "smtp", "queue"]),
    _q("Trace CORS failure path between browser and backend config.", BUG_TRACE, ["cors", "origin", "preflight", "middleware"]),
    _q("Trace silent auth expiry handling on frontend request layer.", BUG_TRACE, ["401", "interceptor", "refresh", "redirect"]),
    _q("Verify password hashing implementation and plaintext risk.", SECURITY, ["hash", "bcrypt", "password", "security"]),
    _q("Assess ownership checks preventing cross-user resource access.", SECURITY, ["owner", "authorization", "forbidden", "resource"]),
    _q("Assess SQL injection exposure in database query construction.", SECURITY, ["sql", "injection", "query", "parameter"]),
    _q("Assess secret management and accidental leakage risks.", SECURITY, ["secret", "env", "log", "expose"]),
    _q("Assess missing brute-force protections on auth endpoints.", SECURITY, ["rate limit", "auth", "brute force", "login"]),
]


EVAL_REPOS: list[EvalRepo] = [
    EvalRepo(
        name="psf/requests",
        github_url="https://github.com/psf/requests",
        branch="main",
        language="python",
        approx_files=150,
        approx_chunks=900,
        questions=REQUESTS_QUESTIONS,
    ),
    EvalRepo(
        name="expressjs/express",
        github_url="https://github.com/expressjs/express",
        branch="master",
        language="javascript",
        approx_files=180,
        approx_chunks=1100,
        questions=EXPRESS_QUESTIONS,
    ),
    EvalRepo(
        name="gin-gonic/gin",
        github_url="https://github.com/gin-gonic/gin",
        branch="master",
        language="go",
        approx_files=200,
        approx_chunks=1400,
        questions=GIN_QUESTIONS,
    ),
    EvalRepo(
        name="fastapi/fastapi",
        github_url="https://github.com/fastapi/fastapi",
        branch="master",
        language="python",
        approx_files=300,
        approx_chunks=2000,
        questions=FASTAPI_QUESTIONS,
    ),
    EvalRepo(
        name="tiangolo/full-stack-fastapi-template",
        github_url="https://github.com/tiangolo/full-stack-fastapi-template",
        branch="master",
        language="python+typescript",
        approx_files=400,
        approx_chunks=2800,
        questions=FULLSTACK_FASTAPI_TEMPLATE_QUESTIONS,
    ),
]


def get_repo_by_name(name: str) -> EvalRepo | None:
    lookup = name.strip().lower()
    for repo in EVAL_REPOS:
        if repo.name.lower() == lookup:
            return repo
    return None


def get_questions_by_category(repo: EvalRepo, category: str) -> list[EvalQuestion]:
    normalized = category.strip().lower()
    return [question for question in repo.questions if question.category == normalized]
