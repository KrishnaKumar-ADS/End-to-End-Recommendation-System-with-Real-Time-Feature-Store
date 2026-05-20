# AI Codebase Intelligence Platform

Ask questions about any repository and get grounded answers with code citations.

This project combines semantic retrieval, graph expansion, and task-aware LLM routing to support code explanation, bug tracing, architecture understanding, and security-oriented analysis.

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-green)](https://github.com/your-username/ai-codebase-intelligence/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this does

- Ingests repositories from GitHub.
- Parses symbols and dependencies.
- Stores vectors in Qdrant and graph structure in Neo4j.
- Answers questions with grounded context and source references.
- Exposes graph and search endpoints for interactive exploration.

## Architecture

```text
Frontend (Next.js)
  -> FastAPI backend
      -> Celery worker + Redis
      -> PostgreSQL metadata
      -> Qdrant vectors
      -> Neo4j graph
      -> LLM providers (Qwen via OpenRouter, Gemini embeddings)
```

Ingestion flow:

```text
GitHub URL -> clone -> scan -> parse -> chunk -> embed -> index
```

Query flow:

```text
Question -> embed -> retrieve -> graph expand -> prompt -> route -> answer
```

## Quick start

Prerequisites:

- Docker + Docker Compose
- Git
- API keys: GEMINI_API_KEY and OPENROUTER_API_KEY

```bash
git clone https://github.com/your-username/ai-codebase-intelligence
cd ai-codebase-intelligence
cp .env.example .env
# Fill keys in .env
make dev
```

Services:

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs
- Qdrant: http://localhost:6333/dashboard
- Neo4j: http://localhost:7474

## Provider strategy

- qwen/qwen-2.5-coder-32b-instruct: code explanation and bug tracing
- qwen/qwen-max: architecture reasoning and security-heavy prompts
- Gemini text-embedding-004 (or gemini-embedding-001): embeddings

Approximate pricing assumptions used in reports:

| Model                            | Input / 1M | Output / 1M |
| -------------------------------- | ---------: | ----------: |
| qwen/qwen-2.5-coder-32b-instruct |       0.18 |        0.18 |
| qwen/qwen-max                    |       1.60 |        6.40 |
| gemini-2.0-flash                 |      0.075 |        0.30 |
| text-embedding-004               |       0.00 |        0.00 |

## Benchmarks and reports

Week 15 and Week 16 reporting outputs:

- data/benchmarks/BENCHMARK_REPORT.md
- data/benchmarks/BENCHMARK_REPORT.html
- data/benchmarks/PROVIDER_COMPARISON.md
- data/benchmarks/PROVIDER_COMPARISON.html

Generate comparison report:

```bash
make compare-providers
```

## API reference

Core endpoints:

- POST /api/v1/ingest
- GET /api/v1/status/{task_id}
- GET /api/v1/search
- POST /api/v1/ask
- GET /api/v1/graph/{repo_id}
- GET /health

Use interactive docs at /docs for schemas and examples.

## Development commands

```bash
make test
make lint
make eval-quick
make eval
make smoke-test
make smoke-test-week16
```

## Deployment and operations

Detailed docs:

- docs/DEPLOYMENT.md
- docs/ARCHITECTURE.md
- docs/BRANCH_PROTECTION.md
- docs/WHAT_I_LEARNED.md

## Release notes

See CHANGELOG.md for version history and Week-by-Week milestones.

## Contributing

- Open issues with templates in .github/ISSUE_TEMPLATE
- Use pull request checklist in .github/PULL_REQUEST_TEMPLATE.md
- Ensure tests and lint pass before requesting review

## License

MIT. See LICENSE.
