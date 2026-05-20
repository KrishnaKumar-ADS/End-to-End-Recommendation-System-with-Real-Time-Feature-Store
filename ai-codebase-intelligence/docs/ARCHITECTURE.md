# Architecture

## Overview

The platform turns a repository into an indexed knowledge graph and supports grounded natural-language question answering.

Main planes:

- Ingestion plane: clone, parse, chunk, embed, index
- Retrieval plane: hybrid search and graph expansion
- Reasoning plane: task routing, prompt assembly, answer generation
- Presentation plane: Next.js frontend with chat and graph views

## Runtime components

- FastAPI backend
- Celery worker
- Redis (broker and transient cache)
- PostgreSQL (metadata)
- Qdrant (vector similarity search)
- Neo4j (graph storage and traversals)
- Next.js frontend
- Nginx (production reverse proxy and TLS)

## Ingestion flow

1. Client calls POST /api/v1/ingest
2. Backend stores repository record and enqueues Celery task
3. Worker clones repository and scans source files
4. Parsers extract symbols and metadata
5. Chunker produces semantic code chunks
6. Embeddings are generated and written to Qdrant
7. Graph edges and nodes are written to Neo4j
8. Status endpoint reflects progress until completed

## Query flow

1. Client sends question to POST /api/v1/ask
2. Backend retrieves relevant chunks using hybrid retrieval
3. Graph expansion adds call-path context
4. Prompt is assembled from chunks, graph, and conversation history
5. LLM router selects provider/model by task type
6. Response is streamed or returned as JSON
7. Sources and graph path are attached for explainability

## Data model boundaries

PostgreSQL:

- Repository
- SourceFile
- CodeChunk

Qdrant:

- Chunk vectors
- Metadata payload for file path, symbol, language, line ranges

Neo4j:

- Function/Class/File nodes
- CALLS, IMPORTS, INHERITS_FROM and related edges

## Reliability and safety

- Production startup validates required environment variables
- Service health checks are defined in compose manifests
- Rate limiting is configured at nginx for expensive ask endpoint
- Budget guards in cost tracker prevent runaway evaluation spend

## Evaluation subsystem

Week 15 introduced evaluation modules:

- eval framework orchestrator
- latency benchmark
- cost tracker
- quality scorer (LLM-as-judge)
- report generator

Week 16 adds provider comparison reporting and release documentation artifacts.
