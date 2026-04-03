---
title: Recommender API
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Production-Style Movie Recommendation System

This is an end-to-end recommendation platform built around a hybrid retrieval + ranking architecture and deployed as a fullstack application.

The project combines:

- Offline data engineering and feature generation
- Neural retrieval (Two-Tower + FAISS)
- Learning-to-rank (LightGBM LambdaRank)
- Optional online adaptation (Thompson Sampling bandits)
- A/B experimentation and monitoring
- FastAPI serving backend
- React dashboard frontend

The goal is to demonstrate a complete recommender lifecycle from raw interactions to online inference, feedback capture, and monitoring.

## Live Demo

- Hugging Face Space: https://huggingface.co/spaces/krishnakumar19/ds19-recommender-api
- Hugging face repository: https://huggingface.co/spaces/krishnakumar19/ds19-recommender-api/tree/main

## 1) Project Objective

This project solves a practical recommendation problem:

- Input: user interaction history and item metadata
- Output: relevant top-k movie recommendations in real time

Instead of using one model for everything, DS19 follows a two-stage architecture:

- Stage 1 (Retrieval): quickly narrow the full catalog to strong candidates
- Stage 2 (Ranking): deeply score and reorder candidates for precision

This mirrors production recommender stacks where speed and quality must both be optimized.

## 2) End-to-End Architecture

### 2.1 Offline Layer (Data + Model Preparation)

1. Load and validate source data.
2. Build user/item index mappings.
3. Construct user interaction sequences.
4. Engineer user/item/context features.
5. Train retrieval and ranking models.
6. Export artifacts needed for low-latency serving.

### 2.2 Online Layer (Serving + Feedback)

1. Receive recommendation request via API.
2. Build user representation from interaction sequence.
3. Retrieve candidate items using FAISS ANN search.
4. Build ranking feature matrix for candidates.
5. Predict ranking scores with LightGBM.
6. Optionally apply bandit reranking.
7. Return final top-k recommendations.
8. Record feedback and update online components.

### 2.3 Monitoring + Experimentation

1. Poll `/health` for component-level status.
2. Track latency and cache hit metrics.
3. Split traffic by A/B variants.
4. Compare CTR and significance statistics.

## 3) Core Technical Components

### 3.1 Retrieval Model

- Two-Tower encoder computes user and item embeddings.
- FAISS stores item embeddings for nearest-neighbor search.
- Retrieval is optimized for speed and recall.

### 3.2 Ranking Model

- LightGBM LambdaRank reranks retrieved candidates.
- Features include retrieval signals, user features, item features, and interaction features.
- Ranking is optimized for top-k quality.

### 3.3 Bandits (Optional Online Adaptation)

- Thompson Sampling updates item-level posteriors from feedback.
- Can blend with ranker outputs for exploration/exploitation.
- Gracefully degrades when Redis is disabled or unavailable.

### 3.4 API Layer

- FastAPI provides low-latency endpoints.
- Includes health/readiness/liveness probes.
- Exposes metrics for observability.

### 3.5 Frontend Layer

- React + TypeScript frontend for recommendation UI and monitoring.
- Includes A/B dashboard and health visualization.
- Built with Vite and served from the same Docker runtime in Hugging Face Spaces.

## 4) Online Recommendation Flow (Detailed)

When `GET /recommend` is called:

1. Validate `user_id` and `top_k`.
2. Attempt cache lookup (if Redis enabled).
3. Build/lookup user sequence and pad to fixed length.
4. Encode sequence with Two-Tower user tower.
5. Query FAISS index to retrieve candidate items.
6. Fetch in-memory feature tables for user and candidates.
7. Construct ranking feature matrix.
8. Run LightGBM prediction.
9. Normalize and sort scores.
10. Optionally rerank with bandit service.
11. Map item IDs to titles/genres.
12. Return typed response payload.
13. Emit telemetry and request metrics.

When `POST /feedback` is called:

1. Persist feedback event log.
2. Invalidate cache for affected user.
3. Update bandit posterior if enabled.
4. Log A/B conversion event.

## 5) Health Model Semantics

Health endpoint returns:

- `components.models`: model loading and serving status
- `components.feast` (feature layer): in-memory feature store status
- `components.redis`: cache service status

Important behavior in single-container cloud runtimes:

- If Redis is intentionally disabled (`REDIS_ENABLED=false`), Redis reports `disabled`.
- Overall system can still report `healthy` when core serving components are healthy.

## 6) API Endpoints

### Recommendation APIs

- `GET /recommend?user_id=<int>&top_k=<int>`
- `GET /recommend/by-movie?movie_title=<str>&top_k=<int>`
- `GET /recommend/movie-suggestions?query=<str>&limit=<int>`

### Feedback + Experiment APIs

- `POST /feedback`
- `GET /ab/report`

### System + Monitoring APIs

- `GET /health`
- `GET /ready`
- `GET /live`
- `GET /metrics`
- `GET /docs`

## 7) Final Project Structure (GitHub-Clean)

The structure below reflects the intended push layout and core modules.
Files/folders starting with `week` and `WEEK_VISE/` are excluded from GitHub by policy.

```text
DS19/
├─ backend/
│  ├─ app/
│  │  ├─ api/                # HTTP route handlers
│  │  ├─ core/               # config + model loading
│  │  ├─ middleware/         # request/latency logging middleware
│  │  ├─ schemas/            # pydantic request/response models
│  │  ├─ services/           # retrieval/ranking/pipeline/cache logic
│  │  ├─ metrics.py          # prometheus metrics definitions
│  │  └─ main.py             # FastAPI app entrypoint
│  └─ requirements.txt
├─ data/
│  ├─ download_dataset.py
│  ├─ generate_processed_data.py
│  ├─ raw/                   # source/raw artifacts (selective)
│  ├─ processed/             # processed mappings/meta (selective)
│  ├─ features/              # feature tables used at inference
│  ├─ sequences/             # sequence artifacts used at inference
│  ├─ splits/
│  └─ events/
├─ feature_store/
│  ├─ feature_repo/
│  ├─ pipelines/
│  ├─ services/
│  ├─ training/
│  └─ tests/
├─ frontend/
│  ├─ src/
│  │  ├─ api/
│  │  ├─ components/
│  │  ├─ hooks/
│  │  ├─ pages/
│  │  └─ types/
│  ├─ public/
│  ├─ package.json
│  └─ vite.config.ts
├─ mlops/
│  ├─ ab_testing/
│  ├─ bandits/
│  ├─ mlflow_setup/
│  ├─ reports/
│  └─ ci_cd/
├─ models/
│  ├─ two_tower/
│  ├─ ranking/
│  ├─ matrix_factorization/
│  ├─ transformer/
│  └─ saved/                 # exported model artifacts (selective)
├─ retrieval/
│  ├─ retrieve.py
│  ├─ faiss_index.py
│  └─ faiss artifacts
├─ tests/
│  ├─ unit/
│  └─ integration/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ requirements-dev.txt
├─ script_deploy_hf_space.py
└─ README.md
```

## 8) Tooling and Stack

- Language: Python 3.10, TypeScript
- Backend: FastAPI, Uvicorn, Pydantic
- Frontend: React, React Query, Vite, Tailwind
- ML: PyTorch, LightGBM, FAISS, scikit-learn
- MLOps: MLflow, A/B logging, bandit feedback loops
- Cache/Online state: Redis (optional in cloud runtimes)
- Deployment: Docker, Hugging Face Spaces (Docker SDK)

## 9) Local Development Setup

### 9.1 Backend Setup

1. Create and activate Python environment.
2. Install dependencies from `requirements.txt`.
3. Ensure required artifacts exist in:
   - `data/processed/`
   - `data/features/`
   - `data/sequences/`
   - `models/saved/`
   - `retrieval/`
4. Start backend:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

5. Open API docs at `http://localhost:8000/docs`.

### 9.2 Frontend Setup

1. Move to `frontend/`.
2. Install dependencies.
3. Start Vite dev server.

```bash
npm install
npm run dev
```

4. Open UI at `http://localhost:5173`.

### 9.3 Fullstack Docker (Local)

Use the provided Docker assets to run integrated services where needed.

## 10) Hugging Face Spaces Deployment

The project includes automated deployment helper:

- `script_deploy_hf_space.py`

Typical deployment command:

```bash
python script_deploy_hf_space.py --clean
```

What it does:

1. Creates/uses a Docker Space repo.
2. Uploads only allow-listed runtime files.
3. Retries transient 5xx upload errors.
4. Optionally cleans old files before upload.

## 11) GitHub Push Policy Used in This Project

To keep the repository focused and maintainable:

- Do not push `WEEK_VISE/`.
- Do not push files/folders whose names start with `week` (case-insensitive).
- Do not push non-essential empty placeholder files.
- Avoid committing local runtime artifacts, virtual environments, caches, and logs.

These rules are enforced through `.gitignore` and curated staging.

## 12) Current Status

Completed:

- End-to-end retrieval + ranking API
- Frontend integration and dashboarding
- Health/readiness/liveness and metrics
- Redis-disabled cloud fallback semantics
- Hugging Face Docker Space deployment flow

In progress / extendable:

- richer online learning policies
- stronger experiment automation
- additional ranking feature families
- continuous retraining automation

---

If you are onboarding to this project, start with:

1. `backend/app/main.py` (application entrypoint)
2. `backend/app/services/pipeline_service.py` (serving pipeline)
3. `backend/app/core/model_loader.py` (artifact loading)
4. `frontend/src/pages/ABDashboard.tsx` (health + experiment UI)
5. `script_deploy_hf_space.py` (deployment automation)
