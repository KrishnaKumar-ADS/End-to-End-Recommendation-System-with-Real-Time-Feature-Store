# End-to-End Recommendation System with Real-Time Feature Store

A production-grade recommender system with a real-time feature store, online/offline feature serving, A/B testing framework, and full monitoring — simulating how Netflix/Spotify deploy recommenders.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐
│  Client      │───▶│  FastAPI Gateway  │───▶│  A/B Test Router  │
└─────────────┘    └──────────────────┘    └─────────┬─────────┘
                                                      │
                           ┌──────────────────────────┼──────────────────────────┐
                           │                          │                          │
                   ┌───────▼───────┐         ┌───────▼────────┐        ┌───────▼────────┐
                   │  Retrieval     │         │  Feature Store  │        │  Monitoring     │
                   │  (FAISS ANN)   │         │  (Feast+Redis)  │        │  (Prometheus)   │
                   └───────┬───────┘         └───────┬────────┘        └───────┬────────┘
                           │                          │                          │
                   ┌───────▼───────┐         ┌───────▼────────┐        ┌───────▼────────┐
                   │  Re-Ranking    │◀────────│  Online Store   │        │  Grafana        │
                   │  (LightGBM)    │         │  (Redis)        │        │  Dashboard      │
                   └───────────────┘         └────────────────┘        └────────────────┘
```

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── prepare_data.py        # MovieLens 25M download, preprocess, split
│   │   └── event_stream.py        # Simulated real-time user event stream
│   ├── models/
│   │   ├── two_tower.py           # Two-Tower retrieval model (PyTorch)
│   │   ├── faiss_index.py         # FAISS ANN index for candidate generation
│   │   └── reranker.py            # LightGBM re-ranker
│   ├── feature_store/
│   │   ├── feature_definitions.py # Feast feature definitions
│   │   └── store_manager.py       # Feature store manager (Feast + Redis)
│   ├── serving/
│   │   ├── api.py                 # FastAPI recommendation API
│   │   └── ab_testing.py          # A/B testing framework
│   ├── mlops/
│   │   ├── model_registry.py      # MLflow model registry integration
│   │   └── train_pipeline.py      # End-to-end training pipeline
│   └── monitoring/
│       └── dashboard.py           # Grafana dashboard generator
├── feature_repo/
│   └── feature_store.yaml         # Feast configuration
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml         # Scrape config
│   │   └── alerts.yml             # Alert rules
│   └── grafana/
│       ├── dashboards/
│       │   └── recsys.json        # Pre-built dashboard
│       └── provisioning/          # Auto-provisioning configs
├── notebooks/
│   └── ab_testing_simulation.ipynb
├── tests/
│   └── test_recsys.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Components

### 1. Two-Tower Retrieval Model

- User tower + Item tower producing normalized embeddings
- Trained with BCE loss on implicit feedback (rating ≥ 4 → positive)
- Exports user/item embeddings for FAISS indexing

### 2. FAISS ANN Index

- Supports `Flat` (brute-force), `IVFFlat`, and `HNSW` index types
- Sub-millisecond retrieval of top-K candidates from millions of items
- Serializable for deployment

### 3. LightGBM Re-Ranker

- Takes candidates from retrieval + rich features from feature store
- Features: retrieval score, user profile, item profile, genre, real-time activity
- Pointwise ranking with early stopping

### 4. Feature Store (Feast + Redis)

- **Offline store**: Parquet files for training (point-in-time correct joins)
- **Online store**: Redis for low-latency feature serving
- **Real-time features**: Event-driven updates via Redis (1h sliding window)
- Feature definitions in Feast format

### 5. A/B Testing Framework

- Deterministic user-to-group assignment (hash-based)
- Chi-squared test for binary metrics (CTR)
- Welch's t-test for continuous metrics
- Power analysis & minimum sample size calculation
- Sequential testing with p-value monitoring

### 6. Monitoring (Prometheus + Grafana)

Metrics tracked:
| Metric | Type | Description |
|--------|------|-------------|
| `recsys_request_total` | Counter | Request count by endpoint & A/B group |
| `recsys_request_latency_seconds` | Histogram | End-to-end latency |
| `recsys_retrieval_latency_seconds` | Histogram | FAISS retrieval time |
| `recsys_rerank_latency_seconds` | Histogram | Re-ranker inference time |
| `recsys_feature_fetch_latency_seconds` | Histogram | Feature store fetch time |
| `recsys_item_coverage` | Gauge | Unique items recommended |
| `recsys_ctr_events` | Counter | CTR tracking by A/B group |

### 7. MLflow Model Registry

- Tracks experiments for both retrieval and ranking models
- Model versioning for A/B test deployments
- Metric logging (train/val loss, AUC, log-loss)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
python -m src.mlops.train_pipeline
```

This will:

- Download MovieLens 25M (first run only)
- Preprocess data with temporal splits
- Train Two-Tower model
- Build FAISS index
- Train LightGBM re-ranker
- Log everything to MLflow

### 3. Start the API (standalone)

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

### 4. Start Full Stack (Docker)

```bash
docker-compose up -d
```

Services:

- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / recsys2024)

## API Endpoints

| Method | Endpoint     | Description                   |
| ------ | ------------ | ----------------------------- |
| `POST` | `/recommend` | Get top-K recommendations     |
| `POST` | `/event`     | Ingest user interaction event |
| `GET`  | `/health`    | Health check                  |
| `GET`  | `/metrics`   | Prometheus metrics            |
| `GET`  | `/ab/config` | View A/B test config          |
| `POST` | `/ab/config` | Update A/B test config        |

### Example Request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_idx": 42, "top_k": 10}'
```

### Example Response

```json
{
  "user_idx": 42,
  "recommendations": [
    { "item_idx": 1234, "retrieval_score": 0.95, "rerank_score": 0.87 },
    { "item_idx": 567, "retrieval_score": 0.93, "rerank_score": 0.82 }
  ],
  "ab_group": "control",
  "retrieval_latency_ms": 1.23,
  "rerank_latency_ms": 2.45,
  "total_latency_ms": 5.67
}
```

## Feature Store Details

### Feature Definitions

| Feature View    | Entity     | Features                                                      | TTL    | Store            |
| --------------- | ---------- | ------------------------------------------------------------- | ------ | ---------------- |
| `user_profile`  | `user_idx` | rating_count, rating_mean, rating_std, positive_ratio         | 1 day  | Online + Offline |
| `item_profile`  | `item_idx` | rating_count, rating_mean, rating_std, positive_ratio, genres | 1 day  | Online + Offline |
| `user_realtime` | `user_idx` | recent_click_count, recent_view_count, avg_view_duration      | 1 hour | Online (Redis)   |

### Point-in-Time Correctness

The feature store ensures no data leakage by performing backward-looking joins:

- Training features are computed only from data before the event timestamp
- Online features have TTL-based expiry
- Real-time features use sliding windows

## Dataset

- **MovieLens 25M**: 25M ratings from 162K users on 62K movies
- **Simulated Event Stream**: Real-time user interactions (click, view, rating, search)
- **Temporal Split**: Last 30 days → test, previous 30 days → val, rest → train

## Tech Stack

| Component        | Technology              |
| ---------------- | ----------------------- |
| Retrieval Model  | PyTorch (Two-Tower)     |
| ANN Search       | FAISS                   |
| Re-Ranker        | LightGBM                |
| Feature Store    | Feast + Redis           |
| Model Registry   | MLflow                  |
| API              | FastAPI + Uvicorn       |
| Monitoring       | Prometheus + Grafana    |
| Containerization | Docker + Docker Compose |
| CI/CD            | GitHub Actions          |

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## License

MIT
