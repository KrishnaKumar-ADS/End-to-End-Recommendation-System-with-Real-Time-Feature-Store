"""
DS19 — Backend Configuration
All config values for the FastAPI server in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}

# ─────────────────────────────────────────────────────────────
# PROJECT PATHS (relative to project root, not backend/)
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR       = BASE_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
FEATURES_DIR   = DATA_DIR / "features"
SEQUENCES_DIR  = DATA_DIR / "sequences"
SPLITS_DIR     = DATA_DIR / "splits"
RAW_DIR        = DATA_DIR / "raw"
MODELS_DIR     = BASE_DIR / "models" / "saved"
RETRIEVAL_DIR  = BASE_DIR / "retrieval"
LOGS_DIR       = BASE_DIR / "logs"

# ─────────────────────────────────────────────────────────────
# MODEL FILE PATHS
# ─────────────────────────────────────────────────────────────

TWO_TOWER_MODEL_PATH   = MODELS_DIR / "two_tower_best.pt"
TWO_TOWER_CONFIG_PATH  = MODELS_DIR / "two_tower_config.json"
FAISS_INDEX_PATH       = RETRIEVAL_DIR / "faiss_item.index"
FAISS_ITEM_MAP_PATH    = RETRIEVAL_DIR / "item_idx_map.npy"
LGBM_MODEL_PATH        = MODELS_DIR / "lgbm_ranker.pkl"

# ─────────────────────────────────────────────────────────────
# DATA FILE PATHS
# ─────────────────────────────────────────────────────────────

INTERACTIONS_FILE     = PROCESSED_DIR / "interactions.csv"
ITEM2IDX_FILE         = PROCESSED_DIR / "item2idx.json"
IDX2ITEM_FILE         = PROCESSED_DIR / "idx2item.json"
DATASET_META_FILE     = PROCESSED_DIR / "dataset_meta.json"
SEQUENCES_FILE        = SEQUENCES_DIR / "sequences.pkl"
MOVIES_FILE           = RAW_DIR / "movies.csv"
USER_FEATURES_FILE    = FEATURES_DIR / "user_features.parquet"
ITEM_FEATURES_FILE    = FEATURES_DIR / "item_features.parquet"

# ─────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS (must match training config)
# ─────────────────────────────────────────────────────────────

EMBED_DIM      = 64       # Alias for compatibility
HIDDEN_DIM     = 64       # Must match TwoTowerModel hidden_dim parameter
NUM_HEADS      = 2
NUM_BLOCKS     = 2
MAX_SEQ_LEN    = 50
PAD_TOKEN      = 0
DROPOUT        = 0.2      # Must match TwoTowerModel dropout parameter

# ─────────────────────────────────────────────────────────────
# INFERENCE CONFIG
# ─────────────────────────────────────────────────────────────

TOP_K_RETRIEVAL = 100
TOP_K_FINAL     = 10
DEFAULT_TOP_K   = 10

# ─────────────────────────────────────────────────────────────
# REDIS CONFIG
# ─────────────────────────────────────────────────────────────

REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB          = 0
REDIS_CACHE_TTL   = 300
REDIS_ENABLED     = _env_bool("REDIS_ENABLED", True)

# ─────────────────────────────────────────────────────────────
# API SERVER CONFIG
# ─────────────────────────────────────────────────────────────

API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", os.getenv("PORT", 8000)))
API_WORKERS  = 1
API_RELOAD   = False

# ─────────────────────────────────────────────────────────────
# LOGGING CONFIG
# ─────────────────────────────────────────────────────────────

API_LOG_FILE       = LOGS_DIR / "api_requests.jsonl"
LATENCY_LOG_FILE   = LOGS_DIR / "latency_log.csv"

# ─────────────────────────────────────────────────────────────
# DEVICE CONFIG
# ─────────────────────────────────────────────────────────────

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS (must match Week 5 feature engineering)
# ─────────────────────────────────────────────────────────────

RANKING_FEATURE_COLS = [
    "faiss_score",
    "faiss_rank_normalized",
    "item_log_popularity",
    "item_avg_rating",
    "item_rating_count_log",
    "item_recency_days",
    "item_year",
    "item_active_days",
    "user_interaction_count_log",
    "user_avg_rating",
    "user_recency_days",
    "user_seq_length",
    "genre_overlap",
    "is_same_top_genre",
    "mf_score",
]

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western", "IMAX", "(no genres listed)"
]