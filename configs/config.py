# DS19 — Central Configuration File

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
SEQUENCES_DIR  = DATA_DIR / "sequences"
FEATURES_DIR   = DATA_DIR / "features"
SPLITS_DIR     = DATA_DIR / "splits"
MODELS_DIR     = BASE_DIR / "models" / "saved"
LOGS_DIR       = BASE_DIR / "logs"

# ─── Dataset ─────────────────────────────────────────────────
DATASET_NAME   = "ml-25m"
RATINGS_FILE   = RAW_DIR / "ratings.csv"
MOVIES_FILE    = RAW_DIR / "movies.csv"
TAGS_FILE      = RAW_DIR / "tags.csv"

# ─── Preprocessing ───────────────────────────────────────────
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
POSITIVE_RATING_THRESHOLD = 4.0

# ─── Sequence ────────────────────────────────────────────────
MAX_SEQ_LEN    = 50
PAD_TOKEN      = 0

# ─── Training ────────────────────────────────────────────────
SEED           = 42
BATCH_SIZE     = 256
LEARNING_RATE  = 1e-3
EPOCHS         = 20
EMBEDDING_DIM  = 64

# ─── FAISS ───────────────────────────────────────────────────
TOP_K_CANDIDATES = 100
TOP_K_FINAL      = 10

# ─── Redis ───────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))

# ─── MLflow ──────────────────────────────────────────────────
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ─── API ─────────────────────────────────────────────────────
API_HOST       = "0.0.0.0"
API_PORT       = 8000
