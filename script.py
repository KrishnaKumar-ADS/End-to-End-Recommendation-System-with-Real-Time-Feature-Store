import os
import json
from pathlib import Path  # ✅ ADD THIS
# ─────────────────────────────────────────────────────────────
# DEFINE FOLDER STRUCTURE
# ─────────────────────────────────────────────────────────────

folders = [
    # DATA LAYER
    "data/raw",
    "data/processed",
    "data/sequences",
    "data/events",
    "data/splits",
    "data/features",

    # FEATURE STORE
    "feature_store/feature_repo",
    "feature_store/offline_store",
    "feature_store/online_store",

    # MODELS
    "models/matrix_factorization",
    "models/transformer",
    "models/two_tower",
    "models/ranking",
    "models/saved",

    # RETRIEVAL
    "retrieval",

    # BANDITS
    "bandits",

    # BACKEND
    "backend/app/api",
    "backend/app/services",
    "backend/app/schemas",
    "backend/app/middleware",

    # FRONTEND
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/services",
    "frontend/src/hooks",
    "frontend/public",

    # MLOPS
    "mlops/mlflow",
    "mlops/ci_cd",
    "mlops/docker",

    # MONITORING
    "monitoring",

    # EXPERIMENTS
    "experiments",

    # NOTEBOOKS
    "notebooks/week1",
    "notebooks/week2",
    "notebooks/week3",
    "notebooks/week4",
    "notebooks/week5",
    "notebooks/week6",
    "notebooks/week7",
    "notebooks/week8",
    "notebooks/week9",
    "notebooks/week10",

    # LOGS
    "logs",

    # CONFIGS
    "configs",

    # TESTS
    "tests/unit",
    "tests/integration",
]

# ─────────────────────────────────────────────────────────────
# DEFINE FILES TO BOOTSTRAP
# ─────────────────────────────────────────────────────────────

files = {
    # Root files
    "requirements.txt": """# DS19 Core Dependencies
# Install with: pip install -r requirements.txt

# Data
pandas==2.1.4
numpy==1.26.2
scipy==1.11.4

# ML
torch==2.1.2
scikit-learn==1.3.2
lightgbm==4.1.0

# Retrieval
faiss-cpu==1.7.4

# Feature Store
feast==0.35.0

# Backend
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
httpx==0.25.2

# MLOps
mlflow==2.8.1

# Monitoring
prometheus-client==0.18.0

# Utils
python-dotenv==1.0.0
pydantic==2.5.0
tqdm==4.66.1
loguru==0.7.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
""",

    ".env": """# DS19 Environment Variables
PROJECT_NAME=DS19_RecSys
DATA_DIR=data
MODEL_DIR=models/saved
LOG_DIR=logs
REDIS_HOST=localhost
REDIS_PORT=6379
MLFLOW_TRACKING_URI=http://localhost:5000
""",

    ".gitignore": """# Python
__pycache__/
*.py[cod]
*.egg
*.egg-info/
dist/
build/
*.so

# Environments
.env
.venv/
venv/

# Data (large files)
data/raw/*.csv
data/raw/*.zip
*.parquet

# Models (large files)
models/saved/*.pt
models/saved/*.pkl
*.index

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs
logs/
*.log

# MLflow
mlruns/
""",

    "configs/config.py": """# DS19 — Central Configuration File

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
""",

    "configs/__init__.py": "",

    # Week 1 notebooks placeholder
    "notebooks/week1/01_eda.ipynb": json.dumps({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Week 1 — EDA Notebook\nRun the scripts instead. This is for exploration only."]
            }
        ]
    }, indent=2),

    # Init files for Python packages
    "models/__init__.py": "",
    "models/matrix_factorization/__init__.py": "",
    "models/transformer/__init__.py": "",
    "models/two_tower/__init__.py": "",
    "models/ranking/__init__.py": "",
    "retrieval/__init__.py": "",
    "bandits/__init__.py": "",
    "backend/__init__.py": "",
    "backend/app/__init__.py": "",
    "backend/app/api/__init__.py": "",
    "backend/app/services/__init__.py": "",
    "backend/app/schemas/__init__.py": "",
    "experiments/__init__.py": "",
    "tests/__init__.py": "",
    "tests/unit/__init__.py": "",
    "tests/integration/__init__.py": "",
}

# ─────────────────────────────────────────────────────────────
# CREATE FOLDERS
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("DS19 — Creating Project Structure")
print("=" * 60)

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  ✅ Created folder : {folder}/")

print()

# ─────────────────────────────────────────────────────────────
# CREATE FILES
# ─────────────────────────────────────────────────────────────

for filepath, content in files.items():
    full_path = Path(filepath)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if not full_path.exists():
        with open(full_path, "w", encoding="utf-8") as f:  # ✅ FIX HERE
            f.write(content)
        print(f"  📄 Created file   : {filepath}")
    else:
        print(f"  ⏭️  Already exists : {filepath}")

print()
print("=" * 60)
print("✅  Project structure created successfully!")
print("=" * 60)
print()
print("Next steps:")
print("  1. pip install -r requirements.txt")
print("  2. Download MovieLens 25M dataset")
print("  3. Run: python data/download_dataset.py")
print("=" * 60)