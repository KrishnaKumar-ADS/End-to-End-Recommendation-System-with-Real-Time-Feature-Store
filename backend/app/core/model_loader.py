import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
import faiss
import lightgbm as lgb
from pathlib import Path
from loguru import logger

# ─── Import Two-Tower Model Architecture ──────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from models.two_tower.model import TwoTowerModel

from backend.app.core.config import (
    TWO_TOWER_MODEL_PATH, TWO_TOWER_CONFIG_PATH,
    FAISS_INDEX_PATH, FAISS_ITEM_MAP_PATH,
    LGBM_MODEL_PATH,
    INTERACTIONS_FILE, ITEM2IDX_FILE, IDX2ITEM_FILE,
    DATASET_META_FILE, SEQUENCES_FILE, MOVIES_FILE,
    USER_FEATURES_FILE, ITEM_FEATURES_FILE,
    DEVICE, LOGS_DIR
)


class ModelLoader:
    """
    Holds all loaded models and lookup tables.
    Created ONCE at startup, shared across all requests.
    """

    def __init__(self):
        self.two_tower_model     = None
        self.faiss_index         = None
        self.faiss_item_map      = None
        self.lgbm_model          = None
        self.user2idx            = None
        self.item2idx            = None
        self.idx2item            = None
        self.dataset_meta        = None
        self.n_items             = None
        self.interactions_dict   = None
        self.user_features_dict  = None
        self.item_features_dict  = None
        self.idx2title           = None
        self.item_genres_dict    = None
        self.device              = DEVICE

    def load_all(self):
        """Load everything. Called once at startup."""
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("  DS19 — Model Loader: Starting up...")
        logger.info("=" * 60)

        self._load_metadata()
        self._load_interactions()
        self._load_two_tower()
        self._load_faiss()
        self._load_lgbm()
        self._load_features()
        self._load_movie_metadata()

        elapsed = time.time() - t0
        logger.info(f"✅ All models loaded in {elapsed:.1f}s")
        logger.info(f"   Device:  {self.device}")
        logger.info(f"   n_users: {len(self.user2idx):,}")
        logger.info(f"   n_items: {self.n_items:,}")
        logger.info("=" * 60)

    def _load_metadata(self):
        logger.info("Loading metadata...")
        with open(DATASET_META_FILE) as f:
            self.dataset_meta = json.load(f)
        self.n_items = self.dataset_meta["n_items"]

        with open(ITEM2IDX_FILE) as f:
            self.item2idx = json.load(f)
        with open(IDX2ITEM_FILE) as f:
            self.idx2item = json.load(f)

        user2idx_path = Path(INTERACTIONS_FILE).parent / "user2idx.json"
        if user2idx_path.exists():
            with open(user2idx_path) as f:
                self.user2idx = json.load(f)
        else:
            df = pd.read_csv(INTERACTIONS_FILE, usecols=["user_idx"])
            unique_users = sorted(df["user_idx"].unique())
            self.user2idx = {str(u): u for u in unique_users}

        logger.info(f"  ✅ Metadata: {self.n_items:,} items, {len(self.user2idx):,} users")

    def _load_interactions(self):
        logger.info("Loading user interaction sequences...")

        if Path(SEQUENCES_FILE).exists():
            with open(SEQUENCES_FILE, "rb") as f:
                loaded = pickle.load(f)

            if not isinstance(loaded, dict):
                raise ValueError("sequences.pkl must contain a dict[user_idx] -> sequence")

            sequences = {}
            for user_idx, sequence in loaded.items():
                try:
                    user_idx_int = int(user_idx)
                except (TypeError, ValueError):
                    continue

                if sequence is None:
                    continue

                seq_list = [int(item_idx) for item_idx in list(sequence)]
                sequences[user_idx_int] = seq_list

            self.interactions_dict = sequences
            logger.info(
                f"  ✅ Loaded sequences.pkl for {len(self.interactions_dict):,} users"
            )
            return

        df = pd.read_csv(
            INTERACTIONS_FILE,
            usecols=["user_idx", "item_idx", "timestamp"]
        )
        df = df.sort_values(["user_idx", "timestamp"])

        self.interactions_dict = (
            df.groupby("user_idx")["item_idx"]
            .apply(list)
            .to_dict()
        )
        logger.info(f"  ✅ Loaded sequences for {len(self.interactions_dict):,} users")

    def _load_two_tower(self):
        logger.info("Loading Two-Tower model...")

        # Load config
        if TWO_TOWER_CONFIG_PATH.exists():
            with open(TWO_TOWER_CONFIG_PATH) as f:
                cfg = json.load(f)
            hidden_dim  = cfg.get("hidden_dim", cfg.get("embed_dim", 64))
            num_heads   = cfg.get("num_heads", 2)
            num_blocks  = cfg.get("num_blocks", 2)
        else:
            hidden_dim, num_heads, num_blocks = 64, 2, 2

        # Instantiate model - use hidden_dim (NOT embed_dim)
        self.two_tower_model = TwoTowerModel(
            n_items     = self.n_items,        # NO +1 (model handles PAD internally)
            hidden_dim  = hidden_dim,
            num_heads   = num_heads,
            num_blocks  = num_blocks,
            max_seq_len = 50,
        )

        # Load weights - extract model_state from checkpoint
        checkpoint = torch.load(
            TWO_TOWER_MODEL_PATH,
            map_location=self.device
        )
        
        # Extract model_state if present (from training checkpoint)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            logger.info("  ✅ Loaded checkpoint with model_state key")
        else:
            state_dict = checkpoint
            logger.info("  ✅ Loaded direct state_dict")
        
        self.two_tower_model.load_state_dict(state_dict)
        self.two_tower_model.to(self.device)
        self.two_tower_model.eval()

        logger.info(f"  ✅ Two-Tower loaded on {self.device}")

    def _load_faiss(self):
        logger.info("Loading FAISS index...")
        self.faiss_index    = faiss.read_index(str(FAISS_INDEX_PATH))
        self.faiss_item_map = np.load(str(FAISS_ITEM_MAP_PATH))
        logger.info(
            f"  ✅ FAISS index: {self.faiss_index.ntotal:,} vectors "
            f"({self.faiss_index.d}-dim)"
        )

    def _load_lgbm(self):
        logger.info("Loading LightGBM ranker...")
        with open(LGBM_MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)

        # Some training pipelines persist a metadata dict with the booster under "model".
        if isinstance(artifact, dict) and "model" in artifact:
            self.lgbm_model = artifact["model"]
            logger.info("  ✅ LightGBM ranker loaded from artifact['model']")
        else:
            self.lgbm_model = artifact
            logger.info("  ✅ LightGBM ranker loaded")

    def _load_features(self):
        logger.info("Loading feature tables into memory...")

        # User features
        user_df = pd.read_parquet(USER_FEATURES_FILE)
        
        # Reset index if 'encoded_user_id' is the index
        if user_df.index.name == 'encoded_user_id':
            user_df = user_df.reset_index()
            user_df = user_df.rename(columns={'encoded_user_id': 'user_idx'})
        
        self.user_features_dict = user_df.set_index("user_idx").to_dict(orient="index")
        logger.info(f"  ✅ User features: {len(self.user_features_dict):,} users")

        # Item features
        item_df = pd.read_parquet(ITEM_FEATURES_FILE)
        
        # Reset index if needed
        if item_df.index.name == 'encoded_item_id':
            item_df = item_df.reset_index()
            item_df = item_df.rename(columns={'encoded_item_id': 'item_idx'})
        
        self.item_features_dict = item_df.set_index("item_idx").to_dict(orient="index")
        logger.info(f"  ✅ Item features: {len(self.item_features_dict):,} items")

    def _load_movie_metadata(self):
        logger.info("Loading movie metadata...")
        movies_df = pd.read_csv(MOVIES_FILE)

        idx2title       = {}
        item_genres_dict = {}

        for _, row in movies_df.iterrows():
            movie_id_str = str(int(row["movieId"]))
            if movie_id_str in self.item2idx:
                item_idx = self.item2idx[movie_id_str]
                idx2title[item_idx]        = row["title"]
                genres = row["genres"].split("|") if pd.notna(row["genres"]) else []
                item_genres_dict[item_idx] = genres

        self.idx2title        = idx2title
        self.item_genres_dict = item_genres_dict
        logger.info(f"  ✅ Movie metadata: {len(self.idx2title):,} titles loaded")