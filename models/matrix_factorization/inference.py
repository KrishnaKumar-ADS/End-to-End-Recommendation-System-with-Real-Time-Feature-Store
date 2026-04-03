import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
sys.modules['numpy._core'] = np.core

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.matrix_factorization.model import MatrixFactorization
from models.matrix_factorization.dataset import build_user_item_sets

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
RAW_DIR       = Path("data/raw")

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────
# RECOMMENDER CLASS
# ─────────────────────────────────────────────────────────────

class MFRecommender:
    """
    Stateful recommender class that wraps the trained MF model.

    Designed for:
      - Interactive inference (this script)
      - FastAPI integration (Week 7)

    Loads everything ONCE at init, then answers queries instantly.

    Attributes:
        model      : trained MatrixFactorization model
        device     : torch.device
        user2idx   : {original_user_id: internal_user_idx}
        item2idx   : {original_movie_id: internal_item_idx}
        idx2item   : {internal_item_idx: original_movie_id}
        user_items : {user_idx: set of all interacted item_idxs}
        movies_df  : DataFrame with movieId, title, genres (for display)
    """

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        """
        Initializes the recommender by loading model, mappings, and metadata.

        Args:
            checkpoint_path : path to trained model checkpoint (.pt)
            device          : torch.device (auto-detected if None)
        """
        self.device = device or get_device()
        self._load_mappings()
        self._load_model(checkpoint_path)
        self._load_user_history()
        self._load_movie_metadata()

    def _load_mappings(self):
        """Loads user/item ID mappings (original IDs ↔ internal indices)."""
        print("  Loading ID mappings...")

        with open(PROCESSED_DIR / "user2idx.json") as f:
            self.user2idx = {int(k): int(v) for k, v in json.load(f).items()}

        with open(PROCESSED_DIR / "item2idx.json") as f:
            # item2idx maps original movie_id → internal item_idx (0-based)
            # BUT we shifted item_idx by +1 in pipeline (0=PAD)
            # So actual internal idx = item2idx[movie_id] + 1
            raw_item2idx = {int(k): int(v) for k, v in json.load(f).items()}
            self.item2idx = {k: v + 1 for k, v in raw_item2idx.items()}  # +1 for PAD shift

        with open(PROCESSED_DIR / "idx2item.json") as f:
            # idx2item: internal idx → original movie_id
            # Also needs PAD shift correction
            raw_idx2item = {int(k): int(v) for k, v in json.load(f).items()}
            self.idx2item = {k + 1: v for k, v in raw_idx2item.items()}  # +1 for PAD shift

        print(f"  ✅ Loaded {len(self.user2idx):,} users, {len(self.item2idx):,} items")

    def _load_model(self, checkpoint_path: str):
        """Loads the trained MF model from checkpoint."""
        print(f"  Loading model from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        hp = checkpoint['hyperparams']

        self.model = MatrixFactorization(
            n_users       = hp['n_users'],
            n_items       = hp['n_items'],
            embedding_dim = hp['embedding_dim'],
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        print(f"  ✅ Model loaded: {hp['n_users']:,} users × {hp['n_items']:,} items × d={hp['embedding_dim']}")

    def _load_user_history(self):
        """Loads each user's interaction history for masking in inference."""
        print("  Loading user interaction history...")

        df = pd.read_csv(PROCESSED_DIR / "interactions.csv")
        self.user_items = build_user_item_sets(df)

        print(f"  ✅ Loaded history for {len(self.user_items):,} users")

    def _load_movie_metadata(self):
        """Loads movie titles and genres for displaying recommendations."""
        movies_path = RAW_DIR / "movies.csv"
        if movies_path.exists():
            self.movies_df = pd.read_csv(movies_path)
            print(f"  ✅ Loaded metadata for {len(self.movies_df):,} movies")
        else:
            self.movies_df = None
            print(f"  ⚠️  movies.csv not found — titles won't be shown")

    def get_movie_title(self, movie_id: int) -> str:
        """Returns movie title for a given original movie_id."""
        if self.movies_df is None:
            return f"movie_{movie_id}"
        row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(row) == 0:
            return f"movie_{movie_id}"
        return row.iloc[0]['title']

    def get_movie_genres(self, movie_id: int) -> str:
        """Returns genres for a given movie_id."""
        if self.movies_df is None:
            return ""
        row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(row) == 0:
            return ""
        return row.iloc[0]['genres']

    @torch.no_grad()
    def recommend(
        self,
        user_id    : int,
        top_k      : int = 10,
        exclude_seen : bool = True,
    ) -> List[Dict]:
        """
        Main inference method.
        Returns top-K item recommendations for a given user_id.

        Args:
            user_id      : ORIGINAL user ID (from ratings.csv)
            top_k        : number of recommendations to return
            exclude_seen : whether to exclude items the user already interacted with

        Returns:
            List of dicts:
            [
              {'rank': 1, 'item_idx': 4231, 'movie_id': 1891, 'title': 'Star Wars...', 'score': 2.341},
              {'rank': 2, ...},
              ...
            ]

        Raises:
            KeyError: if user_id is not in the training data
        """
        # ── Validate user_id ─────────────────────────────────
        if user_id not in self.user2idx:
            raise KeyError(
                f"user_id={user_id} not found in training data. "
                f"Valid range: {min(self.user2idx.keys())} - {max(self.user2idx.keys())}"
            )

        # ── Get internal index ────────────────────────────────
        user_idx = self.user2idx[user_id]

        # ── Get user embedding ────────────────────────────────
        user_tensor = torch.tensor(user_idx, dtype=torch.long).to(self.device)
        user_emb    = self.model.user_emb(user_tensor)   # [d]

        # ── Score ALL items ───────────────────────────────────
        all_item_embs = self.model.item_emb.weight.data  # [n_items, d]
        scores = (all_item_embs @ user_emb).cpu().numpy() # [n_items]

        # ── Mask PAD ──────────────────────────────────────────
        scores[0] = -np.inf

        # ── Mask seen items ───────────────────────────────────
        if exclude_seen:
            seen = self.user_items.get(user_idx, set())
            for item_idx in seen:
                if 0 <= item_idx < len(scores):
                    scores[item_idx] = -np.inf

        # ── Rank items ────────────────────────────────────────
        ranked_idxs = np.argsort(-scores)   # [n_items] descending

        # ── Build recommendations ─────────────────────────────
        recommendations = []
        for rank, item_idx in enumerate(ranked_idxs[:top_k * 2], start=1):
            if scores[item_idx] == -np.inf:
                continue  # Skip masked items

            # Convert internal item_idx → original movie_id
            movie_id = self.idx2item.get(item_idx, -1)
            if movie_id == -1:
                continue

            title  = self.get_movie_title(movie_id)
            genres = self.get_movie_genres(movie_id)

            recommendations.append({
                'rank'     : len(recommendations) + 1,
                'item_idx' : int(item_idx),
                'movie_id' : int(movie_id),
                'title'    : title,
                'genres'   : genres,
                'score'    : float(scores[item_idx]),
            })

            if len(recommendations) >= top_k:
                break

        return recommendations

    def get_user_history(self, user_id: int, n: int = 10) -> List[Dict]:
        """
        Returns the user's recent interaction history.
        Useful for showing context alongside recommendations.

        Args:
            user_id: original user ID
            n      : number of recent items to return

        Returns:
            List of dicts with movie info
        """
        if user_id not in self.user2idx:
            return []

        user_idx  = self.user2idx[user_id]
        seen_idxs = list(self.user_items.get(user_idx, set()))[:n]

        history = []
        for item_idx in seen_idxs:
            movie_id = self.idx2item.get(item_idx, -1)
            if movie_id == -1:
                continue
            history.append({
                'item_idx' : int(item_idx),
                'movie_id' : int(movie_id),
                'title'    : self.get_movie_title(movie_id),
                'genres'   : self.get_movie_genres(movie_id),
            })

        return history[:n]


# ─────────────────────────────────────────────────────────────
# MAIN — CLI INTERFACE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DS19 — Matrix Factorization Inference"
    )
    parser.add_argument(
        '--user_id', type=int, default=1,
        help='Original user_id from ratings.csv (e.g. 1, 42, 500)'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Number of recommendations to return (default: 10)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=str(MODELS_DIR / "mf_best.pt"),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--show_history', action='store_true',
        help='Also show user interaction history'
    )
    args = parser.parse_args()

    print_section("DS19 — Week 2 | MF Inference")

    # ── Initialize recommender ────────────────────────────────
    print("\n  Loading recommender system...")
    recommender = MFRecommender(
        checkpoint_path = args.checkpoint,
        device          = get_device(),
    )
    print(f"\n  ✅ Recommender ready!")

    # ── Show user history (optional) ──────────────────────────
    if args.show_history:
        print_section(f"User {args.user_id} — Interaction History")
        history = recommender.get_user_history(args.user_id, n=10)
        if history:
            for item in history:
                print(f"  [{item['item_idx']:>6}]  {item['title']:<50}  {item['genres']}")
        else:
            print(f"  ⚠️  No history found for user {args.user_id}")

    # ── Get recommendations ───────────────────────────────────
    print_section(f"Top-{args.top_k} Recommendations for User {args.user_id}")

    try:
        recs = recommender.recommend(
            user_id      = args.user_id,
            top_k        = args.top_k,
            exclude_seen = True,
        )
    except KeyError as e:
        print(f"\n  ❌ {e}")
        return

    # ── Display ───────────────────────────────────────────────
    print(f"\n  {'Rank':<6} {'Score':>8}  {'Movie Title':<50}  Genres")
    print(f"  {'----':<6} {'-----':>8}  {'------------------':<50}  ------")

    for rec in recs:
        print(
            f"  #{rec['rank']:<5} {rec['score']:>8.4f}  "
            f"{rec['title'][:50]:<50}  "
            f"{rec['genres'][:30]}"
        )

    print()
    print(f"  User {args.user_id} | Top-{args.top_k} Recommendations Generated ✅")
    print()


if __name__ == "__main__":
    main()