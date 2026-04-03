import sys
import json
import time
import numpy as np
import pickle
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.two_tower.model    import TwoTowerModel
from retrieval.faiss_index import load_faiss_index
# ─────────────────────────────────────────────────────────────
# CONFIG - MUST MATCH TRAINING CONFIG EXACTLY
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
SEQUENCES_DIR = Path("data/sequences")
MODELS_DIR    = Path("models/saved")
RETRIEVAL_DIR = Path("retrieval")

EMBED_DIM     = 64
NUM_HEADS     = 2
NUM_BLOCKS    = 2
MAX_SEQ_LEN   = 30    # Must match training
PAD_TOKEN     = 0
DROPOUT       = 0.0

# Set to True to force CPU if CUDA produces NaN
FORCE_CPU     = False

# ─────────────────────────────────────────────────────────────
# RETRIEVAL ENGINE
# ─────────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Stateful retrieval engine with NaN/Inf protection.
    """

    def __init__(self):
        self.model     = None
        self.index     = None
        self.item_ids  = None
        self.sequences = None
        self.idx2item  = None
        self.device    = None
        self._loaded   = False
        self.n_items   = 0
        self.max_seq_len = MAX_SEQ_LEN

    def load(self):
        """Load all components with corruption checks."""
        print("  [RetrievalEngine] Loading components...")
        t0 = time.time()

        # ── Device ────────────────────────────────────────────
        if FORCE_CPU:
            self.device = torch.device("cpu")
            print("  [RetrievalEngine] Forcing CPU mode")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [RetrievalEngine] Device: {self.device}")

        # ── Metadata ──────────────────────────────────────────
        meta         = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
        self.n_items = meta['n_items']
        print(f"  [RetrievalEngine] n_items from meta: {self.n_items}")

        # ── Two-Tower Model ───────────────────────────────────
        ckpt_path = MODELS_DIR / "two_tower_best.pt"
        
        # Load checkpoint safely
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
        config    = ckpt.get('config', {})

        embed_dim = config.get('hidden_dim', EMBED_DIM)
        num_heads = config.get('num_heads', NUM_HEADS)
        num_blocks = config.get('num_blocks', NUM_BLOCKS)
        max_seq_len = config.get('max_seq_len', MAX_SEQ_LEN)
        
        print(f"  [RetrievalEngine] Model config: embed={embed_dim}, heads={num_heads}, "
              f"blocks={num_blocks}, max_seq_len={max_seq_len}")

        self.model = TwoTowerModel(
            n_items    = self.n_items,
            hidden_dim = embed_dim,
            num_heads  = num_heads,
            num_blocks = num_blocks,
            max_seq_len= max_seq_len,
            dropout    = DROPOUT,
        )
        
        # Load state dict with NaN checking
        model_state = ckpt['model_state']
        
        # Check for NaN/Inf in checkpoint
        nan_params = []
        for name, param in model_state.items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_params.append(name)
        
        if nan_params:
            print(f"  ❌  [RetrievalEngine] NaN/Inf in checkpoint weights: {nan_params}")
            raise ValueError("Model checkpoint corrupted - contains NaN weights")
        
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"  ⚠️  [RetrievalEngine] Missing keys: {missing}")
        if unexpected:
            print(f"  ⚠️  [RetrievalEngine] Unexpected keys: {unexpected}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store config
        self.max_seq_len = max_seq_len

        # ── FAISS Index ───────────────────────────────────────
        self.index, self.item_ids = load_faiss_index()
        
        # Verify index is not corrupted
        if self.index is None or self.item_ids is None:
            raise ValueError("FAISS index failed to load")

        # ── User Sequences ────────────────────────────────────
        self.sequences = pickle.load(open(SEQUENCES_DIR / "sequences.pkl", "rb"))

        # ── ID Mappings ───────────────────────────────────────
        self.idx2item = json.load(open(PROCESSED_DIR / "idx2item.json"))
        self.idx2item = {int(k): v for k, v in self.idx2item.items()}
        self.item2idx = json.load(open(PROCESSED_DIR / "item2idx.json"))
        self.item2idx = {int(k): v for k, v in self.item2idx.items()}

        elapsed = time.time() - t0
        self._loaded = True
        print(f"  [RetrievalEngine] Ready in {elapsed:.2f}s")
        print(f"  [RetrievalEngine] {self.index.ntotal:,} items indexed")
        print(f"  [RetrievalEngine] {len(self.sequences):,} user sequences loaded")

    def _build_padded_seq(self, user_idx: int) -> torch.Tensor:
        """
        Builds a padded sequence tensor with bounds checking.
        """
        if user_idx not in self.sequences:
            seq = [PAD_TOKEN] * self.max_seq_len
        else:
            seq = self.sequences[user_idx]
            
            # Clamp to valid range [0, n_items-1], ensure integers
            seq = [min(max(int(item_idx), 0), self.n_items - 1) for item_idx in seq]
            
            # Truncate or pad
            if len(seq) < self.max_seq_len:
                seq = [PAD_TOKEN] * (self.max_seq_len - len(seq)) + seq
            else:
                seq = seq[-self.max_seq_len:]

        tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
        
        # Final safety check
        if (tensor >= self.n_items).any() or (tensor < 0).any():
            invalid = tensor[(tensor >= self.n_items) | (tensor < 0)]
            print(f"  ⚠️  WARNING: Invalid indices detected: {invalid.tolist()}")
            # Force clamp
            tensor = torch.clamp(tensor, min=0, max=self.n_items - 1)
            
        return tensor

    @torch.no_grad()
    def retrieve(self,
                 user_idx: int,
                 top_k: int = 100,
                 exclude_history: bool = True) -> dict:
        """
        End-to-end retrieval with NaN protection.
        """
        if not self._loaded:
            self.load()

        t0 = time.time()

        # ── Encode user ───────────────────────────────────────
        seq_tensor = self._build_padded_seq(user_idx)
        
        try:
            user_emb = self.model.encode_user(seq_tensor)
            
            # Ensure float32
            user_emb = user_emb.float()
            
            # CRITICAL: Check for NaN/Inf in user embedding
            if torch.isnan(user_emb).any() or torch.isinf(user_emb).any():
                print(f"  ⚠️  NaN/Inf detected in user embedding for user {user_idx}")
                # Fallback: use zero vector (will return random-ish results but won't crash)
                user_emb = torch.randn(1, user_emb.shape[1], device=self.device) * 0.01
                user_emb = user_emb / torch.norm(user_emb, dim=1, keepdim=True)
            
            # Normalize safely
            norms = torch.norm(user_emb, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            user_emb = user_emb / norms
            
        except Exception as e:
            print(f"  ❌  Error encoding user {user_idx}: {e}")
            # Fallback embedding
            user_emb = torch.randn(1, self.model.hidden_dim, device=self.device)
            user_emb = user_emb / torch.norm(user_emb, dim=1, keepdim=True)

        user_emb_np = user_emb.cpu().numpy().astype(np.float32)

        # ── FAISS search ──────────────────────────────────────
        search_k = min(top_k * 3, self.index.ntotal) if exclude_history else top_k
        
        try:
            scores_np, faiss_indices = self.index.search(user_emb_np, k=search_k)
            scores_np = scores_np[0]
            faiss_indices = faiss_indices[0]
        except Exception as e:
            print(f"  ❌  FAISS search error for user {user_idx}: {e}")
            return {
                'user_idx': user_idx,
                'candidate_idxs': [],
                'scores': [],
                'latency_ms': 0.0,
                'error': str(e)
            }

        # ── Map FAISS indices → item IDs ──────────────────────
        valid_mask = faiss_indices >= 0
        if not valid_mask.any():
            print(f"  ⚠️  No valid results from FAISS for user {user_idx}")
            return {
                'user_idx': user_idx,
                'candidate_idxs': [],
                'scores': [],
                'latency_ms': (time.time() - t0) * 1000,
            }
            
        candidate_ids = self.item_ids[faiss_indices[valid_mask]]
        candidate_scores = scores_np[valid_mask]
        
        # Check for NaN in scores (shouldn't happen but safety first)
        valid_scores = ~(np.isnan(candidate_scores) | np.isinf(candidate_scores))
        if not valid_scores.all():
            candidate_ids = candidate_ids[valid_scores]
            candidate_scores = candidate_scores[valid_scores]
            print(f"  ⚠️  Filtered { (~valid_scores).sum() } NaN/Inf scores")

        # ── Filter: remove training history ───────────────────
        if exclude_history and user_idx in self.sequences:
            history_set = set(self.sequences[user_idx])
            keep = [
                (int(c), float(s))
                for c, s in zip(candidate_ids, candidate_scores)
                if int(c) not in history_set
            ]
        else:
            keep = [(int(c), float(s)) for c, s in zip(candidate_ids, candidate_scores)]

        # ── Trim to top_k ─────────────────────────────────────
        keep = keep[:top_k]
        latency_ms = (time.time() - t0) * 1000

        return {
            'user_idx':       user_idx,
            'candidate_idxs': [x[0] for x in keep],
            'scores':         [x[1] for x in keep],
            'latency_ms':     round(latency_ms, 2),
        }

# ─────────────────────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DS19 Retrieval Pipeline (Fixed)")
    parser.add_argument("--user_id", type=int, default=1,
                        help="Internal user index (1-indexed)")
    parser.add_argument("--top_k",   type=int, default=100,
                        help="Number of candidates to retrieve")
    parser.add_argument("--latency_test", action="store_true",
                        help="Run 100 retrievals and report avg latency")
    args = parser.parse_args()

    print("=" * 64)
    print("  DS19 — Week 4 | Retrieval Pipeline (Fixed)")
    print("=" * 64)

    engine = RetrievalEngine()
    engine.load()

    # ── Single retrieval ──────────────────────────────────────
    print(f"\n  Retrieving top-{args.top_k} for user_idx={args.user_id} ...")
    result = engine.retrieve(args.user_id, top_k=args.top_k)

    print(f"\n  user_idx         : {result['user_idx']}")
    print(f"  candidates found : {len(result['candidate_idxs'])}")
    print(f"  latency          : {result['latency_ms']:.2f} ms")
    
    if result['candidate_idxs']:
        print(f"\n  Top-10 item indices and scores:")
        for i, (item_idx, score) in enumerate(
                zip(result['candidate_idxs'][:10], result['scores'][:10]), 1):
            movie_id = engine.idx2item.get(item_idx, "?")
            print(f"    {i:2d}. item_idx={item_idx:5d}  score={score:.4f}  movie_id={movie_id}")
    else:
        print("  ⚠️  No candidates returned!")

    # ── Latency test ──────────────────────────────────────────
    if args.latency_test:
        print(f"\n  Running latency test (100 users)...")
        import random
        user_ids = random.sample(list(engine.sequences.keys()), min(100, len(engine.sequences)))
        latencies = []
        for uid in user_ids:
            r = engine.retrieve(uid, top_k=100)
            latencies.append(r['latency_ms'])

        latencies = sorted(latencies)
        print(f"  Avg latency : {np.mean(latencies):.2f} ms")
        print(f"  P50 latency : {np.percentile(latencies, 50):.2f} ms")
        print(f"  P95 latency : {np.percentile(latencies, 95):.2f} ms")
        print(f"  P99 latency : {np.percentile(latencies, 99):.2f} ms")

if __name__ == "__main__":
    main()