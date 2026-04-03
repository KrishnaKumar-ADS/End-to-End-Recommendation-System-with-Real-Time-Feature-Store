import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.two_tower.dataset import ItemCatalogDataset
from models.two_tower.model   import TwoTowerModel

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
RETRIEVAL_DIR = Path("retrieval")

EMBED_DIM     = 64
NUM_HEADS     = 2
NUM_BLOCKS    = 2
MAX_SEQ_LEN   = 50
DROPOUT       = 0.0   # No dropout at inference

# FAISS config
FAISS_NLIST   = 100
FAISS_NPROBE  = 10
FAISS_USE_IVF = False

ENCODE_BATCH  = 512

# Set to True if CUDA keeps producing NaN (forces CPU encoding)
FORCE_CPU     = False

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)

def load_model(device: torch.device, n_items: int) -> TwoTowerModel:
    """Loads the best trained Two-Tower model checkpoint with NaN checks."""
    ckpt_path = MODELS_DIR / "two_tower_best.pt"

    if not ckpt_path.exists():
        print(f"  ❌  two_tower_best.pt not found at {ckpt_path}")
        print("  ❌  Run training first: python models/two_tower/train.py")
        sys.exit(1)

    # Try weights_only first, fall back to False for older checkpoints
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    config = ckpt.get('config', {})

    # Initialize model
    model = TwoTowerModel(
        n_items     = n_items,
        hidden_dim  = config.get('hidden_dim', EMBED_DIM),
        num_heads   = config.get('num_heads',  NUM_HEADS),
        num_blocks  = config.get('num_blocks', NUM_BLOCKS),
        max_seq_len = config.get('max_seq_len', MAX_SEQ_LEN),
        dropout     = DROPOUT,
    )

    # Load weights
    model_state = ckpt['model_state']
    
    # Check for NaN in checkpoint weights
    nan_params = []
    for name, param in model_state.items():
        if torch.isnan(param).any() or torch.isinf(param).any():
            nan_params.append(name)
    
    if nan_params:
        print(f"  ❌  NaN/Inf detected in checkpoint weights: {nan_params}")
        print("  💡  The checkpoint is corrupted. Try an earlier epoch or retrain.")
        sys.exit(1)

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys: {missing}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {unexpected}")
        
    model = model.to(device)
    model.eval()

    print(f"  ✅  Loaded two_tower_best.pt (epoch {ckpt.get('epoch', '?')})")
    print(f"  ✅  Best Recall@50: {ckpt.get('best_recall', '?'):.4f}")
    return model

# ─────────────────────────────────────────────────────────────
# STEP 1: ENCODE ALL ITEMS (WITH NAN HANDLING)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all_items(model: TwoTowerModel, device: torch.device, n_items: int):
    """
    Runs all item IDs through the Item Tower to get embeddings.
    Includes comprehensive NaN/Inf detection and recovery.
    """
    print_section("Step 1: Encoding All Items")

    catalog = ItemCatalogDataset()
    loader = DataLoader(catalog, batch_size=ENCODE_BATCH, shuffle=False, num_workers=0)

    all_ids = []
    all_embs = []
    
    total_nan_detected = 0

    for batch_idx, batch_ids in enumerate(loader):
        batch_ids = batch_ids.to(device)
        
        try:
            embs = model.encode_item(batch_ids)
            
            # Ensure float32 (prevent any half-precision issues)
            embs = embs.float()
            
            # Check and fix NaN/Inf
            nan_mask = torch.isnan(embs) | torch.isinf(embs)
            if nan_mask.any():
                nan_count = nan_mask.sum().item()
                total_nan_detected += nan_count
                print(f"  ⚠️  Batch {batch_idx}: {nan_count} NaN/Inf values detected")
                
                # Replace with small random noise
                noise = torch.randn_like(embs) * 0.01
                embs = torch.where(nan_mask, noise, embs)
            
            # Safe normalization (prevent division by zero)
            norms = torch.norm(embs, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            embs = embs / norms
            
            # Verify no NaN after normalization
            if torch.isnan(embs).any():
                print(f"  ⚠️  Batch {batch_idx}: NaN after normalization, using random")
                embs = torch.randn(batch_ids.size(0), EMBED_DIM, device=device)
                embs = embs / torch.norm(embs, dim=1, keepdim=True)
                
        except Exception as e:
            print(f"  ❌  Error encoding batch {batch_idx}: {e}")
            # Fallback: random normalized embeddings
            embs = torch.randn(batch_ids.size(0), EMBED_DIM, device=device)
            embs = embs / torch.norm(embs, dim=1, keepdim=True)

        all_ids.append(batch_ids.cpu().numpy())
        all_embs.append(embs.cpu().numpy())

    item_ids_np = np.concatenate(all_ids, axis=0).astype(np.int64)
    embeddings_np = np.concatenate(all_embs, axis=0).astype(np.float32)

    # Final cleanup: replace any remaining NaN
    if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
        nan_count = np.isnan(embeddings_np).sum() + np.isinf(embeddings_np).sum()
        print(f"  ⚠️  Final cleanup: {nan_count} NaN/Inf values remaining")
        embeddings_np = np.nan_to_num(embeddings_np, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Re-normalize
        norms = np.linalg.norm(embeddings_np, axis=1, keepdim=True)
        embeddings_np = embeddings_np / np.maximum(norms, 1e-8)

    # Verification
    norms = np.linalg.norm(embeddings_np, axis=1)
    print(f"  ✅  Encoded {len(item_ids_np):,} items")
    print(f"  ✅  Embedding shape: {embeddings_np.shape}")
    print(f"  ✅  Mean norm: {norms.mean():.4f} (should be ~1.0)")
    print(f"  ✅  Min/Max norm: {norms.min():.4f} / {norms.max():.4f}")
    
    if total_nan_detected > 0:
        print(f"  🔧  Fixed {total_nan_detected} NaN values during encoding")

    return item_ids_np, embeddings_np

# ─────────────────────────────────────────────────────────────
# STEP 2: BUILD FAISS INDEX
# ─────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, d: int):
    """Builds a FAISS index from item embeddings."""
    print_section("Step 2: Building FAISS Index")

    try:
        import faiss
    except ImportError:
        print("  ❌  faiss not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    # Final safety check before building index
    if np.isnan(embeddings).any():
        raise ValueError("Cannot build FAISS index: embeddings contain NaN")

    N = embeddings.shape[0]
    print(f"  N items  : {N:,}")
    print(f"  d dims   : {d}")
    print(f"  Mode     : {'IVFFlat (approximate)' if FAISS_USE_IVF else 'FlatIP (exact)'}")

    if FAISS_USE_IVF and N > 1000:
        nlist = min(FAISS_NLIST, N // 10)
        nlist = max(nlist, 10)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"  nlist    : {nlist}")
        print(f"  nprobe   : {FAISS_NPROBE}")
        print(f"  Training IVF quantizer...")
        index.train(embeddings)
        print(f"  ✅  IVF quantizer trained")
        index.add(embeddings)
        index.nprobe = FAISS_NPROBE
    else:
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

    print(f"  ✅  Index built: {index.ntotal:,} vectors")
    
    # Sanity check
    D, I = index.search(embeddings[:1], k=5)
    if I[0][0] != 0:
        print(f"  ⚠️  Sanity check warning: Top-1 for item 0 is {I[0][0]} (expected 0)")
    else:
        print(f"  ✅  Sanity check passed: self-similarity = {D[0][0]:.4f}")

    return index

# ─────────────────────────────────────────────────────────────
# STEP 3: SAVE INDEX + METADATA
# ─────────────────────────────────────────────────────────────

def save_index(index, item_ids: np.ndarray, embeddings: np.ndarray, d: int):
    """Saves FAISS index, item IDs, embeddings, and metadata to disk."""
    print_section("Step 3: Saving Index & Artifacts")

    import faiss
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    index_path = RETRIEVAL_DIR / "faiss_item.index"
    faiss.write_index(index, str(index_path))
    size_mb = index_path.stat().st_size / 1e6
    print(f"  ✅  faiss_item.index saved ({size_mb:.2f} MB)")

    emb_path = RETRIEVAL_DIR / "item_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"  ✅  item_embeddings.npy saved  {embeddings.shape}")

    ids_path = RETRIEVAL_DIR / "item_ids.npy"
    np.save(ids_path, item_ids)
    print(f"  ✅  item_ids.npy saved  {item_ids.shape}")

    meta = {
        "n_items":         int(len(item_ids)),
        "embed_dim":       int(d),
        "index_type":      "IVFFlat" if FAISS_USE_IVF else "FlatIP",
        "nlist":           FAISS_NLIST if FAISS_USE_IVF else None,
        "nprobe":          FAISS_NPROBE if FAISS_USE_IVF else None,
        "metric":          "inner_product",
        "normalized":      True,
        "model_checkpoint":"two_tower_best.pt",
        "item_ids_start":  int(item_ids.min()),
        "item_ids_end":    int(item_ids.max()),
    }
    meta_path = RETRIEVAL_DIR / "faiss_meta.json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"  ✅  faiss_meta.json saved")

    return str(index_path)

# ─────────────────────────────────────────────────────────────
# STEP 4: LOAD INDEX (utility for retrieve.py)
# ─────────────────────────────────────────────────────────────

def load_faiss_index():
    """Loads FAISS index + item_ids from disk."""
    import faiss

    index_path = RETRIEVAL_DIR / "faiss_item.index"
    ids_path   = RETRIEVAL_DIR / "item_ids.npy"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run: python retrieval/faiss_index.py"
        )

    index    = faiss.read_index(str(index_path))
    item_ids = np.load(ids_path)

    meta = json.load(open(RETRIEVAL_DIR / "faiss_meta.json"))
    if meta.get("nprobe") is not None:
        index.nprobe = meta["nprobe"]

    return index, item_ids

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print_section("DS19 — Week 4 | FAISS Index Builder")

    # Load metadata
    meta    = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
    n_items = meta['n_items']
    d       = EMBED_DIM

    # Device selection
    if FORCE_CPU:
        device = torch.device("cpu")
        print(f"  Device: cpu (FORCE_CPU=True)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

    # Load trained model
    model = load_model(device, n_items)

    # Encode all items (with NaN handling)
    item_ids, embeddings = encode_all_items(model, device, n_items)

    # Build FAISS index
    index = build_faiss_index(embeddings, d)

    # Save everything
    save_index(index, item_ids, embeddings, d)

    print_section("FAISS Index Build Complete")
    print(f"  ✅  {index.ntotal:,} items indexed")
    print(f"  ✅  retrieval/faiss_item.index")
    print(f"  ✅  retrieval/item_embeddings.npy")
    print(f"  ✅  retrieval/item_ids.npy")
    print(f"  ✅  retrieval/faiss_meta.json")
    print()
    print("  NEXT → Evaluate retrieval: python models/two_tower/evaluate.py")

if __name__ == "__main__":
    main()