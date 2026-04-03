import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from models.two_tower.dataset  import TwoTowerEvalDataset
from models.two_tower.model    import TwoTowerModel
from retrieval.faiss_index     import load_faiss_index

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR    = Path("data/splits")
MODELS_DIR    = Path("models/saved")
LOGS_DIR      = Path("logs")

EMBED_DIM     = 64
NUM_HEADS     = 2
NUM_BLOCKS    = 2
MAX_SEQ_LEN   = 50
DROPOUT       = 0.0

EVAL_K_LIST   = [10, 50, 100, 200]
EVAL_BATCH    = 256

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)

def load_model(device: torch.device, n_items: int):
    """Load model with proper config extraction."""
    global MAX_SEQ_LEN
    
    ckpt_path = MODELS_DIR / "two_tower_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get('config', {})
    
    saved_max_len = config.get('max_seq_len', MAX_SEQ_LEN)
    hidden_dim = config.get('hidden_dim', EMBED_DIM)
    
    if saved_max_len != MAX_SEQ_LEN:
        print(f"  ⚠️  Adjusting MAX_SEQ_LEN: {MAX_SEQ_LEN} → {saved_max_len}")
        MAX_SEQ_LEN = saved_max_len
    
    model = TwoTowerModel(
        n_items=n_items,
        hidden_dim=hidden_dim,
        num_heads=config.get('num_heads', NUM_HEADS),
        num_blocks=config.get('num_blocks', NUM_BLOCKS),
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )
    
    model.load_state_dict(ckpt['model_state'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"  ✅  Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    
    # CRITICAL: Test if model already normalizes
    test_input = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long, device=device)
    with torch.no_grad():
        test_output = model.encode_user(test_input)
        output_norm = torch.norm(test_output, dim=1).item()
    
    needs_normalize = abs(output_norm - 1.0) > 0.01
    print(f"  🔍  Model output norm: {output_norm:.4f} "
          f"({'needs normalize' if needs_normalize else 'already normalized'})")
    
    return model, MAX_SEQ_LEN, needs_normalize

@torch.no_grad()
def encode_users_batch(model, eval_ds, device, batch_size=EVAL_BATCH, normalize=True):
    """
    Encode all users with optional L2 normalization.
    CRITICAL FIX: Added F.normalize() to match training assumptions.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embs = []
    all_labels = []

    for i, (seqs, labels) in enumerate(loader):
        seqs = seqs.to(device)
        embs = model.encode_user(seqs)  # [B, d]
        
        # CRITICAL FIX: Apply L2 normalization if needed
        if normalize:
            embs = F.normalize(embs, p=2, dim=1)
        
        all_embs.append(embs.cpu().numpy())
        all_labels.append(labels.numpy())
        
        if i == 0:
            norms = np.linalg.norm(embs.cpu().numpy(), axis=1)
            print(f"    Batch 0 norms: min={norms.min():.4f}, max={norms.max():.4f}, "
                  f"mean={norms.mean():.4f}")

    user_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    label_items = np.concatenate(all_labels, axis=0).astype(np.int64)

    return user_embs, label_items

def compute_recall_at_k(faiss_index, item_ids, user_embs, label_items, k_list, debug=True):
    """Compute recall with detailed debugging."""
    M = len(user_embs)
    max_k = max(k_list)
    search_k = min(max_k * 5, faiss_index.ntotal)

    print(f"  Searching FAISS: {M:,} users × top-{search_k}...")
    
    # Ensure contiguous array for FAISS
    user_embs = np.ascontiguousarray(user_embs.astype(np.float32))
    
    t0 = time.time()
    scores_all, indices_all = faiss_index.search(user_embs, k=search_k)
    print(f"  FAISS search: {time.time() - t0:.2f}s")

    # DEBUG: Show first 3 users
    if debug:
        print("\n  Debug - First 3 users:")
        for i in range(min(3, M)):
            true_item = label_items[i]
            top5 = item_ids[indices_all[i][:5]]
            top5_scores = scores_all[i][:5]
            rank = np.where(top5 == true_item)[0]
            status = f"✅ rank {rank[0]+1}" if len(rank) > 0 else "❌ not in top5"
            print(f"    User {i}: true={true_item}, top5={top5}, scores={top5_scores}, {status}")

    # Compute recalls
    recall_at_k = {k: 0 for k in k_list}
    
    for i in range(M):
        fi = indices_all[i]
        valid = fi >= 0
        cand_idx = fi[valid]
        if len(cand_idx) == 0:
            continue
        cand_ids = item_ids[cand_idx]
        label = label_items[i]

        for k in k_list:
            if label in cand_ids[:k]:
                recall_at_k[k] += 1

    return {k: v / M for k, v in recall_at_k.items()}

def diagnose_low_recall(user_embs, label_items, faiss_index, item_ids):
    """Run diagnostics when recall is poor."""
    print_section("🔍 DIAGNOSTICS")
    
    # Check 1: Embedding statistics
    print("\n1. Embedding Statistics:")
    print(f"   Shape: {user_embs.shape}")
    norms = np.linalg.norm(user_embs, axis=1)
    print(f"   L2 norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    
    # Check 2: Check for model collapse
    sample = user_embs[:min(100, len(user_embs))]
    similarities = np.dot(sample, sample.T)
    avg_sim = similarities[np.triu_indices_from(similarities, k=1)].mean()
    print(f"\n2. User Similarity (avg off-diagonal): {avg_sim:.4f}")
    if avg_sim > 0.9:
        print("   ❌ WARNING: All users are too similar! Model may have collapsed.")
    
    # Check 3: Test specific retrievals
    print("\n3. Sample Retrievals:")
    for i in range(min(5, len(user_embs))):
        query = user_embs[i:i+1]
        D, I = faiss_index.search(query, k=100)
        true_item = label_items[i]
        retrieved = item_ids[I[0]]
        rank_list = np.where(retrieved == true_item)[0]
        if len(rank_list) > 0:
            print(f"   User {i}: true item at rank {rank_list[0]+1} (score={D[0][rank_list[0]]:.4f})")
        else:
            print(f"   User {i}: ❌ true item not in top 100")

def main():
    global MAX_SEQ_LEN
    
    print_section("DS19 — Week 4 | Recall Evaluation (FIXED)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
    n_items = meta['n_items']
    print(f"  Device: {device}, Items: {n_items:,}")

    # Load model
    model, MAX_SEQ_LEN, needs_normalize = load_model(device, n_items)

    # Load FAISS
    print_section("Loading FAISS Index")
    faiss_index, item_ids = load_faiss_index()
    print(f"  FAISS: {faiss_index.ntotal:,} items, dim={faiss_index.d}")
    
    # Check FAISS vectors are normalized
    test_query = np.random.randn(1, faiss_index.d).astype(np.float32)
    test_query = test_query / np.linalg.norm(test_query)
    D, I = faiss_index.search(test_query, k=1)
    print(f"  FAISS max inner product: {D[0][0]:.4f} "
          f"({'normalized' if D[0][0] < 1.1 else 'NOT normalized'})")

    results = {}

    for split in ["val", "test"]:
        print_section(f"Evaluating: {split.upper()}")
        
        eval_ds = TwoTowerEvalDataset(split=split, max_seq_len=MAX_SEQ_LEN)
        print(f"  Users: {len(eval_ds):,}")

        # Encode with explicit L2 normalization
        print("  Encoding users...")
        user_embs, label_items = encode_users_batch(
            model, eval_ds, device, 
            normalize=needs_normalize  # Normalize if model doesn't do it
        )

        # Compute recall
        recall = compute_recall_at_k(
            faiss_index, item_ids, user_embs, label_items, 
            k_list=EVAL_K_LIST, debug=True
        )
        
        results[split] = recall

        # Print results
        print(f"\n  Results ({split.upper()}):")
        for k in EVAL_K_LIST:
            v = recall[k]
            status = "✅" if v >= 0.85 else "⚠️" if v >= 0.75 else "❌"
            print(f"    Recall@{k}: {v:.4f} {status}")

    # Summary
    print_section("Summary")
    test_r100 = results['test'].get(100, 0)
    print(f"  Test Recall@100: {test_r100:.4f}")
    
    if test_r100 < 0.75:
        # Run diagnostics on test set
        eval_ds = TwoTowerEvalDataset(split="test", max_seq_len=MAX_SEQ_LEN)
        user_embs, label_items = encode_users_batch(
            model, eval_ds, device, normalize=needs_normalize
        )
        diagnose_low_recall(user_embs, label_items, faiss_index, item_ids)
        
        print("\n  🔧 QUICK FIXES TO TRY:")
        print("     1. If norms not ~1.0: The normalize=True fix should handle it")
        print("     2. If FAISS not normalized: Rebuild index with normalized vectors")
        print("     3. If model collapsed: Retrain with temperature=0.05, more negatives")

if __name__ == "__main__":
    main()