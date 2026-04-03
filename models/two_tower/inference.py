import sys
import json
import time
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from retrieval.retrieve import RetrievalEngine

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
SEQUENCES_DIR = Path("data/sequences")
SPLITS_DIR    = Path("data/splits")
MODELS_DIR    = Path("models/saved")

# ─────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────

def demo_single_user(engine: RetrievalEngine, user_idx: int, top_k: int, show_history: bool):
    """Demo retrieval for a single user."""
    print(f"\n  User idx : {user_idx}")

    # Show user history if requested
    if show_history and user_idx in engine.sequences:
        history = engine.sequences[user_idx]
        print(f"  History  : {len(history)} interactions")
        print(f"  Last 5 items (most recent): {history[-5:]}")

    # Retrieve
    result = engine.retrieve(user_idx, top_k=top_k)

    print(f"\n  Top-{min(top_k, 20)} Retrieved Candidates:")
    print(f"  {'Rank':<5} {'item_idx':<10} {'movie_id':<12} {'Score':<8}")
    print(f"  {'-'*38}")

    for rank, (item_idx, score) in enumerate(
            zip(result['candidate_idxs'][:20], result['scores'][:20]), 1):
        movie_id = engine.idx2item.get(item_idx, "unknown")
        print(f"  {rank:<5} {item_idx:<10} {str(movie_id):<12} {score:.4f}")

    print(f"\n  Total candidates : {len(result['candidate_idxs'])}")
    print(f"  Latency          : {result['latency_ms']:.2f} ms")

def benchmark(engine: RetrievalEngine, n_users: int, top_k: int = 100):
    """Benchmark retrieval latency across many users."""
    print_section = lambda t: print(f"\n{'='*60}\n  {t}\n{'='*60}")
    print_section(f"Benchmarking {n_users} Users")

    user_ids = list(engine.sequences.keys())[:n_users]
    latencies = []

    for uid in user_ids:
        r = engine.retrieve(uid, top_k=top_k)
        latencies.append(r['latency_ms'])

    latencies = np.array(latencies)
    print(f"\n  Users tested   : {n_users}")
    print(f"  top_k          : {top_k}")
    print(f"  Avg latency    : {latencies.mean():.2f} ms")
    print(f"  P50 latency    : {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95 latency    : {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 latency    : {np.percentile(latencies, 99):.2f} ms")
    print(f"  Max latency    : {latencies.max():.2f} ms")
    print()
    p95 = np.percentile(latencies, 95)
    if p95 < 20:
        print("  ✅  P95 < 20ms — Excellent production latency")
    elif p95 < 50:
        print("  ✅  P95 < 50ms — Good production latency")
    else:
        print("  ⚠️  P95 > 50ms — Consider optimizing FAISS nprobe or batch encoding")

def main():
    parser = argparse.ArgumentParser(description="DS19 Two-Tower Inference")
    parser.add_argument("--user_id",     type=int,   default=1)
    parser.add_argument("--top_k",       type=int,   default=100)
    parser.add_argument("--show_history",action="store_true")
    parser.add_argument("--benchmark",   action="store_true")
    parser.add_argument("--n_users",     type=int,   default=500)
    args = parser.parse_args()

    print("=" * 64)
    print("  DS19 — Week 4 | Two-Tower Inference")
    print("=" * 64)

    engine = RetrievalEngine()
    engine.load()

    if args.benchmark:
        benchmark(engine, n_users=args.n_users, top_k=args.top_k)
    else:
        demo_single_user(engine, args.user_id, args.top_k, args.show_history)

    print("\n  ✅  Inference complete.")
    print("  ✅  Candidates are ready to be fed into LightGBM Ranker (Week 5)")

if __name__ == "__main__":
    main()