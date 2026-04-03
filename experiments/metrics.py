import numpy as np
from typing import List, Dict, Union

# ─────────────────────────────────────────────────────────────
# HIT RATE @ K
# ─────────────────────────────────────────────────────────────

def hit_rate_at_k(rank: Union[int, np.ndarray], k: int) -> float:
    """
    Hit Rate @ K for a SINGLE user or array of users.

    HR@K = 1  if ground truth item rank <= K
           0  otherwise

    Args:
        rank : 1-indexed rank of the ground truth item in the ranked list.
               rank=1 means it's the top recommendation.
               rank=50000 means it's at the very bottom.
               Can be a scalar (int) or array of ranks for multiple users.
        k    : cutoff value (e.g. 5, 10, 20)

    Returns:
        1.0/0.0 for scalar, or mean hit rate across all users for array

    Example:
        hit_rate_at_k(rank=3, k=10)  → 1.0  (rank 3 is in top 10)
        hit_rate_at_k(rank=15, k=10) → 0.0  (rank 15 is NOT in top 10)
        hit_rate_at_k(rank=np.array([1, 5, 15]), k=10) → 0.6667
    """
    rank_arr = np.asarray(rank)
    
    if rank_arr.ndim == 0:
        # Scalar case
        return 1.0 if rank_arr.item() <= k else 0.0
    
    # Array case: return mean hit rate
    return float(np.mean(rank_arr <= k))


# ─────────────────────────────────────────────────────────────
# NDCG @ K
# ─────────────────────────────────────────────────────────────

def ndcg_at_k(rank: Union[int, np.ndarray], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K for a SINGLE user or array of users.

    With binary relevance (1 relevant item, 0 otherwise):

        NDCG@K = 1 / log2(rank + 1)   if rank <= K
               = 0                     otherwise

    The "Ideal DCG" (IDCG) for one relevant item is: 1 / log2(1+1) = 1.0
    So NDCG = DCG / IDCG = DCG / 1.0 = DCG

    Discount interpretation:
        rank=1 → NDCG = 1.000   (perfect, #1 recommendation)
        rank=2 → NDCG = 0.631   (second place, 37% penalty)
        rank=3 → NDCG = 0.500   (third place, 50% penalty)
        rank=5 → NDCG = 0.387
        rank=10→ NDCG = 0.289

    Args:
        rank : 1-indexed rank of ground truth item (scalar or array)
        k    : cutoff value

    Returns:
        NDCG score for this user, or mean NDCG for array of users

    Example:
        ndcg_at_k(rank=1, k=10)  → 1.000
        ndcg_at_k(rank=5, k=10)  → 0.387
        ndcg_at_k(rank=15, k=10) → 0.000
    """
    rank_arr = np.asarray(rank)
    
    if rank_arr.ndim == 0:
        # Scalar case
        r = rank_arr.item()
        return 1.0 / np.log2(r + 1) if r <= k else 0.0
    
    # Array case
    scores = np.where(rank_arr <= k, 1.0 / np.log2(rank_arr + 1), 0.0)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# MRR @ K
# ─────────────────────────────────────────────────────────────

def mrr_at_k(rank: Union[int, np.ndarray], k: int) -> float:
    """
    Mean Reciprocal Rank @ K for a SINGLE user or array of users.

    MRR = 1 / rank   if rank <= K
        = 0           otherwise

    Interpretation:
        rank=1 → 1.000
        rank=2 → 0.500
        rank=5 → 0.200
        rank=10→ 0.100

    Args:
        rank : 1-indexed rank of ground truth item (scalar or array)
        k    : cutoff value

    Returns:
        Reciprocal rank score, or mean MRR for array of users
    """
    rank_arr = np.asarray(rank)
    
    if rank_arr.ndim == 0:
        # Scalar case
        r = rank_arr.item()
        return 1.0 / r if r <= k else 0.0
    
    # Array case
    scores = np.where(rank_arr <= k, 1.0 / rank_arr, 0.0)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# PRECISION @ K
# ─────────────────────────────────────────────────────────────

def precision_at_k(rank: Union[int, np.ndarray], k: int) -> float:
    """
    Precision @ K for a SINGLE user with ONE relevant item or array of users.

    P@K = (# relevant items in top K) / K
        = 1/K   if rank <= K
        = 0     otherwise

    Note: With exactly 1 relevant item, Precision@K is less informative
    than HR@K or NDCG@K. But useful for multi-label relevance scenarios.
    """
    rank_arr = np.asarray(rank)
    
    if rank_arr.ndim == 0:
        # Scalar case
        r = rank_arr.item()
        return 1.0 / k if r <= k else 0.0
    
    # Array case
    scores = np.where(rank_arr <= k, 1.0 / k, 0.0)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# AGGREGATE METRICS (over a list of users)
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    ranks    : List[int],
    k_values : List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Computes aggregate HR@K, NDCG@K, MRR@K over a list of users.

    Args:
        ranks    : list of ground truth ranks (one per user, 1-indexed)
                   e.g. [3, 15, 1, 42, 8, 200, ...]
        k_values : list of K cutoffs to evaluate at

    Returns:
        metrics: {
            'HR@5'  : 0.62,
            'HR@10' : 0.71,
            'HR@20' : 0.80,
            'NDCG@5' : 0.45,
            'NDCG@10': 0.48,
            'NDCG@20': 0.50,
            'MRR@10' : 0.39,
            'n_users': 162541,
        }

    Example:
        ranks = [1, 3, 50, 2, 8, 15, 100, 7]
        metrics = compute_metrics(ranks, k_values=[5, 10])
        # metrics['HR@10'] = fraction of users where rank <= 10
    """
    n = len(ranks)
    if n == 0:
        return {}

    metrics = {}
    ranks_arr = np.array(ranks)

    for k in k_values:
        # HR@K: count of users where ground truth in top K, divided by total users
        hr = np.mean(ranks_arr <= k)
        metrics[f'HR@{k}'] = float(hr)

        # NDCG@K: average discounted gain
        ndcg_scores = np.where(
            ranks_arr <= k,
            1.0 / np.log2(ranks_arr + 1),
            0.0
        )
        metrics[f'NDCG@{k}'] = float(ndcg_scores.mean())

        # MRR@K
        mrr_scores = np.where(
            ranks_arr <= k,
            1.0 / ranks_arr,
            0.0
        )
        metrics[f'MRR@{k}'] = float(mrr_scores.mean())

    metrics['n_users']   = n
    metrics['mean_rank'] = float(np.mean(ranks_arr))
    metrics['median_rank'] = float(np.median(ranks_arr))

    return metrics


# ─────────────────────────────────────────────────────────────
# RANK COMPUTATION FROM SCORES
# ─────────────────────────────────────────────────────────────

def scores_to_rank(
    scores     : np.ndarray,   # [n_items] raw scores for all items
    gt_item    : int,          # ground truth item index
    mask_items : set = None,   # set of item indices to mask out (e.g., training history)
) -> int:
    """
    Converts raw item scores to the rank of a specific ground truth item.

    Steps:
      1. Copy scores (don't modify original)
      2. Mask out training-history items (set to -inf)
      3. Mask out PAD token (index 0)
      4. argsort descending → ranked_items
      5. Find position of gt_item in ranked_items → return rank (1-indexed)

    Args:
        scores     : [n_items] array of raw scores (higher = more relevant)
        gt_item    : the ground truth item index we're looking for
        mask_items : set of item indices to exclude from ranking (user's training history)

    Returns:
        rank: 1-indexed rank of gt_item (1 = top recommendation)
    """
    scores_copy = scores.copy()

    # Mask PAD token
    scores_copy[0] = -np.inf

    # Mask training history
    if mask_items:
        for item in mask_items:
            if 0 <= item < len(scores_copy):
                scores_copy[item] = -np.inf

    # Rank items descending
    ranked_items = np.argsort(-scores_copy)

    # Find ground truth item position
    positions = np.where(ranked_items == gt_item)[0]

    if len(positions) == 0:
        # Should never happen — return worst rank
        return len(scores_copy)

    return int(positions[0]) + 1   # Convert to 1-indexed


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Metrics Smoke Test")
    print("=" * 60)

    # ── Single-user tests ─────────────────────────────────────
    print("\n  HR@K:")
    for rank, k in [(1,10), (5,10), (10,10), (11,10), (20,10)]:
        hr = hit_rate_at_k(rank, k)
        print(f"    rank={rank:3d}, k={k} → HR@{k} = {hr:.1f}")

    print("\n  NDCG@K:")
    for rank in [1, 2, 3, 5, 10, 11]:
        ndcg = ndcg_at_k(rank, k=10)
        print(f"    rank={rank:3d}, k=10 → NDCG@10 = {ndcg:.4f}")

    print("\n  MRR@K:")
    for rank in [1, 2, 5, 10, 11]:
        mrr = mrr_at_k(rank, k=10)
        print(f"    rank={rank:3d}, k=10 → MRR@10 = {mrr:.4f}")

    # ── Aggregate metrics test ─────────────────────────────────
    print("\n  Aggregate Metrics Test:")
    # Mock ranks for 10 users
    mock_ranks = [1, 3, 5, 7, 12, 2, 4, 50, 9, 100]
    metrics = compute_metrics(mock_ranks, k_values=[5, 10, 20])

    for metric, val in metrics.items():
        if isinstance(val, float):
            print(f"    {metric:<15} : {val:.4f}")
        else:
            print(f"    {metric:<15} : {val}")

    # Manual verification:
    # ranks <= 10: [1,3,5,7,2,4,9] = 7 out of 10 → HR@10 = 0.70
    expected_hr10 = sum(1 for r in mock_ranks if r <= 10) / len(mock_ranks)
    assert abs(metrics['HR@10'] - expected_hr10) < 1e-6, "HR@10 calculation error!"
    print(f"\n  ✅ HR@10 verified: {metrics['HR@10']:.4f} == {expected_hr10:.4f}")

    # ── scores_to_rank test ────────────────────────────────────
    print("\n  scores_to_rank test:")
    mock_scores = np.array([0.0, 0.9, 0.1, 0.95, 0.7, 0.3, 0.6, 0.5])
    # Item 3 has score 0.95 (highest), so rank=1
    rank_of_3 = scores_to_rank(mock_scores, gt_item=3)
    print(f"    gt_item=3 (score=0.95, highest) → rank={rank_of_3}  (expected: 1)")
    assert rank_of_3 == 1, f"Expected rank 1, got {rank_of_3}"

    # Item 1 has score 0.9 (second highest), so rank=2
    rank_of_1 = scores_to_rank(mock_scores, gt_item=1)
    print(f"    gt_item=1 (score=0.90)          → rank={rank_of_1}  (expected: 2)")
    assert rank_of_1 == 2, f"Expected rank 2, got {rank_of_1}"

    # With masking: mask out item 3 → item 1 becomes rank 1
    rank_of_1_masked = scores_to_rank(mock_scores, gt_item=1, mask_items={3})
    print(f"    gt_item=1 (masked out item 3)   → rank={rank_of_1_masked}  (expected: 1)")
    assert rank_of_1_masked == 1, f"Expected rank 1 after masking, got {rank_of_1_masked}"

    print(f"\n  ✅ All metric tests PASSED")