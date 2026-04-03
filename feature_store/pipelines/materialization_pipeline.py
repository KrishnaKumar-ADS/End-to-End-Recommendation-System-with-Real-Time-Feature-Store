"""
DS19 — Week 7: Materialization Pipeline (OFFLINE → ONLINE)
Syncs features from Parquet → Redis via Feast.

Run: python feature_store/pipelines/materialization_pipeline.py
"""

import sys
import time
import redis
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from feast import FeatureStore

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_PATH    = PROJECT_ROOT / "feature_store" / "feature_repo"

REDIS_HOST = "localhost"
REDIS_PORT = 6379


# ─────────────────────────────────────────────────────────────
# STEP 1: CHECK REDIS
# ─────────────────────────────────────────────────────────────

def check_redis_connection():
    print("  Checking Redis connection...")
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=3)
        r.ping()
        info = r.info("memory")
        used_mb = info["used_memory"] / 1024**2
        print(f"  ✅ Redis connected ({REDIS_HOST}:{REDIS_PORT})")
        print(f"  ✅ Redis memory before: {used_mb:.1f} MB")
        return r
    except redis.ConnectionError as e:
        print(f"  ❌ Redis connection failed: {e}")
        print("  Fix: docker run -d -p 6379:6379 redis:alpine")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# STEP 2: LOAD FEAST STORE
# ─────────────────────────────────────────────────────────────

def load_store():
    print(f"  Loading FeatureStore from: {REPO_PATH}")
    try:
        store = FeatureStore(repo_path=str(REPO_PATH))
        print("  ✅ FeatureStore loaded")
        return store
    except Exception as e:
        print(f"  ❌ Failed to load FeatureStore: {e}")
        print("  Fix: cd feature_store/feature_repo && feast apply")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# STEP 3: DETERMINE TIME RANGE
# ─────────────────────────────────────────────────────────────

def get_materialization_time_range():
    """
    Builds a deterministic window from actual parquet timestamps.

    Why this matters:
    - materialize_incremental(end_date=now) + short TTL can exclude old snapshot data
    - using exact data bounds guarantees historical backfill into Redis
    """
    user_parquet = PROJECT_ROOT / "feature_store" / "data" / "user_features.parquet"
    item_parquet = PROJECT_ROOT / "feature_store" / "data" / "item_features.parquet"
    
    if not user_parquet.exists():
        print(f"  ❌ Parquet file not found: {user_parquet}")
        print("  Fix: Run user_features_pipeline.py first")
        sys.exit(1)

    if not item_parquet.exists():
        print(f"  ❌ Parquet file not found: {item_parquet}")
        print("  Fix: Run item_features_pipeline.py first")
        sys.exit(1)
    
    user_df = pd.read_parquet(user_parquet, columns=["event_timestamp"])
    item_df = pd.read_parquet(item_parquet, columns=["event_timestamp"])

    user_ts = pd.to_datetime(user_df["event_timestamp"], utc=True, errors="coerce").dropna()
    item_ts = pd.to_datetime(item_df["event_timestamp"], utc=True, errors="coerce").dropna()

    if user_ts.empty or item_ts.empty:
        print("  ❌ event_timestamp is empty in one or both parquet files")
        print("  Fix: Regenerate features so event_timestamp is populated")
        sys.exit(1)

    user_min = user_ts.min().to_pydatetime()
    user_max = user_ts.max().to_pydatetime()
    item_min = item_ts.min().to_pydatetime()
    item_max = item_ts.max().to_pydatetime()

    min_ts = min(user_min, item_min)
    max_ts = max(user_max, item_max)

    # Add small guard-bands to avoid edge-exclusion on exact boundaries.
    start_date = min_ts - timedelta(days=1)
    end_date = max_ts + timedelta(seconds=1)
    
    print(f"  Feature timestamp range: {min_ts} → {max_ts}")
    print(f"  Materialization window: {start_date} → {end_date}")
    
    return start_date, end_date


# ─────────────────────────────────────────────────────────────
# STEP 4: RUN MATERIALIZATION (BOUNDED FULL BACKFILL)
# ─────────────────────────────────────────────────────────────

def run_materialization(store: FeatureStore, start_date, end_date):
    """
    Uses bounded full materialization for historical snapshot datasets.
    """
    print(f"\n  Starting materialization...")
    print(f"  ⚠️  This will take 60-120 seconds — do not interrupt")
    
    t_start = time.time()
    
    try:
        print("  Running feast.materialize(start_date, end_date)...")
        store.materialize(start_date=start_date, end_date=end_date)
        
        elapsed = time.time() - t_start
        print(f"\n  ✅ Materialization completed in {elapsed:.1f} seconds")
        
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"\n  ❌ Materialization failed after {elapsed:.1f}s: {e}")
        print("\n  Troubleshooting:")
        print("    1. Check Redis: docker ps | findstr redis")
        print("    2. Check registry: cd feature_store/feature_repo && feast feature-views list")
        print("    3. Check Parquet: dir feature_store\\data\\*.parquet")
        raise


# ─────────────────────────────────────────────────────────────
# STEP 5: VERIFY MATERIALIZATION
# ─────────────────────────────────────────────────────────────

def verify_materialization(store: FeatureStore, r: redis.Redis):
    """
    Verifies features were written to Redis.
    """
    print("\n  Verifying materialization...")
    
    # Check Redis key count
    key_count = r.dbsize()
    print(f"  📊 Total Redis keys: {key_count:,}")
    
    if key_count == 0:
        print("  ❌ CRITICAL: No keys in Redis! Materialization failed.")
        print("  Fixes:")
        print("    1. Ensure materialization window covers event_timestamp values")
        print("    2. Ensure FeatureView ttl is not excluding historical data")
        print("    3. Check feature_store.yaml has entity_key_serialization_version: 3")
        sys.exit(1)
    
    # Spot check: fetch sample user features
    print("\n  Spot check — user features...")
    sample_users = [0, 1, 100]
    user_hits = 0
    
    for user_idx in sample_users:
        try:
            result = store.get_online_features(
                entity_rows=[{"user_idx": user_idx}],
                features=[
                    "user_features_view:avg_rating",
                    "user_features_view:total_interactions",
                ]
            ).to_dict()
            
            avg_r = result.get("avg_rating", [None])[0]
            total = result.get("total_interactions", [None])[0]
            
            if avg_r is not None:
                user_hits += 1
                print(f"    ✅ user_idx={user_idx}: avg_rating={avg_r:.2f}, total_int={total}")
            else:
                print(f"    ⚠️  user_idx={user_idx}: features are None")
                
        except Exception as e:
            print(f"    ❌ user_idx={user_idx}: {e}")
    
    # Spot check: fetch sample item features
    print("\n  Spot check — item features...")
    sample_items = [0, 1, 50]
    item_hits = 0
    
    for item_idx in sample_items:
        try:
            result = store.get_online_features(
                entity_rows=[{"item_idx": item_idx}],
                features=[
                    "item_features_view:global_popularity",
                    "item_features_view:avg_item_rating",
                ]
            ).to_dict()
            
            pop = result.get("global_popularity", [None])[0]
            rating = result.get("avg_item_rating", [None])[0]
            
            if pop is not None:
                item_hits += 1
                print(f"    ✅ item_idx={item_idx}: popularity={pop}, avg_rating={rating:.2f}")
            else:
                print(f"    ⚠️  item_idx={item_idx}: features are None")
                
        except Exception as e:
            print(f"    ❌ item_idx={item_idx}: {e}")

    if user_hits == 0 or item_hits == 0:
        print("\n  ❌ CRITICAL: Online features are empty for sampled entities")
        print("  Fixes:")
        print("    1. Re-run feast apply in feature_store/feature_repo")
        print("    2. Verify join keys user_idx/item_idx exist in parquet files")
        print("    3. Re-check materialization time window and FeatureView ttl")
        sys.exit(1)
    
    # Latency benchmark
    print("\n  Latency benchmark (50 requests)...")
    latencies = []
    for i in range(50):
        t0 = time.perf_counter()
        store.get_online_features(
            entity_rows=[{"user_idx": i % 100}],
            features=["user_features_view:avg_rating"]
        ).to_dict()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    print(f"    p50: {p50:.1f}ms, p95: {p95:.1f}ms")
    
    if p95 < 10.0:
        print("  ✅ Latency is excellent (<10ms p95)")
    else:
        print("  ⚠️  Latency is high (>10ms p95)")
    
    return key_count


# ─────────────────────────────────────────────────────────────
# STEP 6: REDIS MEMORY REPORT
# ─────────────────────────────────────────────────────────────

def redis_memory_report(r: redis.Redis):
    info = r.info("memory")
    used_mb = info["used_memory"] / 1024**2
    peak_mb = info["used_memory_peak"] / 1024**2
    print(f"\n  Redis Memory Report:")
    print(f"    Used:  {used_mb:.1f} MB")
    print(f"    Peak:  {peak_mb:.1f} MB")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  DS19 Week 7 — Materialization Pipeline (Offline → Online)")
    print("=" * 65)
    
    print("\n[1/6] Checking Redis...")
    r = check_redis_connection()
    
    print("\n[2/6] Loading Feast store...")
    store = load_store()
    
    print("\n[3/6] Determining time range...")
    start_date, end_date = get_materialization_time_range()
    
    print("\n[4/6] Running materialization...")
    run_materialization(store, start_date, end_date)
    
    print("\n[5/6] Verifying materialization...")
    key_count = verify_materialization(store, r)
    
    print("\n[6/6] Redis memory report...")
    redis_memory_report(r)
    
    print("\n" + "=" * 65)
    print("✅ Materialization pipeline completed successfully!")
    print(f"   Redis keys: {key_count:,} (expected: approx user_entities + item_entities)")
    print("   Online store is ready for FastAPI serving.")
    print("=" * 65)


if __name__ == "__main__":
    main()