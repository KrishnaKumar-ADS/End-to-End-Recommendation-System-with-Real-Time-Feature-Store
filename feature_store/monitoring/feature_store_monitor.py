import sys
import time
import numpy as np
import pandas as pd
import redis
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_PATH    = PROJECT_ROOT / "feature_store" / "feature_repo"

REDIS_HOST = "localhost"
REDIS_PORT = 6379


def check_1_redis():
    """Check Redis is up and responsive."""
    print("\n[CHECK 1] Redis Connectivity")
    print("─" * 50)
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
        t0 = time.perf_counter()
        r.ping()
        ping_ms = (time.perf_counter() - t0) * 1000

        info = r.info("memory")
        used_mb  = info["used_memory"] / 1024**2
        peak_mb  = info["used_memory_peak"] / 1024**2

        info_srv = r.info("server")
        redis_version = info_srv.get("redis_version", "?")

        print(f"  ✅ Redis {redis_version} at {REDIS_HOST}:{REDIS_PORT}")
        print(f"  ✅ PING latency: {ping_ms:.2f}ms")
        print(f"  ✅ Memory used: {used_mb:.1f} MB (peak: {peak_mb:.1f} MB)")

        # Count keys
        n_keys = r.dbsize()
        print(f"  ✅ Total keys: {n_keys:,}")

        if n_keys == 0:
            print("  ⚠️  No keys in Redis — run materialization first!")
            print("     python feature_store/pipelines/materialization_pipeline.py")

        return True, r
    except Exception as e:
        print(f"  ❌ Redis check failed: {e}")
        return False, None


def check_2_feast_registry(store):
    """Check Feast registry has all required objects."""
    print("\n[CHECK 2] Feast Registry Integrity")
    print("─" * 50)

    views = {v.name: v for v in store.list_feature_views()}
    entities = {e.name: e for e in store.list_entities()}
    services = {s.name: s for s in store.list_feature_services()}

    required_views = {"user_features_view", "item_features_view"}
    required_entities = {"user_entity", "item_entity"}
    required_services = {"ranking_feature_service"}

    all_ok = True
    for name in required_views:
        if name in views:
            n_feats = len(views[name].features)
            print(f"  ✅ FeatureView: {name} ({n_feats} features)")
        else:
            print(f"  ❌ Missing FeatureView: {name}")
            all_ok = False

    for name in required_entities:
        if name in entities:
            print(f"  ✅ Entity: {name}")
        else:
            print(f"  ❌ Missing Entity: {name}")
            all_ok = False

    for name in required_services:
        if name in services:
            print(f"  ✅ FeatureService: {name}")
        else:
            print(f"  ❌ Missing FeatureService: {name}")
            all_ok = False

    return all_ok


def check_3_online_population(store):
    """Verify online store has data for sample users and items."""
    print("\n[CHECK 3] Online Store Population")
    print("─" * 50)

    test_users = [0, 1, 10, 100, 1000]
    test_items = [0, 1, 10, 100, 500]

    # Check users
    user_ok_count = 0
    for uid in test_users:
        try:
            result = store.get_online_features(
                entity_rows=[{"user_idx": uid}],
                features=["user_features_view:avg_rating",
                          "user_features_view:total_interactions"]
            ).to_dict()
            avg_r = result.get("avg_rating", [None])[0]
            if avg_r is not None:
                user_ok_count += 1
                print(f"  ✅ user_idx={uid}: avg_rating={avg_r:.2f}")
            else:
                print(f"  ⚠️  user_idx={uid}: features are None (not materialized)")
        except Exception as e:
            print(f"  ❌ user_idx={uid}: {e}")

    # Check items
    item_ok_count = 0
    for iid in test_items:
        try:
            result = store.get_online_features(
                entity_rows=[{"item_idx": iid}],
                features=["item_features_view:global_popularity",
                          "item_features_view:avg_item_rating"]
            ).to_dict()
            pop = result.get("global_popularity", [None])[0]
            if pop is not None:
                item_ok_count += 1
                print(f"  ✅ item_idx={iid}: popularity={pop}")
            else:
                print(f"  ⚠️  item_idx={iid}: features are None")
        except Exception as e:
            print(f"  ❌ item_idx={iid}: {e}")

    print(f"\n  Users populated: {user_ok_count}/{len(test_users)}")
    print(f"  Items populated: {item_ok_count}/{len(test_items)}")
    return user_ok_count >= 3 and item_ok_count >= 3


def check_4_feature_staleness():
    """Check how old the offline Parquet files are."""
    print("\n[CHECK 4] Feature Staleness")
    print("─" * 50)

    parquet_files = {
        "User features": PROJECT_ROOT / "feature_store" / "data" / "user_features.parquet",
        "Item features": PROJECT_ROOT / "feature_store" / "data" / "item_features.parquet",
    }

    all_ok = True
    for name, path in parquet_files.items():
        if not path.exists():
            print(f"  ❌ {name}: file not found ({path})")
            all_ok = False
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_days = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 86400

        size_mb = path.stat().st_size / 1024**2

        if age_days < 1:
            status = "✅ Fresh (< 1 day)"
        elif age_days < 7:
            status = "✅ Recent (< 1 week)"
        elif age_days < 30:
            status = "⚠️  Getting stale (< 1 month)"
        else:
            status = "❌ Stale (> 1 month) — consider re-running pipeline"
            all_ok = False

        print(f"  {name}: {size_mb:.1f} MB, age={age_days:.1f} days — {status}")

    return all_ok


def check_5_latency_benchmark(store):
    """Benchmarks online feature retrieval latency."""
    print("\n[CHECK 5] Online Feature Retrieval Latency")
    print("─" * 50)

    # Single-entity latency
    single_latencies = []
    for i in range(50):
        t0 = time.perf_counter()
        store.get_online_features(
            entity_rows=[{"user_idx": i % 500}],
            features=["user_features_view:avg_rating",
                      "user_features_view:total_interactions",
                      "user_features_view:genre_diversity"]
        )
        single_latencies.append((time.perf_counter() - t0) * 1000)

    # Batch entity latency (simulating 100-item ranking feature fetch)
    batch_latencies = []
    for i in range(20):
        item_idxs = list(range(i * 100, i * 100 + 100))
        t0 = time.perf_counter()
        store.get_online_features(
            entity_rows=[{"item_idx": idx} for idx in item_idxs],
            features=["item_features_view:global_popularity",
                      "item_features_view:avg_item_rating",
                      "item_features_view:primary_genre_idx"]
        )
        batch_latencies.append((time.perf_counter() - t0) * 1000)

    print(f"  Single-entity (user, 3 features) — 50 requests:")
    print(f"    p50: {np.percentile(single_latencies, 50):.1f}ms")
    print(f"    p95: {np.percentile(single_latencies, 95):.1f}ms")
    print(f"    p99: {np.percentile(single_latencies, 99):.1f}ms")

    print(f"\n  Batch-entity (100 items, 3 features) — 20 requests:")
    print(f"    p50: {np.percentile(batch_latencies, 50):.1f}ms")
    print(f"    p95: {np.percentile(batch_latencies, 95):.1f}ms")
    print(f"    p99: {np.percentile(batch_latencies, 99):.1f}ms")

    target_single = 5.0
    target_batch  = 20.0

    if np.percentile(single_latencies, 95) < target_single:
        print(f"\n  ✅ Single-entity p95 < {target_single}ms — target met")
    else:
        print(f"\n  ⚠️  Single-entity p95 > {target_single}ms — Redis may be under load")

    if np.percentile(batch_latencies, 95) < target_batch:
        print(f"  ✅ Batch-entity p95 < {target_batch}ms — target met")
    else:
        print(f"  ⚠️  Batch-entity p95 > {target_batch}ms — consider pipelining")


def main():
    print("=" * 65)
    print("  DS19 Week 7 — Feature Store Health Monitor")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Check 1: Redis
    redis_ok, r = check_1_redis()
    if not redis_ok:
        print("\n❌ CRITICAL: Redis is not running. All other checks will fail.")
        print("   Fix: docker run -d -p 6379:6379 redis:alpine")
        sys.exit(1)

    # Load Feast store
    try:
        from feast import FeatureStore
        store = FeatureStore(repo_path=str(REPO_PATH))
    except Exception as e:
        print(f"\n❌ Failed to load FeatureStore: {e}")
        print("   Did you run: cd feature_store/feature_repo && feast apply")
        sys.exit(1)

    # Checks 2-5
    registry_ok    = check_2_feast_registry(store)
    population_ok  = check_3_online_population(store)
    staleness_ok   = check_4_feature_staleness()
    check_5_latency_benchmark(store)

    # Final summary
    print("\n" + "=" * 65)
    print("  HEALTH CHECK SUMMARY")
    print("=" * 65)

    checks = {
        "Redis connectivity":     redis_ok,
        "Feast registry":         registry_ok,
        "Online store populated": population_ok,
        "Feature freshness":      staleness_ok,
    }

    all_ok = True
    for check, ok in checks.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {check}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  🎉 Feature store is fully healthy!")
        print("     Ready for production serving.")
    else:
        print("  ⚠️  Some checks failed — investigate before serving in production.")

    print("=" * 65)


if __name__ == "__main__":
    main()