import csv
import statistics
from pathlib import Path

REPORTS_DIR = Path("load_testing/reports")

SLA_P95_MS = 200    # target P95 latency
SLA_ERROR_RATE = 1  # max acceptable error rate (%)
MIN_RPS = 50        # minimum acceptable throughput

print("\n" + "=" * 70)
print("  DS19 LOAD TEST ANALYSIS REPORT")
print("=" * 70)


def analyze_csv(csv_prefix: str, label: str):
    """Parse Locust CSV output and compute summary statistics."""
    stats_file = REPORTS_DIR / f"{csv_prefix}_stats.csv"
    if not stats_file.exists():
        print(f"\n  ⚠ {label}: CSV not found ({stats_file}). Run load test first.")
        return

    print(f"\n\n  📊 {label}")
    print(f"  {'─' * 60}")

    with open(stats_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Aggregate (last row is "Aggregated" total)
    for row in rows:
        name = row["Name"]
        if name == "Aggregated":
            total_rps = float(row.get("Requests/s", 0))
            total_errors = float(row.get("Failure Count", 0))
            total_requests = float(row.get("Request Count", 1))
            error_rate = total_errors / max(total_requests, 1) * 100
            p95 = float(row.get("95%", 9999))
            p99 = float(row.get("99%", 9999))
            p50 = float(row.get("50%", 9999))

            sla_pass = (
                p95 <= SLA_P95_MS and
                error_rate <= SLA_ERROR_RATE and
                total_rps >= MIN_RPS
            )

            print(f"  Overall RPS:       {total_rps:.1f} req/s  {'✅' if total_rps >= MIN_RPS else '❌ (target: 50 RPS)'}")
            print(f"  Error rate:        {error_rate:.2f}%       {'✅' if error_rate <= SLA_ERROR_RATE else '❌ (target: <1%)'}")
            print(f"  P50 latency:       {p50:.0f}ms")
            print(f"  P95 latency:       {p95:.0f}ms       {'✅' if p95 <= SLA_P95_MS else '❌ (target: <200ms)'}")
            print(f"  P99 latency:       {p99:.0f}ms")
            print(f"\n  SLA Result: {'✅ PASS' if sla_pass else '❌ FAIL'}")
            break

    print(f"\n  Per-Endpoint Breakdown:")
    for row in rows:
        name = row["Name"]
        if name == "Aggregated":
            continue
        requests = int(row.get("Request Count", 0))
        failures = int(row.get("Failure Count", 0))
        p95_ep = float(row.get("95%", 0))
        avg_ep = float(row.get("Average (ms)", 0))
        print(f"    {name:<35} reqs={requests:5d} | avg={avg_ep:.0f}ms | P95={p95_ep:.0f}ms | err={failures}")


# Analyze all test runs
analyze_csv("sanity", "Sanity Test (5 users, 30s)")
analyze_csv("sla", "SLA Test (50 users, 120s)")
analyze_csv("ramp", "Ramp Test (200 users, 180s)")

print("\n\n  📖 BOTTLENECK GUIDE:")
print("""
  If P95 latency is high:
  ──────────────────────────────────────────────────────────────────
  Check Grafana → ds19_model_inference_seconds:
    "faiss" high  → FAISS nprobe too high, try nprobe=32 instead of 64
    "lgbm" high   → LightGBM n_estimators too high for serving, use <200
    "feast" high  → Redis under load; check Redis memory + connection pool
    "bandit" high → Thompson Sampling in pure Python; consider numpy batch

  Check Grafana → ds19_cache_hit_ratio:
    < 20%  → Cache barely working; check Redis TTL settings
    > 80%  → Most requests served from cache; good, but check diversity

  Redis connection pool exhaustion:
    Error: "Too many connections"
    Fix: In main.py, set redis.ConnectionPool(max_connections=50)
    Also: In docker-compose.yml, add: command: redis-server --maxclients 1000

  FAISS saturating CPU:
    Fix: Run FAISS with multiple threads:
         import faiss; faiss.omp_set_num_threads(4)
    Fix: Use FAISS GPU index if CUDA available (RTX 3050 has 4GB VRAM)
""")
print("=" * 70 + "\n")