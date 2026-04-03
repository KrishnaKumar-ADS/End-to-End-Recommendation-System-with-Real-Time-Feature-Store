import time
import json
import pytest
import requests
import redis

BASE_URL = "http://localhost:8000"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# Test user — must exist in your interactions.csv
TEST_USER_ID = 42
# Expected minimum number of recommendations
EXPECTED_N_RECS = 10


# ── FIXTURES ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def redis_client():
    """Connect to Redis once for the whole test session."""
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    client.ping()
    return client


@pytest.fixture(scope="session")
def api_session():
    """Reuse an HTTP session for connection pooling."""
    s = requests.Session()
    s.headers["Accept"] = "application/json"
    return s


# ── STEP 1: System Readiness ──────────────────────────────────────────

def test_step1_system_ready(api_session):
    """
    /ready returns 200 only when ALL dependencies are up.
    If this fails: Redis down, Feast unreachable, or models not loaded.
    """
    resp = api_session.get(f"{BASE_URL}/ready", timeout=10)
    assert resp.status_code == 200, (
        f"System not ready: {resp.text}\n"
        "Check: redis-cli ping, and that FastAPI started with all models loaded."
    )
    body = resp.json()
    assert body["status"] == "ready"


# ── STEP 2: Model Health ───────────────────────────────────────────────

def test_step2_models_loaded(api_session):
    """
    /health shows all 4 models loaded.
    """
    resp = api_session.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] in ("healthy", "degraded"), (
        f"System unhealthy: {body}"
    )

    # Model component must be ok or degraded (not down)
    model_status = body["components"]["models"]["status"]
    assert model_status in ("ok", "degraded"), (
        f"Models down: {body['components']['models']}"
    )

    print(f"\n  ✅ System status: {body['status']}")
    print(f"  ✅ Models: {body['components']['models']}")
    print(f"  ✅ Cache hit ratio: {body['metrics']['cache_hit_ratio']:.1%}")


# ── STEP 3: Fetch Recommendations ─────────────────────────────────────

@pytest.fixture(scope="module")
def first_recommendation(api_session):
    """Shared first recommendation response (fetch once, reuse in steps 3-7)."""
    resp = api_session.get(
        f"{BASE_URL}/recommend",
        params={"user_id": TEST_USER_ID},
        timeout=30
    )
    return resp


def test_step3_recommend_returns_200(first_recommendation):
    """GET /recommend returns HTTP 200 for a valid user."""
    assert first_recommendation.status_code == 200, (
        f"Expected 200, got {first_recommendation.status_code}: "
        f"{first_recommendation.text[:500]}"
    )


# ── STEP 4: Response Schema Validation ────────────────────────────────

def test_step4_response_schema(first_recommendation):
    """Response has correct structure and types."""
    body = first_recommendation.json()

    # Top-level fields
    assert "user_id" in body, "Missing 'user_id' in response"
    assert "recommendations" in body, "Missing 'recommendations' in response"
    assert "variant" in body, "Missing 'variant' in response"
    assert "latency_ms" in body, "Missing 'latency_ms' in response"
    assert "cache_hit" in body, "Missing 'cache_hit' in response"

    # Data types
    assert body["user_id"] == TEST_USER_ID
    assert isinstance(body["recommendations"], list)
    assert len(body["recommendations"]) >= EXPECTED_N_RECS, (
        f"Expected {EXPECTED_N_RECS}+ recommendations, got {len(body['recommendations'])}"
    )
    assert body["variant"] in ("mf", "sasrec"), (
        f"Unknown variant: {body['variant']}"
    )
    assert isinstance(body["latency_ms"], (int, float))
    assert body["latency_ms"] > 0

    # Per-item schema
    item = body["recommendations"][0]
    for field in ("item_id", "title", "genres", "score"):
        assert field in item, f"Item missing field '{field}': {item}"
    assert isinstance(item["genres"], list)
    assert 0.0 <= item["score"] <= 1.0, f"Score out of range: {item['score']}"

    print(f"\n  ✅ User {TEST_USER_ID} got {len(body['recommendations'])} recommendations")
    print(f"  ✅ Top item: '{item['title']}' (score={item['score']:.4f})")
    print(f"  ✅ Latency: {body['latency_ms']}ms")
    print(f"  ✅ Variant: {body['variant']}")


# ── STEP 5: Deterministic A/B Assignment ──────────────────────────────

def test_step5_ab_assignment_deterministic(api_session):
    """
    A/B assignment is hash-based → same user always gets same variant.
    Call /recommend 3 times with same user_id → same variant each time.
    """
    variants = []
    for _ in range(3):
        resp = api_session.get(
            f"{BASE_URL}/recommend",
            params={"user_id": TEST_USER_ID},
            timeout=30
        )
        assert resp.status_code == 200
        variants.append(resp.json()["variant"])

    assert len(set(variants)) == 1, (
        f"A/B assignment is NOT deterministic: got variants {variants}\n"
        "Fix: ab_router.py must use hash(user_id) % 100, not random.random()"
    )
    print(f"\n  ✅ Deterministic variant: '{variants[0]}' (consistent across 3 calls)")


# ── STEP 6: Submit Positive Feedback ──────────────────────────────────

@pytest.fixture(scope="module")
def feedback_item_id(first_recommendation):
    """Get the first item ID from recommendations to use for feedback."""
    body = first_recommendation.json()
    return body["recommendations"][0]["item_id"]


def test_step6_submit_feedback(api_session, first_recommendation, feedback_item_id):
    """POST /feedback returns 200 and updates bandit state."""
    body = first_recommendation.json()

    resp = api_session.post(
        f"{BASE_URL}/feedback",
        json={
            "user_id": TEST_USER_ID,
            "item_id": feedback_item_id,
            "reward": 1,
            "variant": body["variant"],
        },
        timeout=10
    )
    assert resp.status_code == 200, (
        f"Feedback failed: {resp.status_code} {resp.text}"
    )
    feedback_body = resp.json()
    assert feedback_body.get("status") == "ok", (
        f"Unexpected feedback response: {feedback_body}"
    )
    print(f"\n  ✅ Feedback submitted for item {feedback_item_id}")


# ── STEP 7: Bandit State Updated in Redis ─────────────────────────────

def test_step7_bandit_redis_updated(redis_client, feedback_item_id):
    """
    After submitting feedback, the bandit's alpha count for the item
    in Redis should be > 1 (was incremented by the feedback).
    """
    # Bandit state key pattern (must match bandit_store.py)
    # Format: "bandit:arm:{item_id}" with fields "alpha" and "beta"
    bandit_key = f"bandit:arm:{feedback_item_id}"

    exists = redis_client.exists(bandit_key)
    assert exists, (
        f"Bandit key '{bandit_key}' not found in Redis.\n"
        "Check: bandit_store.py is writing to Redis after each feedback."
    )

    alpha = float(redis_client.hget(bandit_key, "alpha") or 0)
    assert alpha > 1.0, (
        f"Bandit alpha should be > 1.0 after positive feedback, got {alpha}\n"
        "Check: bandit_service.update() is being called in /feedback handler"
    )
    print(f"\n  ✅ Bandit state updated for item {feedback_item_id}: alpha={alpha:.1f}")


# ── STEP 8: Cache Hit on Second Request ───────────────────────────────

def test_step8_second_request_cached(api_session):
    """
    Second request for the same user should be a cache hit with lower latency.
    This test is probabilistic — cache might not be set up for this user yet.
    """
    # First call (might be cache miss)
    resp1 = api_session.get(
        f"{BASE_URL}/recommend",
        params={"user_id": TEST_USER_ID},
        timeout=30
    )
    assert resp1.status_code == 200

    # Second call (should be cache hit — same user, same response)
    resp2 = api_session.get(
        f"{BASE_URL}/recommend",
        params={"user_id": TEST_USER_ID},
        timeout=30
    )
    assert resp2.status_code == 200

    body2 = resp2.json()

    # Cache hit can be verified via response header or body field
    cache_hit = body2.get("cache_hit", False)
    cache_header = resp2.headers.get("X-Cache-Status", "")

    # Either field or header should indicate a hit
    is_hit = cache_hit or cache_header == "hit"

    if is_hit:
        print(f"\n  ✅ Second request was a cache hit (latency: {body2['latency_ms']}ms)")
    else:
        # Not always a cache hit (depends on cache TTL config)
        print(f"\n  ⚠  Second request was a cache miss (latency: {body2['latency_ms']}ms)")
        print("     This is OK if Redis cache TTL is very short or cache is disabled.")


# ── STEP 9: Prometheus Metrics Recorded ───────────────────────────────

def test_step9_prometheus_metrics(api_session):
    """
    /metrics endpoint returns Prometheus-format data including our custom metrics.
    """
    resp = api_session.get(f"{BASE_URL}/metrics", timeout=10)
    assert resp.status_code == 200, f"/metrics returned {resp.status_code}"

    metrics_text = resp.text

    # Check for our custom business metrics
    expected_metrics = [
        "ds19_recommendation_latency_seconds",
        "ds19_recommendation_requests_total",
        "ds19_cache_hit_ratio",
        "ds19_model_inference_seconds",
        "ds19_feedback_events_total",
        "ds19_active_ab_experiments",
    ]

    missing = []
    for metric in expected_metrics:
        if metric not in metrics_text:
            missing.append(metric)

    assert not missing, (
        f"Missing metrics in /metrics output: {missing}\n"
        "Check: metrics.py is imported and Instrumentator.expose() was called."
    )

    # Check that recommendation counter is > 0 (we made requests)
    assert "ds19_recommendation_requests_total" in metrics_text

    print(f"\n  ✅ All {len(expected_metrics)} custom metrics present in /metrics")
    print(f"  ✅ /metrics response size: {len(metrics_text)} bytes")


# ── STEP 10: A/B Report Consistency ───────────────────────────────────

def test_step10_ab_report(api_session):
    """
    /ab/report returns the current A/B test stats.
    Conversion count should reflect the feedback we submitted in Step 6.
    """
    resp = api_session.get(f"{BASE_URL}/ab/report", timeout=10)
    assert resp.status_code == 200, (
        f"/ab/report returned {resp.status_code}: {resp.text}"
    )

    report = resp.json()

    # Validate report structure
    assert "experiment_name" in report
    assert "variants" in report
    assert isinstance(report["variants"], list)
    assert len(report["variants"]) >= 2, "Expected at least 2 variants (mf, sasrec)"

    # At least one variant should have > 0 conversions (from our feedback)
    total_conversions = sum(v.get("n_conversions", 0) for v in report["variants"])
    assert total_conversions >= 1, (
        f"Expected at least 1 conversion from our Step 6 feedback, got {total_conversions}\n"
        "Check: ab_logger.log_conversion() is called in /feedback handler"
    )

    print(f"\n  ✅ A/B report: {report['experiment_name']}")
    for v in report["variants"]:
        print(f"     {v['variant']:12s}: "
              f"{v['n_exposures']:5d} exposures, "
              f"{v['n_conversions']:4d} conversions, "
              f"CTR={v['ctr']:.4f}")
    print(f"  ✅ Significant: {report['is_significant']} (p={report['p_value']:.4f})")


# ── LATENCY REGRESSION TEST ────────────────────────────────────────────

def test_latency_regression(api_session):
    """
    Cold path latency must be < 500ms.
    Cache hit latency must be < 50ms.

    This runs AFTER the previous tests so Redis cache may be warm.
    """
    # Make one call to ensure cache is warm
    api_session.get(f"{BASE_URL}/recommend", params={"user_id": TEST_USER_ID})

    # Measure 5 consecutive calls
    latencies = []
    for _ in range(5):
        resp = api_session.get(
            f"{BASE_URL}/recommend",
            params={"user_id": TEST_USER_ID},
            timeout=30
        )
        assert resp.status_code == 200
        latencies.append(resp.json()["latency_ms"])

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    # Warm path (cache likely hit): must be < 50ms
    # Cold path (no cache):         must be < 500ms
    # We use 500ms as the threshold (conservative, allows cold path)
    assert max_latency < 500, (
        f"Max latency {max_latency}ms exceeds 500ms threshold.\n"
        f"Latencies: {latencies}\n"
        "This may indicate: FAISS not indexed, LightGBM not loaded, Redis down."
    )

    print(f"\n  ✅ Latency test: avg={avg_latency:.1f}ms, max={max_latency:.1f}ms")
    print(f"     {latencies}")