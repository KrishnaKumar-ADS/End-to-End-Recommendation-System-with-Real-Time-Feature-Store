import random
import json
import time
from locust import HttpUser, task, between, events
from locust.exception import StopUser

# ── USER POOLS ─────────────────────────────────────────────────────
# Sample from MovieLens 25M user range (1–162,541)
# Using known users from your interactions.csv for cache-realistic testing.
# Mix of:
#   - Same 10 users repeatedly   → tests cache effectiveness
#   - Random users from pool     → tests cold-path pipeline
CACHED_USERS = [1, 42, 100, 500, 1000, 2000, 5000, 10000, 50000, 100000]
RANDOM_USER_POOL = list(range(1, 10000))   # users likely in your dataset


def pick_user_id(use_cache: bool = False) -> int:
    """
    Select a user ID for the request.
    use_cache=True  → pick from CACHED_USERS (likely cache hit)
    use_cache=False → pick random (likely cache miss, tests full pipeline)
    """
    if use_cache and random.random() < 0.6:
        # 60% of the time: use a known user → triggers cache hit
        return random.choice(CACHED_USERS)
    else:
        # 40% of the time: use random user → triggers cold path
        return random.choice(RANDOM_USER_POOL)


# ══════════════════════════════════════════════════════════════════════
# USER TYPE 1: Basic Recommendation Fetcher
# Simulates: user opens app, sees recommendations.
# Weight: 60% of all simulated users.
# ══════════════════════════════════════════════════════════════════════
class RecommendationUser(HttpUser):
    """
    Models a passive user who only browses recommendations.
    Represents the majority of users.

    wait_time: between(1, 3)
      → Each virtual user waits 1-3 seconds between requests.
      → Simulates human reading time between page views.
      → 50 users × 1 req / 2s average = ~25 RPS (realistic baseline)
    """
    weight = 3          # 3x more common than FeedbackUser
    wait_time = between(1, 3)

    def on_start(self):
        """Called once when each simulated user starts."""
        self.user_id = pick_user_id(use_cache=False)
        self.session_start = time.time()

    @task(10)
    def get_recommendations(self):
        """
        Primary task: fetch recommendations.
        Weight 10 = 10x more frequent than the health check task.
        """
        with self.client.get(
            "/recommend",
            params={"user_id": self.user_id},
            catch_response=True,
            name="GET /recommend (cold)",
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Expected 200, got {response.status_code}: {response.text[:100]}"
                )
                return

            data = response.json()

            # Validate response structure
            if "recommendations" not in data:
                response.failure("Missing 'recommendations' key in response")
                return

            if len(data["recommendations"]) == 0:
                response.failure(f"Empty recommendations for user {self.user_id}")
                return

            # Check latency SLA at the application level
            if data.get("latency_ms", 9999) > 500:
                response.failure(
                    f"Application-level latency {data['latency_ms']}ms > 500ms SLA"
                )
                return

            response.success()

            # Occasionally switch to a different user (simulates session change)
            if random.random() < 0.1:
                self.user_id = pick_user_id(use_cache=False)

    @task(3)
    def get_recommendations_cached(self):
        """
        Fetch recommendations for a popular user (likely cache hit).
        Tests cache-hit path performance.
        """
        user_id = pick_user_id(use_cache=True)
        with self.client.get(
            "/recommend",
            params={"user_id": user_id},
            catch_response=True,
            name="GET /recommend (cache warm)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Cache warm request failed: {response.status_code}")

    @task(1)
    def check_health(self):
        """
        Occasional health check (monitoring simulation).
        Weight 1 = least frequent.
        """
        self.client.get("/health", name="GET /health")


# ══════════════════════════════════════════════════════════════════════
# USER TYPE 2: Engaged User (View + Click)
# Simulates: user sees recommendations AND clicks on them.
# Weight: 20% of simulated users. Generates POST /feedback load.
# ══════════════════════════════════════════════════════════════════════
class FeedbackUser(HttpUser):
    """
    Models an engaged user who both views and rates recommendations.
    This is your most valuable user for the learning loop.
    More expensive per user (2 API calls), so weight is lower.
    """
    weight = 1
    wait_time = between(2, 5)   # engaged users read more carefully

    def on_start(self):
        self.user_id = pick_user_id(use_cache=False)
        self.last_items = []
        self.last_variant = "unknown"

    @task
    def view_and_click(self):
        """
        Realistic session: fetch recommendations → click on one.
        Simulates positive engagement (reward=1).
        """
        # Step 1: Fetch recommendations
        with self.client.get(
            "/recommend",
            params={"user_id": self.user_id},
            catch_response=True,
            name="GET /recommend (engaged user)",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Recommend failed: {response.status_code}")
                return

            data = response.json()
            self.last_items = [r["item_id"] for r in data["recommendations"]]
            self.last_variant = data.get("variant", "unknown")
            response.success()

        if not self.last_items:
            return

        # Step 2: Simulate reading time (0.5–2 seconds)
        time.sleep(random.uniform(0.5, 2.0))

        # Step 3: Click on a recommendation (positive feedback)
        # Bias: click items at rank 1-3 more often (realistic engagement)
        click_weights = [5, 4, 3, 2, 2, 1, 1, 1, 1, 1][:len(self.last_items)]
        clicked_item = random.choices(self.last_items, weights=click_weights, k=1)[0]

        with self.client.post(
            "/feedback",
            json={
                "user_id": self.user_id,
                "item_id": clicked_item,
                "reward": 1,
                "variant": self.last_variant,
            },
            catch_response=True,
            name="POST /feedback (click)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Feedback failed: {response.status_code}")

    @task
    def view_and_skip(self):
        """
        User views but doesn't click (no-click signal).
        Sends reward=0 feedback for explicit dislike.
        """
        with self.client.get(
            "/recommend",
            params={"user_id": self.user_id},
            catch_response=True,
            name="GET /recommend (skip user)",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Recommend failed: {response.status_code}")
                return
            data = response.json()
            items = [r["item_id"] for r in data["recommendations"]]
            variant = data.get("variant", "unknown")
            response.success()

        if not items:
            return

        # Send explicit dislike on last item (usually least relevant)
        disliked_item = items[-1]
        self.client.post(
            "/feedback",
            json={
                "user_id": self.user_id,
                "item_id": disliked_item,
                "reward": 0,
                "variant": variant,
            },
            name="POST /feedback (dislike)",
        )


# ══════════════════════════════════════════════════════════════════════
# USER TYPE 3: A/B Dashboard Poller
# Simulates: data scientist checking A/B results every 30 seconds.
# ══════════════════════════════════════════════════════════════════════
class DashboardUser(HttpUser):
    """
    Simulates a data scientist monitoring the A/B experiment.
    Polls /ab/report + /health infrequently.
    """
    weight = 1
    wait_time = between(25, 35)   # check roughly every 30 seconds

    @task(3)
    def check_ab_report(self):
        self.client.get("/ab/report", name="GET /ab/report")

    @task(2)
    def check_health(self):
        self.client.get("/health", name="GET /health")

    @task(1)
    def check_metrics(self):
        """
        Simulate Prometheus scrape (though Prometheus does this natively).
        Tests /metrics endpoint doesn't degrade under concurrent load.
        """
        self.client.get("/metrics", name="GET /metrics")


# ══════════════════════════════════════════════════════════════════════
# LOCUST EVENT HOOKS — Custom metrics + reporting
# ══════════════════════════════════════════════════════════════════════

# Tracking custom per-endpoint metrics
_endpoint_latencies: dict = {}
_endpoint_counts: dict = {}
_endpoint_errors: dict = {}


@events.request.add_listener
def on_request(
    request_type, name, response_time, response_length,
    response, context, exception, **kwargs
):
    """
    Called for every request. Records per-endpoint statistics.
    Separate from Locust's built-in stats — used for custom CSV output.
    """
    if name not in _endpoint_latencies:
        _endpoint_latencies[name] = []
        _endpoint_counts[name] = {"success": 0, "failure": 0}

    _endpoint_latencies[name].append(response_time)

    if exception or (response and response.status_code >= 400):
        _endpoint_counts[name]["failure"] += 1
    else:
        _endpoint_counts[name]["success"] += 1


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when load test ends. Prints custom summary.
    """
    import statistics

    print("\n" + "=" * 70)
    print("  DS19 LOAD TEST — CUSTOM SUMMARY REPORT")
    print("=" * 70)

    SLA_P95_MS = 200   # our target P95 latency

    all_pass = True
    for name, latencies in sorted(_endpoint_latencies.items()):
        if not latencies:
            continue

        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        counts = _endpoint_counts.get(name, {})
        total = counts.get("success", 0) + counts.get("failure", 0)
        error_rate = counts.get("failure", 0) / max(total, 1) * 100

        sla_pass = p95 <= SLA_P95_MS
        if not sla_pass:
            all_pass = False

        status = "✅" if sla_pass else "❌"
        print(f"\n  {status} {name}")
        print(f"     Requests: {total} | Errors: {error_rate:.1f}%")
        print(f"     P50: {p50:.0f}ms | P95: {p95:.0f}ms | P99: {p99:.0f}ms")
        if not sla_pass:
            print(f"     ⚠ SLA BREACH: P95 {p95:.0f}ms > {SLA_P95_MS}ms target")

    print("\n" + "=" * 70)
    if all_pass:
        print("  🎉 ALL ENDPOINTS PASSED SLA (P95 < 200ms)")
    else:
        print("  ❌ SLA BREACHED — Investigate bottlenecks")
        print("     Steps:")
        print("     1. Check ds19_model_inference_seconds in Grafana")
        print("     2. Run: redis-cli slowlog get 10")
        print("     3. Check FAISS index nprobe setting")
    print("=" * 70 + "\n")