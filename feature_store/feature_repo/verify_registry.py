import sys
from pathlib import Path
from feast import FeatureStore

# feature_store.yaml is in the current directory
REPO_PATH = Path(__file__).resolve().parent


def main():
    print("=" * 65)
    print("  DS19 Week 7 — Feast Registry Verification")
    print("=" * 65)

    try:
        store = FeatureStore(repo_path=str(REPO_PATH))
        print("\n✅ FeatureStore initialized successfully")
    except Exception as e:
        print(f"\n❌ Failed to initialize FeatureStore: {e}")
        print("   Did you run 'feast apply' first?")
        sys.exit(1)

    # ── Check Entities ──
    print("\n📌 Entities:")
    entities = store.list_entities()
    expected_entities = {"user_entity", "item_entity"}
    found_entities = {e.name for e in entities}
    for entity in entities:
        print(f"  ✅ {entity.name}  (join_key: {entity.join_key})")
    missing_entities = expected_entities - found_entities
    if missing_entities:
        print(f"  ❌ Missing entities: {missing_entities}")

    # ── Check Feature Views ──
    print("\n📊 Feature Views:")
    views = store.list_feature_views()
    expected_views = {"user_features_view", "item_features_view"}
    found_views = {v.name for v in views}
    for view in views:
        n_features = len(view.features)
        print(f"  ✅ {view.name}  ({n_features} features, ttl={view.ttl})")
        for feat in view.features:
            print(f"       {feat.name}: {feat.dtype}")
    missing_views = expected_views - found_views
    if missing_views:
        print(f"  ❌ Missing feature views: {missing_views}")

    # ── Check Feature Services ──
    print("\n🎯 Feature Services:")
    services = store.list_feature_services()
    expected_services = {"ranking_feature_service", "user_profile_service"}
    found_services = {s.name for s in services}
    for svc in services:
        print(f"  ✅ {svc.name}")
    missing_services = expected_services - found_services
    if missing_services:
        print(f"  ❌ Missing services: {missing_services}")

    # ── Final Summary ──
    all_ok = (
        not missing_entities and
        not missing_views and
        not missing_services
    )

    print("\n" + "=" * 65)
    if all_ok:
        print("✅ Registry verification PASSED")
        print("   Next: run materialization to sync offline → online (Redis)")
    else:
        print("❌ Registry verification FAILED — re-run 'feast apply'")
        sys.exit(1)
    print("=" * 65)


if __name__ == "__main__":
    main()