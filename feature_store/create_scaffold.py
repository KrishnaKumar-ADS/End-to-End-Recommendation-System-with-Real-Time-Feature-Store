from pathlib import Path

DIRS = [
    "feature_store",
    "feature_store/feature_repo",
    "feature_store/pipelines",
    "feature_store/data",
    "feature_store/services",
    "feature_store/training",
    "feature_store/monitoring",
    "feature_store/tests",
]

for d in DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)
    init = Path(d) / "__init__.py"
    if not init.exists():
        init.write_text("")
    print(f"✅ Created: {d}/")

print("\n✅ Scaffold created. Now proceed to Step 3.")