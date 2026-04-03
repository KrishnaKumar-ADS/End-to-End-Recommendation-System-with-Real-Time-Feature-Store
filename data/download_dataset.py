import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
RAW_DIR     = Path("data/raw")
ZIP_PATH    = RAW_DIR / "ml-25m.zip"

# ─────────────────────────────────────────────────────────────
# DOWNLOAD PROGRESS
# ─────────────────────────────────────────────────────────────

def show_progress(block_num, block_size, total_size):
    """Shows download progress in terminal."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(
        f"\r  Downloading: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)"
    )
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DS19 — MovieLens 25M Dataset Downloader")
    print("=" * 60)

    # Create directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    ratings_path = RAW_DIR / "ratings.csv"
    if ratings_path.exists():
        print("✅ Dataset already downloaded!")
        print(f"   Location: {RAW_DIR.resolve()}")
        return

    # Download
    print(f"\nSource : {DATASET_URL}")
    print(f"Target : {ZIP_PATH}")
    print()

    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH, reporthook=show_progress)
        print("\n\n✅ Download complete!")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("Please download manually from:")
        print("  https://grouplens.org/datasets/movielens/25m/")
        sys.exit(1)

    # Extract
    print("\n📦 Extracting files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        for file in zip_ref.namelist():
            # Extract only CSVs to data/raw/ directly (skip subfolder)
            if file.endswith(".csv"):
                filename = Path(file).name
                target   = RAW_DIR / filename
                with zip_ref.open(file) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                print(f"  ✅ Extracted: {filename}")

    # Remove zip
    ZIP_PATH.unlink()
    print("\n🗑️  Cleaned up zip file.")

    # Verify
    print("\n📋 Files in data/raw/:")
    for f in sorted(RAW_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:<30} {size_mb:>8.1f} MB")

    print("\n✅ Dataset ready!")


if __name__ == "__main__":
    main()