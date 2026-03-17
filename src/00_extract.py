"""
00_extract.py — Extract Yelp dataset from zip and tar archives.

Usage:
    python src/00_extract.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "Yelp-JSON.zip"
RAW_ZIP_DIR = PROJECT_ROOT / "data" / "raw_zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = [
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_user.json",
    "yelp_academic_dataset_checkin.json",
    "yelp_academic_dataset_tip.json",
]


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, **kwargs)
    return result


def main():
    RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: unzip the outer .zip
    if not any(RAW_ZIP_DIR.iterdir()) if RAW_ZIP_DIR.exists() else True:
        print("Unzipping Yelp-JSON.zip ...")
        run(f'unzip -o "{ZIP_PATH}" -d "{RAW_ZIP_DIR}"')
    else:
        print("Raw zip directory already populated, skipping unzip.")

    # Step 2: find the .tar file (skip macOS metadata files)
    tar_files = [
        p for p in RAW_ZIP_DIR.rglob("*.tar")
        if "__MACOSX" not in str(p) and not p.name.startswith("._")
    ]
    if not tar_files:
        print("ERROR: No .tar file found inside the zip.")
        sys.exit(1)
    tar_path = tar_files[0]
    print(f"Found tar: {tar_path}")

    # Step 3: extract tar into data/raw/
    all_present = all((RAW_DIR / f).exists() for f in EXPECTED_FILES)
    if all_present:
        print("All expected JSON files already present in data/raw/, skipping tar extraction.")
    else:
        print(f"Extracting {tar_path} -> {RAW_DIR} ...")
        run(f'tar -xf "{tar_path}" -C "{RAW_DIR}"')

    # Step 4: verify
    print("\nVerification:")
    all_ok = True
    for fname in EXPECTED_FILES:
        path = RAW_DIR / fname
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  [OK] {fname}  ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    if all_ok:
        print("\nExtraction complete. All files present.")
    else:
        print("\nWARNING: Some files are missing — check the archive structure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
