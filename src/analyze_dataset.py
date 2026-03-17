"""
analyze_dataset.py — Statistics for processed parquet files.

Usage:
    python src/analyze_dataset.py
    python src/analyze_dataset.py --file merged_sample.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = {0: "None (==0)", 1: "Low (==1)", 2: "Medium (2-3)", 3: "High (>=4)"}


def assign_class(u):
    if u == 0:
        return 0
    elif u == 1:
        return 1
    elif u <= 3:
        return 2
    else:
        return 3


def analyze(path: Path):
    print(f"\n{'='*60}")
    print(f"FILE: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{'='*60}")

    df = pd.read_parquet(path, engine="pyarrow")
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Columns: {df.columns.tolist()}")

    # ------------------------------------------------------------------ #
    # Missing values
    # ------------------------------------------------------------------ #
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None")
    else:
        for col, cnt in missing.items():
            print(f"  {col:<30} {cnt:>10,}  ({cnt/len(df)*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # useful column
    # ------------------------------------------------------------------ #
    if "useful" in df.columns:
        u = df["useful"].dropna()
        print(f"\n--- 'useful' column ({len(u):,} non-null) ---")
        print(f"  min    : {u.min()}")
        print(f"  max    : {u.max()}")
        print(f"  mean   : {u.mean():.4f}")
        print(f"  median : {u.median():.1f}")
        print(f"  std    : {u.std():.4f}")
        print(f"\n  Percentiles:")
        for q, v in u.quantile([0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0]).items():
            print(f"    p{int(q*100):>3}  : {v}")

        print(f"\n  Top-10 value counts:")
        for val, cnt in u.value_counts().sort_index().head(10).items():
            print(f"    useful={val:<6} {cnt:>10,}  ({cnt/len(u)*100:.1f}%)")

        # 4-class distribution
        cls = u.apply(assign_class)
        counts = cls.value_counts().sort_index()
        print(f"\n  4-class distribution:")
        for c, cnt in counts.items():
            print(f"    Class {c} [{CLASS_LABELS[c]:<15}]: {cnt:>10,}  ({cnt/len(u)*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # review_stars
    # ------------------------------------------------------------------ #
    if "review_stars" in df.columns:
        stars = df["review_stars"].dropna()
        print(f"\n--- 'review_stars' ---")
        print(f"  distribution: {dict(stars.value_counts().sort_index())}")
        print(f"  mean: {stars.mean():.3f}  |  std: {stars.std():.3f}")

    # ------------------------------------------------------------------ #
    # Numeric columns summary
    # ------------------------------------------------------------------ #
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip = {"useful", "review_stars"}
    numeric_cols = [c for c in numeric_cols if c not in skip]
    if numeric_cols:
        print(f"\n--- Numeric Columns Summary ---")
        summary = df[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
        summary["null%"] = (df[numeric_cols].isnull().sum() / len(df) * 100).values
        with pd.option_context("display.float_format", "{:.3f}".format, "display.max_rows", 50):
            print(summary.to_string())

    # ------------------------------------------------------------------ #
    # Text length
    # ------------------------------------------------------------------ #
    if "text" in df.columns:
        text = df["text"].fillna("")
        word_counts = text.str.split().str.len()
        print(f"\n--- Review text length (words) ---")
        print(f"  mean   : {word_counts.mean():.1f}")
        print(f"  median : {word_counts.median():.1f}")
        print(f"  min    : {word_counts.min()}")
        print(f"  max    : {word_counts.max()}")
        print(f"  p95    : {word_counts.quantile(0.95):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Specific parquet filename in data/processed/ (default: all)")
    args = parser.parse_args()

    paths = [PROCESSED_DIR / args.file] if args.file else sorted(PROCESSED_DIR.glob("*.parquet"))

    import io, sys
    buf = io.StringIO()
    _stdout = sys.stdout

    class Tee:
        def write(self, x):
            _stdout.write(x)
            buf.write(x)
        def flush(self):
            _stdout.flush()
            buf.flush()

    sys.stdout = Tee()

    for path in paths:
        analyze(path)

    sys.stdout = _stdout

    out_path = RESULTS_DIR / "dataset_statistics.txt"
    with open(out_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nStatistics saved to {out_path}")


if __name__ == "__main__":
    main()
