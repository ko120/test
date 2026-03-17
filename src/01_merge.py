"""
01_merge.py — Memory-efficient Yelp merge and parquet writer.

Joins:
    review  LEFT JOIN business ON business_id
            LEFT JOIN user     ON user_id
            LEFT JOIN checkin  ON business_id  (aggregated total check-ins)

Outputs:
    data/processed/merged_reviews.parquet
    data/processed/merged_sample.parquet
    data/processed/merged_balanced.parquet

4-class label bins (based on review 'useful' count):
    0 — None   : useful == 0
    1 — Low    : useful == 1
    2 — Medium : useful in [2, 3]
    3 — High   : useful >= 4
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42
REVIEW_CHUNKSIZE = 100_000


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def assign_class(u: int) -> int:
    """4-class mapping based on useful count."""
    if u == 0:
        return 0
    elif u == 1:
        return 1
    elif u <= 3:
        return 2
    else:
        return 3


def optimize_object_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """
    Convert low-cardinality object columns to category to reduce memory.
    Do not convert ID columns needed for joins unless desired.
    """
    exclude = exclude or set()
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == "object":
            nunique = df[col].nunique(dropna=False)
            if len(df[col]) > 0 and (nunique / len(df[col])) < 0.5:
                df[col] = df[col].astype("category")
    return df


def reduce_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to int32/float32 to save memory without overflow risk."""
    int_cols = df.select_dtypes(include=["int64"]).columns
    float_cols = df.select_dtypes(include=["float64"]).columns

    for col in int_cols:
        df[col] = df[col].astype("int32")
    for col in float_cols:
        df[col] = df[col].astype("float32")
    return df


def load_json_lines_small(path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Load smaller JSONL files fully into memory.
    Use only for business/user where full load is manageable.
    """
    print(f"Loading {path.name} ...")
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=path.stem, unit=" rows"):
            obj = json.loads(line)
            records.append({c: obj.get(c) for c in columns})
    df = pd.DataFrame.from_records(records)
    df = reduce_numeric_dtypes(df)
    df = optimize_object_columns(df)
    print(f"  -> {len(df):,} rows, {df.shape[1]} cols")
    return df


def load_checkin(path: Path) -> pd.DataFrame:
    """Load checkin JSON and aggregate total check-ins per business."""
    print(f"Loading {path.name} ...")
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="checkin", unit=" rows"):
            obj = json.loads(line)
            business_id = obj.get("business_id")
            date_str = obj.get("date", "")
            count = len(date_str.split(",")) if date_str else 0
            records.append((business_id, count))

    df = pd.DataFrame(records, columns=["business_id", "biz_checkin_count"])
    df["biz_checkin_count"] = pd.to_numeric(df["biz_checkin_count"], downcast="integer")
    print(f"  -> {len(df):,} businesses with check-ins")
    return df


def iter_review_chunks(path: Path, columns: list[str], chunksize: int):
    """Yield review chunks instead of loading the entire review table."""
    print(f"Streaming {path.name} in chunks of {chunksize:,} ...")
    with open(path, "r", encoding="utf-8") as fh:
        buf = []
        for line in tqdm(fh, desc=path.stem, unit=" rows"):
            obj = json.loads(line)
            buf.append({c: obj.get(c) for c in columns})
            if len(buf) >= chunksize:
                df = pd.DataFrame.from_records(buf)
                yield reduce_numeric_dtypes(df)
                buf.clear()
        if buf:
            df = pd.DataFrame.from_records(buf)
            yield reduce_numeric_dtypes(df)


def merge_review_chunk(
    reviews: pd.DataFrame,
    businesses: pd.DataFrame,
    users: pd.DataFrame,
    checkins: pd.DataFrame,
) -> pd.DataFrame:
    """Merge one review chunk with lookup tables."""
    merged = reviews.merge(businesses, on="business_id", how="left", copy=False)
    merged = merged.merge(users, on="user_id", how="left", copy=False)
    merged = merged.merge(checkins, on="business_id", how="left", copy=False)

    # Fill NaN from LEFT JOINs for integer-valued columns so they stay int, not float
    int_fill_cols = [
        "biz_review_count", "is_open",
        "user_review_count", "fans", "user_useful", "user_funny", "user_cool",
        "biz_checkin_count",
    ]
    for col in int_fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype("int64")

    merged["useful_class"] = merged["useful"].map(assign_class).astype("int8")

    merged = reduce_numeric_dtypes(merged)
    merged = optimize_object_columns(
        merged,
        exclude={"review_id", "business_id", "user_id", "text", "date", "yelping_since"},
    )
    return merged


def write_parquet_incrementally(chunks, out_path: Path):
    """Write chunked DataFrames into a single parquet file."""
    writer = None
    schema = None
    total_rows = 0

    try:
        for df in chunks:
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(df)
    finally:
        if writer is not None:
            writer.close()

    print(f"Saved full dataset -> {out_path} ({total_rows:,} rows)")
    if out_path.exists():
        print(f"  Size: {out_path.stat().st_size / 1e6:.1f} MB")


def reservoir_sample_by_class(
    df: pd.DataFrame,
    buffers: dict[int, list[dict]],
    target_per_class: int,
    seen_counts: dict[int, int],
) -> None:
    """
    Reservoir sampling per class.
    Keeps only target_per_class rows for each class in memory.
    """
    records = df.to_dict(orient="records")
    rng = np.random.default_rng(RANDOM_STATE)

    for row in records:
        cls = int(row["useful_class"])
        seen_counts[cls] += 1

        if len(buffers[cls]) < target_per_class:
            buffers[cls].append(row)
        else:
            j = rng.integers(0, seen_counts[cls])
            if j < target_per_class:
                buffers[cls][j] = row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    review_cols = ["review_id", "business_id", "user_id", "stars", "text", "date", "useful"]
    biz_cols = ["business_id", "stars", "review_count", "city", "state", "is_open", "categories"]
    user_cols = [
        "user_id", "review_count", "average_stars", "yelping_since",
        "fans", "useful", "funny", "cool", "elite", "friends",
    ]

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 2. Stream-merge reviews and write full parquet incrementally
    # ---------------------------------------------------------------
    full_path = PROCESSED_DIR / "merged_reviews.parquet"

    if not full_path.exists():
        # 1. Load smaller tables (only needed for merging)
        businesses = load_json_lines_small(RAW_DIR / "yelp_academic_dataset_business.json", biz_cols)
        businesses = businesses.rename(columns={
            "stars": "biz_stars",
            "review_count": "biz_review_count",
        })

        users = load_json_lines_small(RAW_DIR / "yelp_academic_dataset_user.json", user_cols)
        users = users.rename(columns={
            "review_count": "user_review_count",
            "useful": "user_useful",
            "funny": "user_funny",
            "cool": "user_cool",
        })

        checkins = load_checkin(RAW_DIR / "yelp_academic_dataset_checkin.json")
        businesses = businesses.copy()
        users = users.copy()
        checkins = checkins.copy()

    class_counts = np.zeros(4, dtype=np.int64)
    sample_target_per_class = SAMPLE_SIZE // 4
    sample_buffers = {i: [] for i in range(4)}
    sample_seen = {i: 0 for i in range(4)}

    if full_path.exists():
        print(f"Found existing {full_path.name}, skipping merge step.")
        print("Computing class counts and sample from existing parquet ...")
        parquet_file_pre = pq.ParquetFile(full_path)
        for batch in parquet_file_pre.iter_batches(batch_size=REVIEW_CHUNKSIZE):
            df = batch.to_pandas()
            counts = df["useful_class"].value_counts().sort_index()
            for cls, cnt in counts.items():
                class_counts[int(cls)] += int(cnt)
            reservoir_sample_by_class(df, sample_buffers, sample_target_per_class, sample_seen)
            del df
    else:
        def merged_chunks():
            for reviews in iter_review_chunks(RAW_DIR / "yelp_academic_dataset_review.json", review_cols, REVIEW_CHUNKSIZE):
                reviews = reviews.rename(columns={"stars": "review_stars"})
                chunk = merge_review_chunk(reviews, businesses, users, checkins)

                counts = chunk["useful_class"].value_counts().sort_index()
                for cls, cnt in counts.items():
                    class_counts[int(cls)] += int(cnt)

                reservoir_sample_by_class(chunk, sample_buffers, sample_target_per_class, sample_seen)

                yield chunk

                del reviews
                del chunk

        print("Writing full merged parquet incrementally ...")
        write_parquet_incrementally(merged_chunks(), full_path)

    # ---------------------------------------------------------------
    # 3. Print class distribution
    # ---------------------------------------------------------------
    labels = {
        0: "None (==0)",
        1: "Low (==1)",
        2: "Medium (2-3)",
        3: "High (>=4)",
    }

    total_rows = int(class_counts.sum())
    print("\n4-class distribution of 'useful' (full dataset):")
    for cls in range(4):
        cnt = int(class_counts[cls])
        pct = (cnt / total_rows * 100) if total_rows else 0.0
        print(f"  Class {cls} [{labels[cls]}]: {cnt:,} ({pct:.1f}%)")

    # ---------------------------------------------------------------
    # 4. Save equal sample (up to SAMPLE_SIZE, balanced by class)
    # ---------------------------------------------------------------
    sample_rows = []
    for cls in range(4):
        sample_rows.extend(sample_buffers[cls])

    sample = pd.DataFrame(sample_rows)
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    sample_path = PROCESSED_DIR / "merged_sample.parquet"
    print(f"\nSaving equal sample -> {sample_path} ...")
    sample.to_parquet(sample_path, index=False, engine="pyarrow", compression="snappy")
    print(f"  Saved: {sample_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Rows : {len(sample):,}")

    # # ---------------------------------------------------------------
    # # 5. Save fully balanced dataset
    # #    Need min class count, so reload full parquet and sample each class
    # # ---------------------------------------------------------------
    # n_per_class = int(class_counts.min())
    # print(f"\nBalanced dataset target: {n_per_class:,} rows per class ({n_per_class * 4:,} total)")

    # balanced_buffers = {i: [] for i in range(4)}
    # balanced_seen = {i: 0 for i in range(4)}

    # print("Reading merged parquet in batches for balanced sampling ...")
    # parquet_file = pq.ParquetFile(full_path)
    # for batch in parquet_file.iter_batches(batch_size=REVIEW_CHUNKSIZE):
    #     df = batch.to_pandas()
    #     reservoir_sample_by_class(df, balanced_buffers, n_per_class, balanced_seen)
    #     del df

    # balanced_rows = []
    # for cls in range(4):
    #     balanced_rows.extend(balanced_buffers[cls])

    # balanced = pd.DataFrame(balanced_rows)
    # balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # balanced_path = PROCESSED_DIR / "merged_balanced.parquet"
    # print(f"Saving balanced dataset -> {balanced_path} ...")
    # balanced.to_parquet(balanced_path, index=False, engine="pyarrow", compression="snappy")
    # print(f"  Saved: {balanced_path.stat().st_size / 1e6:.1f} MB")

    print("\nDone. Summary:")
    print(f"  Full dataset : {total_rows:,} rows")
    print(f"  Sample       : {len(sample):,} rows")
    print(f"  Balanced     : {len(balanced):,} rows ({n_per_class:,} per class)")


if __name__ == "__main__":
    main()