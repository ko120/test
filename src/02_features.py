import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5


def build_tabular_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Derive ~25 tabular features from the merged dataframe."""
    feat = pd.DataFrame(index=df.index)

    # Review-level text features
    feat["review_stars"] = df["review_stars"].fillna(0).astype(float)

    text = df["text"].fillna("")
    feat["review_text_len"] = text.str.len()
    feat["review_word_count"] = text.str.split().str.len()
    feat["review_exclamation_count"] = text.str.count("!")
    feat["review_question_count"] = text.str.count(r"\?")
    feat["review_uppercase_ratio"] = text.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )

    # Date features
    dates = pd.to_datetime(df["date"], errors="coerce")
    feat["review_date_year"] = dates.dt.year.fillna(2015).astype(float)
    feat["review_date_month"] = dates.dt.month.fillna(6).astype(float)
    feat["review_date_dayofweek"] = dates.dt.dayofweek.fillna(2).astype(float)

    # Business features
    feat["biz_stars"] = df["biz_stars"].fillna(df["biz_stars"].median()).astype(float)
    feat["biz_review_count"] = np.log1p(df["biz_review_count"].fillna(0).astype(float))
    feat["biz_is_open"] = df["is_open"].fillna(1).astype(float)
    feat["biz_checkin_count"] = np.log1p(df["biz_checkin_count"].fillna(0).astype(float))
    categories = df["categories"].fillna("")
    feat["biz_category_count"] = categories.str.split(",").str.len().fillna(0).astype(float)

    # User features
    feat["user_review_count"] = np.log1p(df["user_review_count"].fillna(0).astype(float))
    feat["user_avg_stars"] = df["average_stars"].fillna(df["average_stars"].median()).astype(float)

    yelping_since = pd.to_datetime(df["yelping_since"], errors="coerce")
    reference_date = pd.Timestamp("2022-01-01")
    feat["user_yelping_years"] = (
        (reference_date - yelping_since).dt.days / 365.25
    ).fillna(0).clip(lower=0).astype(float)

    feat["user_fans"] = np.log1p(df["fans"].fillna(0).astype(float))
    feat["user_useful"] = np.log1p(df["user_useful"].fillna(0).astype(float))
    feat["user_funny"] = np.log1p(df["user_funny"].fillna(0).astype(float))
    feat["user_cool"] = np.log1p(df["user_cool"].fillna(0).astype(float))

    elite = df["elite"].fillna("")
    feat["user_elite_count"] = elite.apply(
        lambda e: len([x for x in str(e).split(",") if x.strip()]) if e else 0
    ).astype(float)

    friends = df["friends"].fillna("")
    feat["user_friends_count"] = np.log1p(
        friends.apply(lambda f: len([x for x in str(f).split(",") if x.strip()]) if f else 0).astype(float)
    )

    feature_names = feat.columns.tolist()
    X = feat.values.astype(np.float32)
    return X, feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use full dataset instead of sample")
    args = parser.parse_args()

    if args.full:
        parquet_file = "merged_reviews.parquet"
    else:
        parquet_file = "merged_sample.parquet"
    parquet_path = PROCESSED_DIR / parquet_file

    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # clas
    def assign_class(u):
        if u == 0:
            return 0
        elif u == 1:
            return 1
        elif u <= 3:
            return 2
        else:
            return 3

    y = df["useful"].apply(assign_class).astype(np.int32).values

    idx = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.15 / 0.85, stratify=y_trainval, random_state=RANDOM_STATE
    )

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}

    X_tab, feature_names = build_tabular_features(df)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_tab[idx_train] = scaler.fit_transform(X_tab[idx_train])
    X_tab[idx_val] = scaler.transform(X_tab[idx_val])
    X_tab[idx_test] = scaler.transform(X_tab[idx_test])

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    for split_name, idx_split in splits.items():
        path = FEATURES_DIR / f"X_tabular_{split_name}.npy"
        np.save(path, X_tab[idx_split])
        np.save(FEATURES_DIR / f"y_{split_name}.npy", y[idx_split])

    texts = df["text"].fillna("").values
    texts_train = texts[idx_train]

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,
    )
    X_tfidf_train = vectorizer.fit_transform(texts_train)

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    sparse.save_npz(FEATURES_DIR / "X_tfidf_train.npz", X_tfidf_train.astype(np.float32))

    for split_name in ["val", "test"]:
        idx_split = splits[split_name]
        X_tfidf = vectorizer.transform(texts[idx_split])
        path = FEATURES_DIR / f"X_tfidf_{split_name}.npz"
        sparse.save_npz(path, X_tfidf.astype(np.float32))

    feature_info = {
        "tabular_features": feature_names,
        "n_tabular": len(feature_names),
        "n_tfidf": TFIDF_MAX_FEATURES,
        "tfidf_vocab_size": len(vectorizer.vocabulary_),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "class_dist": {k: np.bincount(y[v], minlength=4).tolist() for k, v in splits.items()},
    }
    with open(FEATURES_DIR / "feature_names.json", "w") as f:
        json.dump(feature_info, f, indent=2)
  


if __name__ == "__main__":
    main()
