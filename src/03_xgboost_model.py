
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split: str):
    X_tab = np.load(FEATURES_DIR / f"X_tabular_{split}.npy")
    X_tfidf = sparse.load_npz(FEATURES_DIR / f"X_tfidf_{split}.npz")
    X = sparse.hstack([X_tab, X_tfidf], format="csr")
    y = np.load(FEATURES_DIR / f"y_{split}.npy")
    return X, y


TARGET_NAMES = ["None (==0)", "Low (==1)", "Medium (2-3)", "High (>=4)"]


def evaluate(y_true, y_pred, y_prob, split_name="test"):
    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")),
    }
    return metrics


def main():
    global FEATURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default=None)
    args = parser.parse_args()
    if args.features_dir:
        FEATURES_DIR = Path(args.features_dir)

    # load dataset
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        n_jobs=-1,
        random_state=42,
        verbosity=1,
    )
    # train 
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    model_path = MODELS_DIR / "xgboost_model.pkl"
    joblib.dump(model, model_path)
    # eval
    results = {}
    for split_name, X, y_true in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_prob = model.predict_proba(X)
        y_pred = model.predict(X)
        m = evaluate(y_true, y_pred, y_prob, split_name)
        results[split_name] = m

    # Save test predictions
    y_prob_test = model.predict_proba(X_test)   # (n, 4)
    y_pred_test = model.predict(X_test)
    np.savez(
        RESULTS_DIR / "xgboost_predictions.npz",
        y_true=y_test,
        y_pred=y_pred_test,
        y_prob=y_prob_test,
    )

    # Save metrics
    with open(RESULTS_DIR / "xgboost_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(FEATURES_DIR / "feature_names.json") as f:
        feature_info = json.load(f)
    feature_names = feature_info["tabular_features"]

    importances = model.feature_importances_
    imp_dict = dict(zip(feature_names, importances.tolist()))
    with open(RESULTS_DIR / "xgboost_feature_importance.json", "w") as f:
        json.dump(imp_dict, f, indent=2)


if __name__ == "__main__":
    main()
