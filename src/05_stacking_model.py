import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _mlp_arch import ReviewMLP  # shared MLP architecture

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4


# data loading
def load_dense(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load tabular + TF-IDF features as a dense array (for MLP)."""
    X_tab   = np.load(FEATURES_DIR / f"X_tabular_{split}.npy")
    X_tfidf = sparse.load_npz(FEATURES_DIR / f"X_tfidf_{split}.npz").toarray().astype(np.float32)
    y       = np.load(FEATURES_DIR / f"y_{split}.npy").astype(np.int64)
    return np.concatenate([X_tab, X_tfidf], axis=1), y


def load_sparse(split: str) -> tuple[object, np.ndarray]:
    """Load tabular + TF-IDF features as a sparse matrix (for XGBoost)."""
    X_tab   = np.load(FEATURES_DIR / f"X_tabular_{split}.npy")
    X_tfidf = sparse.load_npz(FEATURES_DIR / f"X_tfidf_{split}.npz")
    y       = np.load(FEATURES_DIR / f"y_{split}.npy")
    return sparse.hstack([X_tab, X_tfidf], format="csr"), y


def xgb_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)


@torch.no_grad()
def mlp_proba(model, X: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    logits = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i : i + batch_size]).to(device)
        logits.append(model(batch).cpu().numpy())
    logits_all = np.concatenate(logits)
    return torch.softmax(torch.from_numpy(logits_all), dim=1).numpy()


# metrics

def compute_metrics(y_true, y_pred, y_prob, split_name: str) -> dict:
    return {
        "split":              split_name,
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "precision_macro":    float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "recall_macro":       float(recall_score   (y_true, y_pred, average="macro",    zero_division=0)),
        "f1_macro":           float(f1_score        (y_true, y_pred, average="macro",    zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted":    float(recall_score   (y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted":        float(f1_score        (y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro":  float(roc_auc_score  (y_true, y_prob, multi_class="ovr",  average="macro")),
    }


# main loop

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base models
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model.pkl")

    X_val_dense, _ = load_dense("val")
    input_dim = X_val_dense.shape[1]
    mlp_model = ReviewMLP(input_dim).to(device)
    mlp_model.load_state_dict(
        {k: v.to(device) for k, v in torch.load(MODELS_DIR / "mlp_model.pt", map_location=device).items()}
    )

    # Build meta-features from val set (used to train meta-learner)
    X_val_sparse, y_val = load_sparse("val")
    meta_val = np.hstack([
        xgb_proba(xgb_model, X_val_sparse),
        mlp_proba(mlp_model, X_val_dense, device),
    ])  # shape: (n_val, 8)

    # Train meta-learner on val predictions
    meta_learner = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial", n_jobs=-1)
    meta_learner.fit(meta_val, y_val)
    joblib.dump(meta_learner, MODELS_DIR / "stacking_meta_learner.pkl")

    # Evaluate on val and test
    results = {}
    for split in ["val", "test"]:
        X_dense, y_true = load_dense(split)
        X_sparse, _     = load_sparse(split)
        meta = np.hstack([xgb_proba(xgb_model, X_sparse), mlp_proba(mlp_model, X_dense, device)])
        y_prob = meta_learner.predict_proba(meta)
        y_pred = y_prob.argmax(axis=1)
        m = compute_metrics(y_true, y_pred, y_prob, split)
        results[split] = m
        print(f"\n--- {split.upper()} ---")
        print(f"  Accuracy      : {m['accuracy']:.4f}")
        print(f"  ROC-AUC (ovr) : {m['roc_auc_ovr_macro']:.4f}")
        print(f"  F1 (macro)    : {m['f1_macro']:.4f}")
        print(f"  F1 (weighted) : {m['f1_weighted']:.4f}")

    # Save test predictions
    X_test_dense,  y_test  = load_dense("test")
    X_test_sparse, _       = load_sparse("test")
    meta_test  = np.hstack([xgb_proba(xgb_model, X_test_sparse), mlp_proba(mlp_model, X_test_dense, device)])
    y_prob_test = meta_learner.predict_proba(meta_test)
    np.savez(
        RESULTS_DIR / "stacking_predictions.npz",
        y_true=y_test,
        y_pred=y_prob_test.argmax(axis=1),
        y_prob=y_prob_test,
    )

    with open(RESULTS_DIR / "stacking_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
