import argparse
import json
from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from _mlp_arch import ReviewMLP  # noqa: E402

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
TARGET_NAMES = ["None (==0)", "Low (==1)", "Medium (2-3)", "High (>=4)"]


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    X_tab = np.load(FEATURES_DIR / f"X_tabular_{split}.npy")
    X_tfidf = sparse.load_npz(FEATURES_DIR / f"X_tfidf_{split}.npz").toarray().astype(np.float32)
    y = np.load(FEATURES_DIR / f"y_{split}.npy").astype(np.int64)
    X = np.concatenate([X_tab, X_tfidf], axis=1)
    return X, y


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_true = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        all_logits.append(logits.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())
    logits_all = np.concatenate(all_logits)                          
    y_prob = torch.softmax(torch.from_numpy(logits_all), dim=1).numpy()  
    y_true = np.concatenate(all_true).astype(int)
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, auc, y_prob, y_true


def evaluate_metrics(y_true, y_pred, y_prob, split_name):
    return {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--features_dir", type=str, default=None)
    args = parser.parse_args()

    global FEATURES_DIR
    if args.features_dir:
        FEATURES_DIR = Path(args.features_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    input_dim = X_train.shape[1]

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size * 2, shuffle=False)
    test_loader = make_loader(X_test, y_test, args.batch_size * 2, shuffle=False)

    # init model, optimizer, scheduler, criterion
    model = ReviewMLP(input_dim, dropout=args.dropout).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc_ovr": val_auc})

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # save best model
    model_path = MODELS_DIR / "mlp_model.pt"
    torch.save(best_state, model_path)

    # Reload best state
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # eval
    results = {}
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        _, _, y_prob, y_true = eval_epoch(model, loader, criterion, device)
        y_pred = y_prob.argmax(axis=1)
        m = evaluate_metrics(y_true, y_pred, y_prob, split_name)
        results[split_name] = m

    # Save test predictions
    _, _, y_prob_test, y_true_test = eval_epoch(model, test_loader, criterion, device)
    y_pred_test = y_prob_test.argmax(axis=1)
    np.savez(
        RESULTS_DIR / "mlp_predictions.npz",
        y_true=y_true_test,
        y_pred=y_pred_test,
        y_prob=y_prob_test,
    )

    results["training_history"] = history
    with open(RESULTS_DIR / "mlp_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
