"""
05_evaluate.py — Comprehensive evaluation and visualization for final report.

Outputs (saved to results/):
  Tables
    table1_performance.txt / .json     — all-model metrics on test set
    table2_per_class.txt               — per-class P/R/F1 for each model

  Figures
    fig1_learning_curves.png           — MLP train/val loss + val AUC
    fig2_model_comparison.png          — grouped bar chart (all metrics × 3 models)
    fig3_confusion_matrices.png        — normalized confusion matrices (3 models)
    fig4_roc_curves.png                — OvR ROC per class (3 models)
    fig5_pr_curves.png                 — Precision-Recall per class (3 models)
    fig6_feature_importance.png        — XGBoost gain + MLP permutation (top 15)
    fig7_feature_consensus.png         — rank-correlation scatter + consensus ranking
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    roc_curve, precision_recall_curve, auc,
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180})

MODELS      = ["xgboost", "mlp", "stacking"]
MODEL_LABEL = {"xgboost": "XGBoost", "mlp": "MLP", "stacking": "Stacking Ensemble"}
COLORS      = {"xgboost": "#2196F3", "mlp": "#FF5722", "stacking": "#4CAF50"}
CLASS_COLORS = ["#9C27B0", "#FF9800", "#00BCD4", "#F44336"]

TARGET_NAMES = ["None (==0)", "Low (==1)", "Medium (2–3)", "High (≥4)"]
NUM_CLASSES  = 4

METRIC_DISPLAY = {
    "accuracy":           "Accuracy",
    "precision_macro":    "Precision (macro)",
    "recall_macro":       "Recall (macro)",
    "f1_macro":           "F1 (macro)",
    "f1_weighted":        "F1 (weighted)",
    "roc_auc_ovr_macro":  "ROC-AUC (OvR macro)",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_predictions(model_name: str):
    data = np.load(RESULTS_DIR / f"{model_name}_predictions.npz")
    return data["y_true"], data["y_pred"], data["y_prob"]


def load_metrics(model_name: str) -> dict:
    with open(RESULTS_DIR / f"{model_name}_metrics.json") as f:
        return json.load(f)


# ── Table 1: All-model performance ────────────────────────────────────────────

def table1_performance(all_metrics: dict):
    """Text + JSON table of all metrics × all models on test set."""
    col_w = 24
    header = f"{'Metric':<{col_w}}" + "".join(f"{MODEL_LABEL[m]:>{col_w}}" for m in MODELS)
    sep    = "=" * (col_w * (len(MODELS) + 1))
    rows   = ["\nTable 1 — Test-Set Performance Comparison", sep, header, "-" * len(sep)]

    comparison = {}
    for key, label in METRIC_DISPLAY.items():
        vals = {m: all_metrics[m].get("test", {}).get(key, float("nan")) for m in MODELS}
        best = max(vals, key=lambda m: vals[m])
        row  = f"  {label:<{col_w-2}}"
        for m in MODELS:
            marker = " *" if m == best else "  "
            row += f"{vals[m]:>{col_w}.4f}{marker}"[: col_w]
        # rebuild cleanly
        row = f"  {label:<{col_w-2}}"
        for m in MODELS:
            marker = "*" if m == best else " "
            row += f" {vals[m]:>10.4f}{marker:1}"
        rows.append(row)
        comparison[key] = vals

    rows += [sep, "  * = best model for this metric\n"]
    text = "\n".join(rows)
    print(text)

    (RESULTS_DIR / "table1_performance.txt").write_text(text)
    with open(RESULTS_DIR / "table1_performance.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("Saved: table1_performance.txt / .json")


# ── Table 2: Per-class report ─────────────────────────────────────────────────

def table2_per_class(all_preds: dict):
    """Per-class precision / recall / F1 / support for every model."""
    lines = ["\nTable 2 — Per-Class Classification Report (Test Set)"]
    for m in MODELS:
        y_true, y_pred, _ = all_preds[m]
        lines.append(f"\n  {MODEL_LABEL[m]}")
        lines.append("  " + "-" * 60)
        report = classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=4)
        lines.extend("  " + l for l in report.splitlines())

    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "table2_per_class.txt").write_text(text)
    print("Saved: table2_per_class.txt")


# ── Figure 1: MLP learning curves ────────────────────────────────────────────

def fig1_learning_curves():
    with open(RESULTS_DIR / "mlp_metrics.json") as f:
        history = json.load(f).get("training_history", [])
    if not history:
        print("No MLP training history — skipping Fig 1.")
        return

    epochs     = [h["epoch"]       for h in history]
    train_loss = [h["train_loss"]  for h in history]
    val_loss   = [h["val_loss"]    for h in history]
    val_auc    = [h["val_auc_ovr"] for h in history]
    best_epoch = epochs[int(np.argmax(val_auc))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 1 — MLP Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="#4CAF50", lw=2)
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="#FF5722", lw=2)
    ax1.axvline(best_epoch, color="gray", lw=1.2, linestyle="--", label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()

    # AUC
    ax2.plot(epochs, val_auc, color="#2196F3", lw=2, marker="o", markersize=4)
    ax2.axvline(best_epoch, color="gray", lw=1.2, linestyle="--", label=f"Best epoch ({best_epoch})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val ROC-AUC (OvR macro)")
    ax2.set_title("Validation ROC-AUC per Epoch")
    ax2.legend()

    plt.tight_layout()
    path = RESULTS_DIR / "fig1_learning_curves.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved: {path.name}")


# ── Figure 2: Model comparison bar chart ─────────────────────────────────────

def fig2_model_comparison(all_metrics: dict):
    metrics = list(METRIC_DISPLAY.keys())
    labels  = [METRIC_DISPLAY[m] for m in metrics]
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(MODELS):
        vals = [all_metrics[model].get("test", {}).get(k, 0) for k in metrics]
        bars = ax.bar(x + i * width, vals, width, label=MODEL_LABEL[model],
                      color=COLORS[model], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Figure 2 — Model Performance Comparison (Test Set)", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = RESULTS_DIR / "fig2_model_comparison.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved: {path.name}")


# ── Figure 3: Normalized confusion matrices ───────────────────────────────────

def fig3_confusion_matrices(all_preds: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Figure 3 — Normalized Confusion Matrices (Test Set)",
                 fontsize=14, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        y_true, y_pred, _ = all_preds[model]
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                    vmin=0, vmax=1, linewidths=0.5, cbar=ax == axes[-1])
        ax.set_title(MODEL_LABEL[model], fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = RESULTS_DIR / "fig3_confusion_matrices.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved: {path.name}")


# ── Figure 4: OvR ROC curves ─────────────────────────────────────────────────

def fig4_roc_curves(all_preds: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Figure 4 — One-vs-Rest ROC Curves (Test Set)",
                 fontsize=14, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        y_true, _, y_prob = all_preds[model]
        y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

        for cls in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(y_bin[:, cls], y_prob[:, cls])
            cls_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.8, color=CLASS_COLORS[cls],
                    label=f"{TARGET_NAMES[cls]} (AUC={cls_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_title(f"{MODEL_LABEL[model]}\nMacro AUC = {macro_auc:.4f}", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("True Positive Rate")
    plt.tight_layout()
    path = RESULTS_DIR / "fig4_roc_curves.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved: {path.name}")


# ── Figure 5: Precision-Recall curves ────────────────────────────────────────

def fig5_pr_curves(all_preds: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Figure 5 — Precision-Recall Curves (Test Set)",
                 fontsize=14, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        y_true, _, y_prob = all_preds[model]
        y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

        for cls in range(NUM_CLASSES):
            prec, rec, _ = precision_recall_curve(y_bin[:, cls], y_prob[:, cls])
            ap = auc(rec, prec)
            ax.plot(rec, prec, lw=1.8, color=CLASS_COLORS[cls],
                    label=f"{TARGET_NAMES[cls]} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_title(MODEL_LABEL[model], fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    axes[0].set_ylabel("Precision")
    plt.tight_layout()
    path = RESULTS_DIR / "fig5_pr_curves.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved: {path.name}")


# ── MLP permutation importance (helper) ───────────────────────────────────────

def _compute_mlp_permutation_importance(feature_names: list[str]) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from scipy import sparse

    sys.path.insert(0, str(Path(__file__).parent))
    from _mlp_arch import ReviewMLP

    X_tab   = np.load(FEATURES_DIR / "X_tabular_val.npy")
    X_tfidf = sparse.load_npz(FEATURES_DIR / "X_tfidf_val.npz").toarray().astype(np.float32)
    y       = np.load(FEATURES_DIR / "y_val.npy").astype(np.int32)

    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(y), min(5000, len(y)), replace=False)
    X_tab_s, X_tfidf_s, y_s = X_tab[idx], X_tfidf[idx], y[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ReviewMLP(X_tab_s.shape[1] + X_tfidf_s.shape[1]).to(device)
    state  = torch.load(MODELS_DIR / "mlp_model.pt", map_location=device)
    state  = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    def score(X_t):
        X = np.concatenate([X_t, X_tfidf_s], axis=1).astype(np.float32)
        with torch.no_grad():
            p = F.softmax(model(torch.from_numpy(X).to(device)), dim=1).cpu().numpy()
        return roc_auc_score(y_s, p, multi_class="ovr", average="macro")

    baseline = score(X_tab_s)
    n_feat   = X_tab_s.shape[1]
    imps     = np.zeros((n_feat, 5))
    print("  Computing MLP permutation importance (5 repeats × 25 features) ...")
    for j in range(n_feat):
        for r in range(5):
            Xp = X_tab_s.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            imps[j, r] = baseline - score(Xp)

    imp_mean = imps.mean(axis=1)
    with open(RESULTS_DIR / "mlp_permutation_importance.json", "w") as f:
        json.dump({feature_names[i]: float(imp_mean[i]) for i in range(n_feat)}, f, indent=2)
    return imp_mean


# ── Figure 6: Feature importance (XGBoost + MLP) ─────────────────────────────

def fig6_feature_importance(feature_names: list[str], top_n: int = 15):
    # XGBoost gain
    with open(RESULTS_DIR / "xgboost_feature_importance.json") as f:
        xgb_raw = json.load(f)
    xgb_pairs = sorted(
        [(k, v) for k, v in xgb_raw.items() if k in feature_names],
        key=lambda x: x[1], reverse=True
    )[:top_n]
    xgb_names  = [p[0] for p in xgb_pairs]
    xgb_vals   = np.array([p[1] for p in xgb_pairs])
    xgb_vals  /= xgb_vals.max()  # normalize to [0,1] for comparability

    # MLP permutation importance
    perm_path = RESULTS_DIR / "mlp_permutation_importance.json"
    if perm_path.exists():
        with open(perm_path) as f:
            perm_dict = json.load(f)
        mlp_imp = np.array([perm_dict.get(n, 0.0) for n in feature_names])
    else:
        mlp_imp = _compute_mlp_permutation_importance(feature_names)

    sorted_idx = np.argsort(mlp_imp)[::-1][:top_n]
    mlp_names  = [feature_names[i] for i in sorted_idx]
    mlp_vals   = mlp_imp[sorted_idx]
    pos_max    = mlp_vals.max()
    if pos_max > 0:
        mlp_vals = mlp_vals / pos_max

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"Figure 6 — Top-{top_n} Feature Importance",
                 fontsize=14, fontweight="bold")

    def _barh(ax, names, vals, color, xlabel, title):
        y_pos = np.arange(len(names))
        bars  = ax.barh(y_pos, vals[::-1], color=color, alpha=0.85, edgecolor="white", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel(xlabel); ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1.12)
        for bar, v in zip(bars, vals[::-1]):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)

    _barh(ax1, xgb_names, xgb_vals, COLORS["xgboost"],
          "Normalized Gain", f"XGBoost — Top {top_n} (Gain, normalized)")
    _barh(ax2, mlp_names, mlp_vals, COLORS["mlp"],
          "Normalized ROC-AUC Drop", f"MLP — Top {top_n} (Permutation, normalized)")

    plt.tight_layout()
    path = RESULTS_DIR / "fig6_feature_importance.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved: {path.name}")


# ── Figure 7: Feature consensus ───────────────────────────────────────────────

def fig7_feature_consensus(feature_names: list[str], top_n: int = 15):
    """
    Rank-correlation scatter of XGBoost vs MLP importances for tabular features,
    plus a consensus top-N bar chart (average normalized rank).
    """
    with open(RESULTS_DIR / "xgboost_feature_importance.json") as f:
        xgb_raw = json.load(f)
    xgb_imp = np.array([xgb_raw.get(n, 0.0) for n in feature_names])

    perm_path = RESULTS_DIR / "mlp_permutation_importance.json"
    if not perm_path.exists():
        print("  MLP permutation importance missing — run fig6 first or re-run evaluate.")
        return
    with open(perm_path) as f:
        perm_dict = json.load(f)
    mlp_imp = np.array([perm_dict.get(n, 0.0) for n in feature_names])

    # Rank (1 = most important)
    xgb_ranks = len(feature_names) - np.argsort(np.argsort(xgb_imp))
    mlp_ranks = len(feature_names) - np.argsort(np.argsort(mlp_imp))

    # Consensus: average rank → lower = more consensus-important
    avg_rank    = (xgb_ranks + mlp_ranks) / 2
    top_idx     = np.argsort(avg_rank)[:top_n]
    top_names   = [feature_names[i] for i in top_idx]
    top_avg     = avg_rank[top_idx]

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Figure 7 — Feature Importance Consensus (XGBoost vs MLP)",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1], figure=fig)

    # Scatter: rank vs rank
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(xgb_ranks, mlp_ranks, color="#607D8B", alpha=0.7, s=60, edgecolors="white")
    for i, name in enumerate(feature_names):
        if i in top_idx:
            ax1.annotate(name, (xgb_ranks[i], mlp_ranks[i]),
                         fontsize=7, ha="left", va="bottom",
                         xytext=(3, 3), textcoords="offset points")
    ax1.set_xlabel("XGBoost Rank (1 = most important)")
    ax1.set_ylabel("MLP Rank (1 = most important)")
    ax1.set_title("Feature Rank Correlation\n(top-consensus features labeled)")
    # rank correlation annotation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(xgb_ranks, mlp_ranks)
    ax1.text(0.97, 0.04, f"Spearman ρ = {rho:.3f}\np = {pval:.3f}",
             transform=ax1.transAxes, ha="right", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Consensus bar chart
    ax2 = fig.add_subplot(gs[1])
    y_pos = np.arange(top_n)
    colors = [COLORS["xgboost"] if xgb_ranks[i] < mlp_ranks[i] else COLORS["mlp"]
              for i in top_idx]
    bars = ax2.barh(y_pos, top_avg[::-1], color=colors[::-1], alpha=0.85,
                    edgecolor="white", height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_names[::-1], fontsize=9)
    ax2.set_xlabel("Average Rank (lower = more important)")
    ax2.set_title(f"Consensus Top-{top_n} Features\n(blue = XGB ranks higher, orange = MLP ranks higher)")
    for bar, v in zip(bars, top_avg[::-1]):
        ax2.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    path = RESULTS_DIR / "fig7_feature_consensus.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global FEATURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default=None)
    args = parser.parse_args()
    if args.features_dir:
        FEATURES_DIR = Path(args.features_dir)

    all_preds   = {m: load_predictions(m) for m in MODELS}
    all_metrics = {m: load_metrics(m)     for m in MODELS}

    with open(FEATURES_DIR / "feature_names.json") as f:
        feature_names = json.load(f)["tabular_features"]

    print("\n[Table 1] Performance comparison ...")
    table1_performance(all_metrics)

    print("\n[Table 2] Per-class report ...")
    table2_per_class(all_preds)

    print("\n[Fig 1] MLP learning curves ...")
    fig1_learning_curves()

    print("\n[Fig 2] Model comparison bar chart ...")
    fig2_model_comparison(all_metrics)

    print("\n[Fig 3] Confusion matrices ...")
    fig3_confusion_matrices(all_preds)

    print("\n[Fig 4] ROC curves ...")
    fig4_roc_curves(all_preds)

    print("\n[Fig 5] Precision-Recall curves ...")
    fig5_pr_curves(all_preds)

    print("\n[Fig 6] Feature importance ...")
    fig6_feature_importance(feature_names)

    print("\n[Fig 7] Feature consensus ...")
    fig7_feature_consensus(feature_names)

    print("\nDone. All outputs in results/")


if __name__ == "__main__":
    main()
