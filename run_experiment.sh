#!/usr/bin/env bash
# run_experiment.sh — End-to-end pipeline for Yelp upvote prediction
# Usage:
#   ./run_experiment.sh                          # run on 500k sample (default)
#   ./run_experiment.sh --full                   # run on full ~7M dataset
#   ./run_experiment.sh --data_dir /path/to/data # custom data directory
#   ./run_experiment.sh --epochs 30 --batch_size 4096

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ── Parse flags ──────────────────────────────────────────────────────────────
FULL_FLAG=""
DATA_DIR=""
FEATURES_DIR_OVERRIDE=""
EPOCHS=20
BATCH_SIZE=2048
LR=1e-3
WEIGHT_DECAY=1e-4
PATIENCE=5
DROPOUT=0.3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)          FULL_FLAG="--full"; shift ;;
    --data_dir)      DATA_DIR="$2"; shift 2 ;;
    --features_dir)  FEATURES_DIR_OVERRIDE="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --batch_size)    BATCH_SIZE="$2"; shift 2 ;;
    --lr)            LR="$2"; shift 2 ;;
    --weight_decay)  WEIGHT_DECAY="$2"; shift 2 ;;
    --patience)      PATIENCE="$2"; shift 2 ;;
    --dropout)       DROPOUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# --data_dir: symlink the entire data/ folder
if [[ -n "$DATA_DIR" ]]; then
  [[ ! -d "$DATA_DIR" ]] && { echo "ERROR: '$DATA_DIR' does not exist." >&2; exit 1; }
  DATA_DIR="$(cd "$DATA_DIR" && pwd)"
  if [[ -L "$PROJECT_ROOT/data" ]]; then
    rm "$PROJECT_ROOT/data"
  elif [[ -d "$PROJECT_ROOT/data" ]]; then
    fail "data/ is a real directory — rename it before using --data_dir"
  fi
  ln -s "$DATA_DIR" "$PROJECT_ROOT/data"
  echo "Using data directory: $DATA_DIR"
fi

# --features_dir: symlink only data/features/ to an existing features folder
if [[ -n "$FEATURES_DIR_OVERRIDE" ]]; then
  [[ ! -d "$FEATURES_DIR_OVERRIDE" ]] && { echo "ERROR: '$FEATURES_DIR_OVERRIDE' does not exist." >&2; exit 1; }
  FEATURES_DIR_OVERRIDE="$(cd "$FEATURES_DIR_OVERRIDE" && pwd)"
  mkdir -p "$PROJECT_ROOT/data"
  FEAT_LINK="$PROJECT_ROOT/data/features"
  [[ -L "$FEAT_LINK" ]] && rm "$FEAT_LINK"
  ln -s "$FEATURES_DIR_OVERRIDE" "$FEAT_LINK"
  echo "Using features directory: $FEATURES_DIR_OVERRIDE"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; RED='\033[0;31m'; RESET='\033[0m'
step() { echo -e "\n${BOLD}${GREEN}==> $*${RESET}"; }
fail() { echo -e "\n${BOLD}${RED}ERROR: $*${RESET}" >&2; exit 1; }

START=$(date +%s)

# # ── Step 0: Extract ───────────────────────────────────────────────────────────
# step "Step 0/5 — Extracting dataset"
# python src/00_extract.py || fail "Extraction failed"

# # ── Step 1: Merge ─────────────────────────────────────────────────────────────
# step "Step 1/5 — Merging tables into parquet"
# python src/01_merge.py || fail "Merge failed"

# # # ── Step 2: Feature engineering ───────────────────────────────────────────────
# step "Step 2/5 — Engineering features${FULL_FLAG:+ (full dataset)}"
# python src/02_features.py $FULL_FLAG || fail "Feature engineering failed"

# ── Step 3: XGBoost ───────────────────────────────────────────────────────────
step "Step 3/5 — Training XGBoost"
python src/03_xgboost_model.py || fail "XGBoost training failed"

# ── Step 4: MLP ───────────────────────────────────────────────────────────────
step "Step 4/6 — Training MLP (epochs=$EPOCHS, batch=$BATCH_SIZE)"
python src/04_mlp_model.py \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --patience "$PATIENCE" \
  --dropout "$DROPOUT" || fail "MLP training failed"

# ── Step 5: Stacking Ensemble ─────────────────────────────────────────────────
step "Step 5/6 — Training Stacking Ensemble"
python src/05_stacking_model.py || fail "Stacking training failed"

# ── Step 6: Evaluate ──────────────────────────────────────────────────────────
step "Step 6/6 — Evaluating and generating plots"
python src/05_evaluate.py || fail "Evaluation failed"

# ── Done ──────────────────────────────────────────────────────────────────────
END=$(date +%s)
ELAPSED=$(( END - START ))
echo -e "\n${BOLD}${GREEN}Done in $(( ELAPSED/60 ))m $(( ELAPSED%60 ))s.${RESET}"
echo "Results: $PROJECT_ROOT/results/"
echo "Models:  $PROJECT_ROOT/models/"
