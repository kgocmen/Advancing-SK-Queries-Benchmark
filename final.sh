#!/usr/bin/env bash
set -euo pipefail

# ---- Global knobs ----
PY=python3
DIM=384
K="1 10"                 # argparse should accept nargs=+
R=20000                  # --r
LAMBDA=2                 # --l
WT=1.0                   # contrastive weight for text
WS=0.05                   # contrastive weight for spatial
SCENARIO="contrast"      # --sce
INDEX="hnsw"             # --idx
SOURCE="custom"          # --so

# ---------- helper ----------

train_city () {
  local CITY="$1"
  local INPUT="$2"
  local CKPT="./contrastive/${CITY}_d${DIM}.pt"

  echo "=== Training contrastive (${CITY}) DIM=${DIM} ==="
  "$PY" Contrastive.py train "$INPUT" \
    --epochs 7 --batch-size 256 \
    --proj-dim "$DIM" --lr 1e-4 --wd 0.01 \
    --checkpoint "$CKPT" >> "${CITY}_log.txt"
}

run_city () {
  local CITY="$1"
  local INPUT="$2"
  local CKPT="./contrastive/${CITY}_d${DIM}.pt"
  local EXP="${CITY}_final_contrastive"

  echo "=== Generating queries EXP=${EXP} (${CITY}) ==="
  "$PY" QueryGenerator.py --exp "$EXP" --so "$SOURCE" --k $K --input "$INPUT" --l "$LAMBDA" --r "$R"  \
    --c-ckpt "$CKPT" --c-proj-dim "$DIM" --c-wtext "$WT" --c-wspatial "$WS" --q "data/queries/${CITY}.csv" >> "${CITY}"_log2.txt

  echo "=== Benchmarking EXP=${EXP} ${CITY} ==="
  "$PY" Benchmark.py --exp "$EXP" --so "$SOURCE" --k $K --input "$INPUT"  --l "$LAMBDA" --sce "$SCENARIO" --idx "$INDEX" \
    --c-ckpt "$CKPT" --c-proj-dim "$DIM" --c-wtext "$WT" --c-wspatial "$WS" >> "${CITY}_log2.txt"

  echo "=== Reading results EXP=${EXP} (${CITY}) ==="
  "$PY" ResultReader.py --exp "$EXP" --so "$SOURCE" --k $K --sce "$SCENARIO" >> "${CITY}"_log2.txt
}

train_city "melbourne" "./data/melbourne_cleaned_sampled_100k.csv"
run_city "melbourne" "./data/melbourne_cleaned_sampled_100k.csv"
train_city "istanbul" "./data/istanbul_cleaned.csv"
run_city "istanbul" "./data/istanbul_cleaned.csv"
train_city "cukurova" "./data/cukurova_cleaned.csv"
run_city "cukurova" "./data/cukurova_cleaned.csv"

echo "All done."
