#!/usr/bin/env bash
set -euo pipefail

K=10
R_ARR=(2000 5000 10000 20000 50000)
H="hnsw"
IF="ivfflat"

NE="non_embedded"
EX="existing"
E="embedded"
F="fused"
CONC="concat"
CONTR="contrast"

DS="dataset"
CUS="custom"

INPUT="./data/melbourne_cleaned_sampled_100k.csv"
CNT=50

: <<'EXP-1'
for R in "${R_ARR[@]}"; do
  EXP="${NE}_${R}"
  echo "=== 1. Running EXP=${EXP} ==="
  python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --r "$R" --cnt "$CNT"
  python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --sce "$NE"
  python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"
done
EXP-1

: <<'EXP-2'
R=10000
K="1 5 10 20 50"
EXP="${NE}_${R}_multi-k"
echo "=== 2. Running EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --r "$R" --cnt "$CNT"
python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --sce "$NE"
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"
EXP-2

: <<'EXP-3'
K="1 5 10 20 50"
EXP="${NE}_noradius_multi-k"
echo "=== 3. Running EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --r "0" --cnt "$CNT"
python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --sce "$NE"
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"
EXP-3

: <<'EXP-4'
for R in "${R_ARR[@]}"; do
  EXP="${EX}_${R}_k10_ds_${H}"
  echo "=== 4. Running EXP=${EXP} ==="
  python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --r "$R" --cnt "$CNT"
  python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --sce "$EX"
  python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$EX"
done
EXP-4

: <<'EXP-5'
EXP="${EX}_noradius_k10_ds_${H}"
echo "=== 5. Running EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --r "0" --cnt "$CNT"
python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --sce "$EX" --idx $H
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$EX"
EXP-5

: <<'EXP-6'
EXP="${E}_k10_r10000_l1"
echo "=== 6. Queries Generating for EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so $DS $CUS --k $K --input "$INPUT" --cnt "$CNT" --r "10000"
EXPH="${EXP}_H"
EXPIF="${EXP}_IF"
rm -rf ./data/workloads/${EXPH}
rm -rf ./data/workloads/${EXPIF}
cp -r ./data/workloads/${EXP} ./data/workloads/${EXPH}
cp -r ./data/workloads/${EXP} ./data/workloads/${EXPIF}
rm -rf ./data/workloads/${EXP}
echo "=== Running EXP=${EXPH} ==="
python3 Benchmark.py --exp "$EXPH" --so $DS $CUS --k $K --input "$INPUT" --sce "$E" --idx $H --l "1" 
python3 ResultReader.py --exp "$EXPH" --so $DS $CUS --k $K --sce "$E"
echo "=== Running EXP=${EXPIF} ==="
python3 Benchmark.py --exp "$EXPIF" --so $DS $CUS --k $K --input "$INPUT" --sce "$E" --idx $IF --l "1"
python3 ResultReader.py --exp "$EXPIF" --so $DS $CUS --k $K --sce "$E"
EXP-6

: <<'EXP-7'
EXP="${E}_k10_noradius_l1"
echo "=== 7. Queries Generating for EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so $DS $CUS --k $K --input "$INPUT" --cnt "$CNT" --r "0"
EXPH="${EXP}_H"
EXPIF="${EXP}_IF"
rm -rf ./data/workloads/${EXPH}
rm -rf ./data/workloads/${EXPIF}
cp -r ./data/workloads/${EXP} ./data/workloads/${EXPH}
cp -r ./data/workloads/${EXP} ./data/workloads/${EXPIF}
rm -rf ./data/workloads/${EXP}
echo "=== Running EXP=${EXPH} ==="
python3 Benchmark.py --exp "$EXPH" --so $DS $CUS --k $K --input "$INPUT" --sce "$E" --idx $H --l "1"
python3 ResultReader.py --exp "$EXPH" --so $DS $CUS --k $K --sce "$E"
echo "=== Running EXP=${EXPIF} ==="
python3 Benchmark.py --exp "$EXPIF" --so $DS $CUS --k $K --input "$INPUT" --sce "$E" --idx $IF --l "1"
python3 ResultReader.py --exp "$EXPIF" --so $DS $CUS --k $K --sce "$E"
EXP-7


: <<'EXP-8'
LAMBDAS="1 2 3 4"
for L in $LAMBDAS
do
  EXP="${CONC}_k${K}_l${L}"
  echo "=== 8. Queries Generating for EXP=${EXP} ==="
  python3 QueryGenerator.py --exp "$EXP" --so $DS $CUS --k $K --input "$INPUT" --cnt "$CNT" --l "$L"
  EXPH="${EXP}_H"
  EXPIF="${EXP}_IF"
  rm -rf ./data/workloads/${EXPH} ./data/workloads/${EXPIF}
  cp -r ./data/workloads/${EXP} ./data/workloads/${EXPH}
  cp -r ./data/workloads/${EXP} ./data/workloads/${EXPIF}
  rm -rf ./data/workloads/${EXP}
  echo "=== Running EXP=${EXPH} ==="
  python3 Benchmark.py --exp "$EXPH" --so $DS $CUS --k $K --input "$INPUT" --sce "$F" --idx $H --l "$L"
  python3 ResultReader.py --exp "$EXPH" --so $DS $CUS --k $K --sce "$F"
  echo "=== Running EXP=${EXPIF} ==="
  python3 Benchmark.py --exp "$EXPIF" --so $DS $CUS --k $K --input "$INPUT" --sce "$F" --idx $IF --l "$L"
  python3 ResultReader.py --exp "$EXPIF" --so $DS $CUS --k $K --sce "$F"
done
EXP-8



#: <<'EXP-9'
set -euo pipefail

T=./data/melbourne_cleaned_sampled_100k.csv
ENCODERS=("sin")          # or: ("mlp" "sin" "mercator")
DIMS=(128)                     # or: (128 256 384)
WEIGHTS=("1.0 1.0")


#: <<'tr'
# 1) Train once per (encoder, dim)
#    - Contrastive.py should auto-fit spatial center/scales from $T
#      and store them as buffers in the checkpoint.
for ENC in "${ENCODERS[@]}"; do
  for DIM in "${DIMS[@]}"; do
    CKPT="./contrastive/model_${ENC}_d${DIM}.pt"
    echo "=== Training ENC=${ENC}, DIM=${DIM} ==="
    python Contrastive.py train "$T" --text-cols tags name \
      --lon-col lon --lat-col lat \
      --epochs 5 --batch-size 128 \
      --text-encoder sentence-transformers/all-MiniLM-L6-v2 \
      --spatial-encoder "$ENC" --proj-dim "$DIM" --freeze-text \
      --lr 1e-4 --wd 0.01 --checkpoint "$CKPT"
  done
done
#tr

# 2) Run experiments (no extra flags). QueryGenerator/Benchmark just load the ckpt
#    and thereby get the same spatial buffers (anchor/scales) baked into the model.
for ENC in "${ENCODERS[@]}"; do
  for DIM in "${DIMS[@]}"; do
    CKPT="./contrastive/model_${ENC}_d${DIM}.pt"
    for W in "${WEIGHTS[@]}"; do
      WT=$(echo "$W" | cut -d' ' -f1)
      WS=$(echo "$W" | cut -d' ' -f2)
      EXP="${CONTR}_${ENC}_k${K}_d${DIM}_wt${WT}_ws${WS}"

      echo "=== Running EXP=${EXP} ==="
      python QueryGenerator.py --exp "$EXP" --so $DS $CUS --k $K --input "$INPUT" \
        --c-ckpt "$CKPT" \
        --c-proj-dim "$DIM" \
        --c-text-encoder "sentence-transformers/all-MiniLM-L6-v2" \
        --c-spatial-encoder "$ENC" \
        --c-freeze \
        --c-wtext "$WT" \
        --c-wspatial "$WS"

      python3 Benchmark.py --exp "$EXP" --so $DS $CUS --k $K --input "$INPUT" \
        --sce $CONTR --idx $H \
        --c-ckpt "$CKPT" \
        --c-proj-dim "$DIM" \
        --c-text-encoder "sentence-transformers/all-MiniLM-L6-v2" \
        --c-spatial-encoder "$ENC" \
        --c-freeze \
        --c-wtext "$WT" \
        --c-wspatial "$WS"

      python3 ResultReader.py --exp "$EXP" --so $DS $CUS --k $K --sce "contrast"
    done
  done
done
#EXP-9
