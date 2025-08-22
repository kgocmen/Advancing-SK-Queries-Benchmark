#!/usr/bin/env bash
set -euo pipefail

K=10
R_ARR=(2000 5000 10000 20000 50000)
SCE=non_embedded
DS=("dataset")                             # --so accepts multiple; keep as array
INPUT="./data/melbourne_cleaned.csv"
CNT=50


: <<'EXP-1'
for R in "${R_ARR[@]}"; do
  EXP="${SCE}_${R}"
  echo "=== Running EXP=${EXP} ==="
  python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --r "$R" --cnt "$CNT"
  python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --sce "$SCE"
  python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$SCE"
done
EXP-1

#: <<'EXP-2'
R=10000
K="1 5 10 20 50"
EXP="bigdata_${SCE}_${R}_multi-k"
echo "=== Running EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --r "$R" --cnt "$CNT"
python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --sce "$SCE"
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$SCE"
#EXP-2