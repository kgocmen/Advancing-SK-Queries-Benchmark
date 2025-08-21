#!/usr/bin/env bash
set -euo pipefail

K=10
R_ARR=(2000 5000 10000 20000 50000)
SCE=non_embedded
DS=("dataset")                             # --so accepts multiple; keep as array
INPUT="./data/melbourne_cleaned.csv"
CNT=50

for R in "${R_ARR[@]}"; do
  EXP="bigdata_${SCE}_${R}"
  echo "=== Running EXP=${EXP} ==="
  
  # 1) Generate SQL workloads for this combo
  python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k "$K" --input "$INPUT" --r "$R" --cnt "$CNT"

  # 2) Run benchmarks
  python3 Benchmark.py --exp "$EXP" --so "$DS" --k "$K" --input "$INPUT" --sce "$SCE"

  # 3) Read and print results
  python3 ResultReader.py --exp "$EXP" --so "$DS" --k "$K" --sce "$SCE"
done 