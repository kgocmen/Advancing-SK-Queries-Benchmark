#!/usr/bin/env bash
set -euo pipefail

K=10
R_ARR=(2000 5000 10000 20000 50000)
H="hnsw"
IF="ivfflat"
NE="non_embedded"
EX="existing"
E="embedded"
DS="dataset"
CUS="custom"
INPUT="./data/melbourne_cleaned_sampled_100k.csv"
CNT=50



: <<'EXP-1'
for R in "${R_ARR[@]}"; do
  EXP="${NE}_${R}"
  echo "=== Running EXP=${EXP} ==="
  python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --r "$R" --cnt "$CNT"
  python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --sce "$NE"
  python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"
done
EXP-1

: <<'EXP-2'
R=10000
K="1 5 10 20 50"
EXP="${NE}_${R}_multi-k"
echo "=== Running EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --r "$R" --cnt "$CNT"
python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --sce "$NE"
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"
EXP-2

: <<'EXP-3'
R=10000
K="1 5 10 20 50"
EXP="${NE}_noradius_multi-k"
echo "=== Running EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --r "0" --cnt "$CNT"
python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --sce "$NE"
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"
EXP-3

: <<'EXP-4'
for R in "${R_ARR[@]}"; do
  EXP="${EX}_${R}_k10_ds_${H}"
  echo "=== Running EXP=${EXP} ==="
  python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --r "$R" --cnt "$CNT"
  python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input $INPUT --sce "$EX"
  python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$EX"
done
EXP-4

#: <<'EXP-5'
EXP="${EX}_noradius_k10_ds_${H}"
echo "=== Running EXP=${EXP} ==="
python3 QueryGenerator.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --r "0" --cnt "$CNT"
python3 Benchmark.py --exp "$EXP" --so "$DS" --k $K --input "$INPUT" --sce "$EX" --idx $H
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$EX"
#EXP-5