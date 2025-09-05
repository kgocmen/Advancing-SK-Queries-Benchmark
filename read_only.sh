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

# === 1. non_embedded across radii ===
for R in "${R_ARR[@]}"; do
  EXP="${NE}_${R}"
  echo "=== ResultReader EXP=${EXP} ==="
  python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"
done

# === 2. non_embedded multi-k (R=10000) ===
R=10000
K="1 5 10 20 50"
EXP="${NE}_${R}_multi-k"
echo "=== ResultReader EXP=${EXP} ==="
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"

# === 3. non_embedded multi-k (no radius) ===
R=10000
K="1 5 10 20 50"
EXP="${NE}_noradius_multi-k"
echo "=== ResultReader EXP=${EXP} ==="
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$NE"

# === 4. existing across radii (k10, dataset, idx=$H) ===
K=10
for R in "${R_ARR[@]}"; do
  EXP="${EX}_${R}_k10_ds_${H}"
  echo "=== ResultReader EXP=${EXP} ==="
  python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$EX"
done

# === 5. existing no radius (k10, dataset, idx=$H) ===
EXP="${EX}_noradius_k10_ds_${H}"
echo "=== ResultReader EXP=${EXP} ==="
python3 ResultReader.py --exp "$EXP" --so "$DS" --k $K --sce "$EX"

# === 6. embedded r=10000, λ=1 (two index variants: H, IF) ===
EXP="${E}_k10_r10000_l1"
EXPH="${EXP}_H"
EXPIF="${EXP}_IF"
echo "=== ResultReader EXP=${EXPH} ==="
python3 ResultReader.py --exp "$EXPH" --so $DS $CUS --k $K --sce "$E"
echo "=== ResultReader EXP=${EXPIF} ==="
python3 ResultReader.py --exp "$EXPIF" --so $DS $CUS --k $K --sce "$E"

# === 7. embedded no radius, λ=1 (two index variants: H, IF) ===
EXP="${E}_k10_noradius_l1"
EXPH="${EXP}_H"
EXPIF="${EXP}_IF"
echo "=== ResultReader EXP=${EXPH} ==="
python3 ResultReader.py --exp "$EXPH" --so $DS $CUS --k $K --sce "$E"
echo "=== ResultReader EXP=${EXPIF} ==="
python3 ResultReader.py --exp "$EXPIF" --so $DS $CUS --k $K --sce "$E"
