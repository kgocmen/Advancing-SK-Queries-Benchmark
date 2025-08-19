# =====================
# Common experiment config
# =====================
EXP=ankara_test
SOURCES="custom"
K="10"
L=1
R=3000
CNT=50
INPUT=./data/ankara_cleaned.csv
SCE=fused
IDX=hnsw
Q=./data/queries/ankara.csv

# =====================
# Run scripts
# =====================
python3 QueryGenerator.py \
  --exp $EXP \
  --so $SOURCES \
  --k $K \
  --l $L \
  --r $R \
  --cnt $CNT \
  --input $INPUT \
  --queries $Q
python3 Benchmark.py \
  --exp $EXP \
  --so $SOURCES \
  --k $K \
  --sce $SCE \
  --l $L \
  --idx $IDX \
  --input $INPUT
python3 ResultReader.py \
  --exp $EXP \
  --so $SOURCES \
  --k $K \
  --sce $SCE
rm -rf ./data/workloads
