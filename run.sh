# =====================
# Common experiment config
# =====================
EXP=all_K10_L1_R5000_custom_melbourne100k
SOURCES="custom dataset"
K="10"
L=1
R=5000
CNT=50
INPUT=./data/melbourne_cleaned_sampled_100k.csv
SCE=embedded
IDX=hnsw

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
  --input $INPUT
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
