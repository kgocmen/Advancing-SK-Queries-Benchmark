# =====================
# Common experiment config
# =====================
EXP=ne_K10_L1_R5000_dataset_melbourne100k
SOURCES="dataset"
K="10"
L=1
R=20000
CNT=50
INPUT=./data/melbourne_cleaned_sampled_100k.csv
SCE=non_embedded
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
