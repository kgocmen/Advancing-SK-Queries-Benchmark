from sentence_transformers import SentenceTransformer

# Database connection settings
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "postgres",
    "user": "postgres",
    "password": "secret",
}

CONTRASTIVE = {
    "ckpt": "./contrastive/d128.pt",
    "proj_dim": 128,
    "w_text": 1.0,
    "w_spatial": 1.0
}

CONTRASTIVE_DIM = CONTRASTIVE["proj_dim"]


EXPERIMENT = "melbourne_sampled"
SOURCE = ["custom"]
K_VALUES = [10]
λ = 1
RADIUS = 10000
POINT_COUNT = 50
INPUT_CSV = "./data/melbourne_cleaned_sampled_100k.csv"
SCENARIO = "embedded"
INDEX_TYPE = "hnsw" 

SEMANTIC = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL = SentenceTransformer(SEMANTIC)
VECTOR_DIM = 384