# Database connection settings
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "postgres",
    "user": "postgres",
    "password": "secret",
}

VECTOR_DIM = 384

EXPERIMENT = "melbourne_sampled"
SOURCE = ["custom"]
K_VALUES = [10]
λ = 1
RADIUS = 10000
POINT_COUNT = 50
INPUT_CSV = "./data/melbourne_cleaned_sampled_100k.csv"
SCENARIO = "embedded"
INDEX_TYPE = "hnsw" 
