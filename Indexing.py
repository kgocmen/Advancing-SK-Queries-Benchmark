from constants import *

def create_spatial_index():
    return ["CREATE INDEX IF NOT EXISTS idx_pois_geom ON PoIs USING GIST (geom);"]
def create_keyword_index():
    return ["CREATE INDEX IF NOT EXISTS idx_pois_tags ON PoIs USING GIN (tags);"]

def create_pgvector_index():
    if INDEX_TYPE == "ivfflat":
        q = ("CREATE INDEX IF NOT EXISTS idx_pois_embedding "
             f"ON PoIs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 300);")
    else:
        q = ("CREATE INDEX IF NOT EXISTS idx_pois_embedding "
             "ON PoIs USING hnsw (embedding vector_cosine_ops) "
             "WITH (m = 16, ef_construction = 64);")
    return [q]
def create_concat_pgvector_index():
    return create_pgvector_index()
def create_spatial_keyword_index():
    return create_spatial_index() + create_keyword_index()
def create_embedding_spatial_index():
    return create_spatial_index() + create_pgvector_index()