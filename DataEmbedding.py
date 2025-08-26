import numpy as np
from constants import *
from sklearn.preprocessing import normalize
import SpatialEmbedder

model = SEMANTIC

def open_embedding(path: str):
    embedding = np.load(path, mmap_mode='r')
    return embedding

def generate_semantic_embedding_query(query: str):
    embedding = model.encode(query)
    embedding = np.array(embedding, dtype=np.float32)
    return embedding


def generate_concat_embedding_query(query_vec, λ, qlat, qlon):
    spa = SpatialEmbedder.SpatialEmbedder(INPUT_CSV)
    query_spatial = np.array(spa._encode_relative(qlat, qlon), dtype=np.float32)
    query_spatial /= np.linalg.norm(query_spatial)
    query_spatial_lambda = np.tile(query_spatial, λ)
    query_fused = np.concatenate([query_spatial_lambda, query_vec])
    query_fused /= np.linalg.norm(query_fused)
    return query_fused

def generate_concat_embedding(semantic_path: str, spatial_path: str, dim=VECTOR_DIM+4*λ, batch_size=100000):

    tag_embedding = open_embedding(semantic_path)
    spatial_embedding = open_embedding(spatial_path)

    assert tag_embedding.shape[0] == spatial_embedding.shape[0]

    chunks = []
    for i in range(0, tag_embedding.shape[0], batch_size):
        spa_chunk = spatial_embedding[i:i+batch_size]
        sem_chunk = tag_embedding[i:i+batch_size]
        spa_lambda = np.hstack([spa_chunk] * λ)

        fchunk = np.hstack([spa_lambda, sem_chunk])
        fchunk = normalize(fchunk, axis=1)

        chunks.append(fchunk)

    fused = np.vstack(chunks)

    assert fused.shape[1] == dim
    return fused

