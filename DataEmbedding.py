import numpy as np
from constants import *
from sklearn.preprocessing import normalize
import SpatialEmbedder
import os
from numpy.lib.format import open_memmap
import pandas as pd, numpy as np, ast, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from Contrastive import ContrastiveModel

model = SEMANTIC_MODEL

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

def generate_contrastive_embedding(
    input_csv: str,
    ckpt: str,
    proj_dim: int,
    freeze_text: bool = True,
    w_text: float = 1.0,            # <-- NEW
    w_spatial: float = 1.0,         # <-- NEW
    batch_size: int = 4096,
    workers: int = 4,
):
    # --- load only what we need
    df = pd.read_csv(input_csv, low_memory=False)
    for col in ("lat", "lon", "tags"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    def _parse_tags(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str) and x.strip().startswith("{"):
            try:
                return ast.literal_eval(x)
            except Exception:
                return {}
        return {}

    tags_dicts = df["tags"].fillna("{}").apply(_parse_tags)

    def _tags_to_text(d):
        if not d:
            return "poi"
        parts = []
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool)):
                vs = str(v).strip()
                if vs:
                    parts.append(f"{k}: {vs}")
        return "; ".join(parts) if parts else "poi"

    texts = tags_dicts.apply(_tags_to_text).tolist()
    coords = df[["lon", "lat"]].values.astype("float32")  # (lon, lat)

    # --- caching path: include ckpt tag + weights to avoid collisions
    ck_tag = os.path.splitext(os.path.basename(ckpt))[0]
    out_path = f"./contrastive/{EXPERIMENT}_{ck_tag}_wt{w_text}_ws{w_spatial}.npy"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # reuse cache if shape matches
    if os.path.exists(out_path):
        try:
            mm = np.load(out_path, mmap_mode="r")
            if mm.shape == (len(texts), proj_dim):
                print(f"Contrastive embeddings already exist: {out_path}  {mm.shape}")
                return np.asarray(mm)
            else:
                print(f"Existing file shape {mm.shape} != expected {(len(texts), proj_dim)}; rebuilding.")
        except Exception as e:
            print(f"Failed to load existing {out_path} ({e}); rebuilding.")

    # --- model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveModel(
        proj_dim=proj_dim,
        spatial_hidden=128,
        freeze_text=freeze_text,
        w_text=w_text,
        w_spatial=w_spatial,          
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    tok = AutoTokenizer.from_pretrained(SEMANTIC)

    fused_out = open_memmap(out_path, mode="w+", dtype="float32", shape=(len(texts), proj_dim))
    write_ptr = 0

    torch.set_grad_enabled(False)
    use_cuda = (device.type == "cuda")
    cur_bs = int(batch_size)
    max_len = 64

    start = 0
    while start < len(texts):
        end = min(start + cur_bs, len(texts))
        try:
            enc = tok(texts[start:end], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_len)
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(device, non_blocking=True)
            lonlat = torch.from_numpy(coords[start:end]).to(device, non_blocking=True)

            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    zt = model.encode_text(input_ids, attention_mask)
                    zl = model.encode_coords(lonlat)
                    z = model._fuse(zt, zl).float().cpu().numpy()
            else:
                zt = model.encode_text(input_ids, attention_mask)
                zl = model.encode_coords(lonlat)
                z = model._fuse(zt, zl).cpu().numpy()

            fused_out[write_ptr:write_ptr + (end - start)] = z
            write_ptr += (end - start)
            start = end

            if cur_bs < batch_size:
                cur_bs = min(batch_size, cur_bs * 2)

        except torch.cuda.OutOfMemoryError:
            if use_cuda:
                torch.cuda.empty_cache()
            if cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                print(f"[contrastive] CUDA OOM → reducing batch_size to {cur_bs} and retrying…")
                continue
            else:
                if use_cuda:
                    print("[contrastive] Still OOM at batch_size=1 → falling back to CPU.")
                    device = torch.device("cpu")
                    use_cuda = False
                    model = model.to(device)
                    continue
                else:
                    raise

    fused_out.flush()
    print(f"Created contrastive embeddings: {out_path}  {(len(texts), proj_dim)}")
    return np.asarray(fused_out)

