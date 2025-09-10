#  QueryGenerator.py
import os
import re
import ast
from typing import List, Tuple, Iterable, Dict
import random
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from Contrastive import ContrastiveModel


from DataEmbedding import (
    generate_semantic_embedding_query,
    generate_concat_embedding_query,
)
from constants import *

# -----------------------
# Helpers
# -----------------------
def safe_parse_json(tag_str):
    if isinstance(tag_str, str):
        try:
            return ast.literal_eval(tag_str)
        except (ValueError, SyntaxError):
            return {}
    return {}

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _as_pgvector(arr) -> str:
    a = np.asarray(arr, dtype=float).reshape(-1)
    vals = ", ".join(f"{float(x):.6f}" for x in a)
    return f"ARRAY[{vals}]::vector"
    #return f"ARRAY{list(arr)}::vector" -> this yields an error


# =======================
# Query Generator
# =======================
class QueryGenerator:
    def __init__(self, input_csv=INPUT_CSV, output_dir: str = "./data/workloads/" + str(EXPERIMENT), k_values: Iterable[int] = K_VALUES, radius: int = RADIUS):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.k_values = list(k_values)
        self.radius = radius
        self.prepositions = ["in","at","on","near","by","around","to","from","with","about","inside","outside","between","among","across","through","over","under"]

        _ensure_dir(self.output_dir)

    def _extract_after_preposition(self, text: str) -> str:
        text = "" if text is None else str(text)
        text_lower = text.lower()
        for prep in self.prepositions:
            pattern = rf"\b{prep}\b (.+)"
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        return text


    def _get_query_path(self):
        base = os.path.basename(self.input_csv)
        prefix = re.split(r'[_.]', base)[0]
        return os.path.join("data", "queries", f"{prefix}.csv")

    
    def _m2deg(self, meters: float) -> float:
        return float(meters) / 111_320.0

    def _sql(self, lon, lat, key_or_pair, k):
        lon_str = f"{float(lon):.8f}"
        lat_str = f"{float(lat):.8f}"
        pt = f"ST_SetSRID(ST_MakePoint({lon_str}, {lat_str}), 4326)"
        deg_radius = self._m2deg(self.radius)

        if isinstance(key_or_pair, (tuple, list)) and len(key_or_pair) == 2:
            key = str(key_or_pair[0]).replace("'", "''")
            val = str(key_or_pair[1]).replace("'", "''")
            rhs = json.dumps({key: val})                # safe JSON: {"key":"value"}
            predicate = f"tags @> '{rhs}'::jsonb"
        else:
            key = str(key_or_pair).replace("'", "''")   # key-existence fallback
            predicate = f"tags ? '{key}'"

        wcl= ""
        if self.radius:
            wcl = f"AND ST_DWithin(geom, {pt}, {deg_radius}) "

        return (
            "SELECT id, tags, "
            f"ST_Distance(geom::geography, {pt}::geography) AS distance "
            "FROM PoIs "
            f"WHERE {predicate} "
            f"{wcl} "
            f"ORDER BY distance "
            f"LIMIT {int(k)};"
        )



    def _c_sql(self, lon, lat, vec, k):
        lon_str = f"{float(lon):.8f}"
        lat_str = f"{float(lat):.8f}"
        pt = f"ST_SetSRID(ST_MakePoint({lon_str}, {lat_str}), 4326)"

        return (
            "SELECT id, tags, "
            f"ST_Distance(geom::geography, {pt}::geography) AS distance, "
            f"1.0 - (embedding <=> {vec}) AS similarity "
            "FROM PoIs "
            f"ORDER BY similarity DESC, distance ASC "
            f"LIMIT {int(k)};"
        )

    def _e_sql(self, lon, lat, vec, k):
        lon_str = f"{float(lon):.8f}"
        lat_str = f"{float(lat):.8f}"
        pt = f"ST_SetSRID(ST_MakePoint({lon_str}, {lat_str}), 4326)"
        deg_radius = self._m2deg(self.radius)

        wcl = ""
        if self.radius:
            wcl = f"WHERE ST_DWithin(geom, {pt}, {deg_radius}) "

        return (
            "SELECT id, tags, "
            f"ST_Distance(geom::geography, {pt}::geography) AS distance, "
            f"1.0 - (embedding <=> {vec}) AS similarity "
            "FROM PoIs "
            f"{wcl} "
            f"ORDER BY similarity DESC, distance ASC "
            f"LIMIT {int(k)};"
        )


    

    # ---------- Embedded from (<keyword>,<lat>,<lon>) ----------
    def produce_from_file(self, path):
        triples = self._read_keyword_lat_lon(path)
        if not triples:
            print(f"No valid rows in {path}. Expected: cafe,-37.8136,144.9631")
            return

        non_by_k: Dict[int, List[str]] = {k: [] for k in self.k_values}
        emb_by_k: Dict[int, List[str]] = {k: [] for k in self.k_values}
        conc_by_k: Dict[int, List[str]] = {k: [] for k in self.k_values}
        cont_by_k: Dict[int, List[str]] = {k: [] for k in self.k_values}   # NEW

        for keyword, lat, lon in triples:
            qvec = generate_semantic_embedding_query(keyword)
            qvec_str = _as_pgvector(qvec)
            cvec = generate_concat_embedding_query(qvec, λ, lat, lon)
            cvec_str = _as_pgvector(cvec)
            zvec = self._contrastive_vec(keyword, lat, lon) # NEW
            z_str = _as_pgvector(zvec) # NEW

            for k in self.k_values:
                non_by_k[k].append(self._sql(lon=lon,lat=lat,key_or_pair=keyword,k=k))
                emb_by_k[k].append(self._e_sql(lon=lon,lat=lat,vec=qvec_str,k=k))
                conc_by_k[k].append(self._c_sql(lon=lon,lat=lat,vec=cvec_str,k=k))
                cont_by_k[k].append(self._c_sql(lon=lon,lat=lat,vec=z_str,k=k)) # NEW

        for k, queries in non_by_k.items():  self._write_sql(f"custom_queries_k{k}.sql", queries)
        for k, queries in emb_by_k.items():  self._write_sql(f"custom_embedded_queries_k{k}.sql", queries)
        for k, queries in conc_by_k.items(): self._write_sql(f"custom_concat_embedded_queries_k{k}.sql", queries)
        for k, queries in cont_by_k.items(): self._write_sql(f"custom_contrastive_embedded_queries_k{k}.sql", queries) # NEW

        print(f"Custom queries written to {self.output_dir}")

    # ---------- Exact keyword + kNN from PoIs CSV ----------
    def generate_from_dataset(self, csv_file: str, point_count: int = POINT_COUNT,
                            search: list = ("amenity","shop","name","cuisine","building","tourism","leisure","highway","public_transport","office")):
        df = pd.read_csv(csv_file)
        df["tags"] = df["tags"].fillna("{}").apply(safe_parse_json)

        non_by_k: Dict[int, List[str]] = {k: [] for k in self.k_values}
        emb_by_k: Dict[int, List[str]] = {k: [] for k in self.k_values}
        conc_by_k: Dict[int, List[str]] = {k: [] for k in self.k_values}
        cont_by_k: Dict[int, List[str]] = {k: [] for k in self.k_values}   # NEW

        idxs = df.index.to_list()
        n = min(point_count, len(df))
        for _ in range(n):
            pos_idx = random.choice(idxs)
            ref_lat = float(df.at[pos_idx, "lat"])
            ref_lon = float(df.at[pos_idx, "lon"])

            # pick (key,value) from a random donor row
            kv = None
            while kv is None:
                donor_tags = df.at[random.choice(idxs), "tags"]
                if isinstance(donor_tags, dict):
                    for s in search:
                        if s in donor_tags and donor_tags[s] is not None:
                            v = donor_tags[s]
                            # accept common scalar types; cast to str for JSON
                            if isinstance(v, (str, int, float, bool)):
                                vs = str(v).strip()
                                if vs:
                                    kv = (s, vs)
                                    break
            if kv is None:
                continue

            # semantic text: prefer "value" (or "key:value" if you like)
            text = kv[1]
            qvec = generate_semantic_embedding_query(text);   qvec_str = _as_pgvector(qvec)
            cvec = generate_concat_embedding_query(qvec, λ, ref_lat, ref_lon); cvec_str = _as_pgvector(cvec)
            zvec = self._contrastive_vec(text, ref_lat, ref_lon); zvec_str = _as_pgvector(zvec) # NEW

            for k in self.k_values:
                non_by_k[k].append(self._sql(lon=ref_lon, lat=ref_lat, key_or_pair=kv, k=k))
                emb_by_k[k].append(self._e_sql(lon=ref_lon, lat=ref_lat, vec=qvec_str, k=k))
                conc_by_k[k].append(self._c_sql(lon=ref_lon, lat=ref_lat, vec=cvec_str, k=k))
                cont_by_k[k].append(self._c_sql(lon=ref_lon, lat=ref_lat, vec=zvec_str, k=k))

        for k, queries in non_by_k.items():  self._write_sql(f"dataset_queries_k{k}.sql", queries)
        for k, queries in emb_by_k.items():  self._write_sql(f"dataset_embedded_queries_k{k}.sql", queries)
        for k, queries in conc_by_k.items(): self._write_sql(f"dataset_concat_embedded_queries_k{k}.sql", queries)
        for k, queries in cont_by_k.items(): self._write_sql(f"dataset_contrastive_embedded_queries_k{k}.sql", queries)

        print(f"Dataset generated queries written to {self.output_dir}")




    # ---------- Internals ----------
    def _read_keyword_lat_lon(self, path: str) -> List[Tuple[str, float, float]]:
        triples: List[Tuple[str, float, float]] = []
        ext = os.path.splitext(path.lower())[1]

        # ---------- Helper: semantic inference for 1-column text ----------
        def infer_coords_for_texts(texts: List[str]) -> List[Tuple[str, float, float]]:
            if not texts:
                return []
            semantic_path = os.path.join(
                "./semantic",
                f"semantic_embeddings_{os.path.basename(self.input_csv).replace('.csv', '.npy')}"
            )
            if not os.path.exists(semantic_path):
                raise FileNotFoundError(f"Semantic embeddings not found: {semantic_path}")

            semantic_matrix = np.load(semantic_path)
            poi_df = pd.read_csv(self.input_csv)
            poi_coords = poi_df[["lat", "lon"]].values

            model = SEMANTIC
            out: List[Tuple[str, float, float]] = []
            for q in texts:
                short_q = self._extract_after_preposition(q)
                q_vec = model.encode(short_q).reshape(1, -1)
                sims = cosine_similarity(q_vec, semantic_matrix)[0]
                idx = int(np.argmax(sims))
                lat, lon = poi_coords[idx]
                out.append((str(q), float(lat), float(lon)))
            return out

        # ---------- CSV path ----------
        if ext == ".csv":
            # Be tolerant: header or no header; 1 or 3 columns
            df = pd.read_csv(path, header=None)
            if df.shape[1] >= 3:
                # 3+ columns: take first three as keyword,lat,lon
                for row in df.itertuples(index=False):
                    try:
                        keyword = str(row[0]).strip()
                        lat = float(row[1]); lon = float(row[2])
                        triples.append((keyword, lat, lon))
                    except Exception:
                        continue
                return triples
            elif df.shape[1] == 1:
                # 1 column text → semantic inference
                texts = [str(x) for x in df.iloc[:, 0].tolist() if str(x).strip()]
                return infer_coords_for_texts(texts)
            else:
                return triples  # empty/invalid

        # ---------- TXT path ----------
        text_only: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split by comma if present; else split on whitespace
                parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
                if len(parts) >= 3:
                    try:
                        keyword = str(parts[0])
                        lat = float(parts[1]); lon = float(parts[2])
                        triples.append((keyword, lat, lon))
                    except Exception:
                        # Could be a text line with commas somewhere else → treat as text
                        text_only.append(line)
                else:
                    text_only.append(line)

        # If we collected any plain text lines, infer their coords and append
        if text_only:
            triples.extend(infer_coords_for_texts(text_only))
        return triples

    def _write_sql(self, filename: str, queries: List[str]):
        _ensure_dir(self.output_dir)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(queries) + ("\n" if queries else ""))
        print(f"Wrote {len(queries)} {filename} queries")

    def _contrastive_encoder_init(self, ckpt: str, text_encoder: str, proj_dim: int,
                                    spatial_encoder: str, freeze_text: bool):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._c_device = device
            self._c_tok = AutoTokenizer.from_pretrained(text_encoder)
            self._c_model = ContrastiveModel(
                text_encoder_name=text_encoder,
                proj_dim=proj_dim,
                spatial_encoder=spatial_encoder,
                spatial_hidden=128,
                freeze_text=freeze_text,
                w_text=CONTRASTIVE.get("w_text", 1.0),        # NEW
                w_spatial=CONTRASTIVE.get("w_spatial", 1.0),  # NEW
            ).to(device)
            state = torch.load(ckpt, map_location=device)
            self._c_model.load_state_dict(state, strict=True)
            self._c_model.eval()

    def _contrastive_vec(self, text: str, lat: float, lon: float):
        enc = self._c_tok(text if text else "", return_tensors="pt",
                        padding="max_length", truncation=True, max_length=64)
        lonlat = torch.tensor([[lon, lat]], dtype=torch.float32)
        with torch.no_grad():
            z = self._c_model._fuse(
                    self._c_model.encode_text(enc["input_ids"].to(self._c_device),
                                            enc["attention_mask"].to(self._c_device)),
                    self._c_model.encode_coords(lonlat.to(self._c_device))
                )
        return z.cpu().numpy().reshape(-1)

def parse_args():
    p = argparse.ArgumentParser(description="Generate SQL workloads for PoI retrieval.")
    p.add_argument("--exp", default=EXPERIMENT)
    p.add_argument("--so", nargs="+", default=SOURCE)
    p.add_argument("--k", nargs="+", type=int, default=K_VALUES)
    p.add_argument("--l", type=int, default=λ)
    p.add_argument("--r", type=int, default=RADIUS)
    p.add_argument("--cnt", type=int, default=POINT_COUNT)
    p.add_argument("--q", type=str, default=None,
                   help="Path to .txt or .csv with rows: <keyword>,<lat>,<lon> for embedded workloads")
    p.add_argument("--input", type=str, default=INPUT_CSV,
                   help="PoIs CSV for exact keyword workloads")
    # NEW: contrastive overrides
    p.add_argument("--c-ckpt", type=str, default=CONTRASTIVE["ckpt"])
    p.add_argument("--c-proj-dim", type=int, default=CONTRASTIVE["proj_dim"])
    p.add_argument("--c-text-encoder", type=str, default=CONTRASTIVE["text_encoder"])
    p.add_argument("--c-spatial-encoder", type=str, default=CONTRASTIVE["spatial_encoder"])
    p.add_argument("--c-freeze", action="store_true", default=CONTRASTIVE["freeze_text"])
    p.add_argument("--c-wtext", type=float, default=CONTRASTIVE["w_text"])
    p.add_argument("--c-wspatial", type=float, default=CONTRASTIVE["w_spatial"])
    return p.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # override globals
    EXPERIMENT = args.exp
    SOURCE = args.so
    K_VALUES = args.k
    λ = args.l
    RADIUS = args.r
    POINT_COUNT = args.cnt
    INPUT_CSV = args.input
    

    print("=== Query Generator Config ===")
    print(f"Input CSV    : {args.input}")
    print(f"K Values     : {args.k}")
    print(f"Lambda (λ)   : {args.l}")
    print(f"Experiment   : {args.exp}")
    print(f"Source       : {args.so}")
    print(f"Radius       : {args.r}")
    print(f"Point Count  : {args.cnt}")
    if args.q:
        print(f"Query CSV/TXT: {args.q}")
    else:
        print(f"Query CSV/TXT: Not specified.")
    print("==============================")

    gen = QueryGenerator(
        input_csv=args.input,
        output_dir="./data/workloads/" + args.exp,
        k_values=args.k,
        radius=args.r
    )
    CONTRASTIVE.update({
        "ckpt": args.c_ckpt,
        "proj_dim": args.c_proj_dim,
        "text_encoder": args.c_text_encoder,
        "spatial_encoder": args.c_spatial_encoder,
        "freeze_text": args.c_freeze,
        "w_text": args.c_wtext,
        "w_spatial": args.c_wspatial
    })

    ckpt_path = CONTRASTIVE["ckpt"]

    # If the default path is still in place, rebuild it from encoder + dim
    if ckpt_path == "./contrastive/contrastive_model.pt":
        ckpt_path = f"./contrastive/model_{CONTRASTIVE['spatial_encoder']}_d{CONTRASTIVE['proj_dim']}.pt"

    gen._contrastive_encoder_init(
        ckpt=ckpt_path,
        text_encoder=CONTRASTIVE["text_encoder"],
        proj_dim=CONTRASTIVE["proj_dim"],
        spatial_encoder=CONTRASTIVE["spatial_encoder"],
        freeze_text=CONTRASTIVE["freeze_text"],
    )


    # embedded and concat queries
    
    if "custom" in SOURCE:
        query_path = gen._get_query_path()
        if args.q:
            gen.produce_from_file(args.q)
        elif os.path.exists(query_path):
            gen.produce_from_file(query_path)
        else:
            print("Nothing created from csv/txt query file!")

    if "dataset" in SOURCE:
        gen.generate_from_dataset(args.input, args.cnt)
    else:
        print("Nothing created from the input file!")
