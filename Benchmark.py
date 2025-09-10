from utilities import *
import pandas as pd
import json
from constants import *
from Indexing import *
from DataEmbedding import *
import SpatialEmbedder
import SemanticEmbedder
import os
import ast
from typing import List, Dict, Tuple
import argparse
import os

SCENARIOS = {
    "concat":{
        "concat_embedding": create_concat_pgvector_index
    },
    "contrast":{
        "contrastive_embedding": create_contrastive_pgvector_index
    },
    "fused": {
        "concat_embedding": create_concat_pgvector_index,
        "contrastive_embedding": create_contrastive_pgvector_index
    },
    "all": {
        "no_index": lambda: None,
        "spatial_index": create_spatial_index,
        "keyword_index": create_keyword_index,
        "spatial_keyword_index": create_spatial_keyword_index,
        "pgvector_embedding": create_pgvector_index,
        "spatial_and_pgvector_embedding": create_embedding_spatial_index,
        "concat_embedding": create_concat_pgvector_index
    },
    "existing": {
        "no_index": lambda: None,
        "spatial_keyword_index": create_spatial_keyword_index,
        "pgvector_embedding": create_pgvector_index,
        "spatial_and_pgvector_embedding": create_embedding_spatial_index
    },
    "non_embedded": {
        "no_index": lambda: None,
        "spatial_index": create_spatial_index,
        "keyword_index": create_keyword_index,
        "spatial_keyword_index": create_spatial_keyword_index
    },
    "embedded": {
        "pgvector_embedding": create_pgvector_index,
        "spatial_and_pgvector_embedding": create_embedding_spatial_index,
        "concat_embedding": create_concat_pgvector_index,
        "contrastive_embedding": create_contrastive_pgvector_index
    }
}
INPUT_LEN = 0

def safe_parse_json(tag_str):
    if isinstance(tag_str, str):
        try:
            return ast.literal_eval(tag_str)  # Convert string to dictionary
        except (ValueError, SyntaxError):
            return {}  # If parsing fails, return empty dictionary
    return {}

def load_csv_data(csv_file: str):
    df = pd.read_csv(csv_file)
    df["tags"] = df["tags"].fillna("{}").apply(safe_parse_json)
    return df

def load_queries_for_k(query_dir: str, k: int, logical_prefix: str, source: str) -> Tuple[List[str], str]:
    sql_basename = f"{source}_{logical_prefix}queries_k{k}.sql"
    query_file = os.path.join(query_dir, sql_basename)
    if os.path.exists(query_file):
        with open(query_file, "r") as f:
            lines = [q for q in f.read().splitlines() if q.strip()]
            return lines
    return []


def finish_embedding_and_setup_database(func, file, length):
    embeddings = []
    # --- semantic-only / spatial+semantic (existing) ---
    if func == create_pgvector_index or func == create_embedding_spatial_index:
        setup_database()
        semantic_path = "./semantic/semantic_embeddings_" + os.path.basename(file).replace(".csv", ".npy")
        sem = SemanticEmbedder.SemanticEmbedder(file, semantic_path); sem.run()
        embeddings = open_embedding(semantic_path)

    # --- concatenation (SE-KE) ---
    elif func == create_concat_pgvector_index:
        total_dim = VECTOR_DIM + 4 * λ
        setup_database(total_dim)
        semantic_path = "./semantic/semantic_embeddings_" + os.path.basename(file).replace(".csv", ".npy")
        sem = SemanticEmbedder.SemanticEmbedder(file, semantic_path); sem.run()
        spatial_path = "./spatial/spatial_embeddings_" + os.path.basename(file).replace(".csv", ".npy")
        spa = SpatialEmbedder.SpatialEmbedder(file); spa.run(spatial_path)
        embeddings = generate_concat_embedding(semantic_path=semantic_path, spatial_path=spatial_path, dim=total_dim)

    # --- contrastive fused (text+coord) ---
    elif func == create_contrastive_pgvector_index:
        proj_dim = CONTRASTIVE_DIM
        setup_database(proj_dim)
        print("Building contrastive embeddings…")
        embeddings = generate_contrastive_embedding(
            input_csv=file,
            ckpt=CONTRASTIVE["ckpt"],
            text_encoder=CONTRASTIVE["text_encoder"],
            spatial_encoder=CONTRASTIVE["spatial_encoder"],
            proj_dim=proj_dim,
            freeze_text=CONTRASTIVE["freeze_text"],
        )

    # --- non-embedded baselines ---
    else:
        setup_database()
        embeddings = [None] * length

    print("Table 'PoIs' constructed.")
    return embeddings


def run_scenario(records: List[Tuple], queries: List[str], index_creation_func, csv_file: str):
    embeddings = finish_embedding_and_setup_database(index_creation_func, csv_file, len(records))
    insertion_time = insert_data(records, embeddings)
    start_time = time.time()
    index_sql = index_creation_func()
    if index_sql:
        with connect_db() as conn, conn.cursor() as cur:
            for q in index_sql:
                cur.execute(q)
            cur.execute("ANALYZE PoIs;")
            conn.commit()
    indexing_time = (time.time() - start_time)
    print("All indexes created in:", indexing_time)

    results = run_queries(queries=queries)
    print("All queries executed.")

    return {
        "insertion_time": insertion_time,
        "index_creation": {"func": index_sql, "time": indexing_time},
        "query_results": results
    }

def benchmark(csv_file: str, k_values: List[int], index_functions: Dict[str, callable], output_dir: str):
    records = load_csv_data(csv_file)
    INPUT_LEN = len(records)
    print(f"{INPUT_CSV} has {INPUT_LEN} rows.")
    os.makedirs(output_dir, exist_ok=True)

    for index_name, index_func in index_functions.items():
        # choose the right workload family
        if "concat" in index_name:
            prefix = "concat_embedded_"
        elif "contrastive" in index_name:
            prefix = "contrastive_embedded_"
        elif "embedding" in index_name:
            prefix = "embedded_"
        else:
            prefix = ""

        for source in SOURCE:
            for k in k_values:
                queries = load_queries_for_k("./data/workloads/" + str(EXPERIMENT), k, prefix, source)
                if not queries:
                    print(f"  ⚠️  No {source} SQL found for k={k} (prefix='{prefix}').")
                    continue

                print(f"Running scenario: {index_name} on {source}, k={k}")
                results = run_scenario(records, queries, index_func, csv_file)

                result_file = os.path.join(output_dir, f"{source}_{index_name}_results_k{k}.json")
                with open(result_file, "w") as f: json.dump(results, f, indent=4)
                print(f"Results saved to {result_file}\n")

def parse_args():
    p = argparse.ArgumentParser(description="Run benchmark scenarios.")
    p.add_argument("--sce", choices=["all","existing","non_embedded","embedded","fused","concat","contrast"], default="embedded")
    p.add_argument("--exp", type=str, default=EXPERIMENT)
    p.add_argument("--so", nargs="+", default=SOURCE)
    p.add_argument("--k", nargs="+", type=int, default=K_VALUES)
    p.add_argument("--l", type=int, default=λ)
    p.add_argument("--idx", choices=["hnsw","ivfflat"], type=str, default=INDEX_TYPE)
    p.add_argument("--input", type=str, default=INPUT_CSV)
    # contrastive overrides
    p.add_argument("--c-ckpt", type=str, default=CONTRASTIVE["ckpt"])
    p.add_argument("--c-proj-dim", type=int, default=CONTRASTIVE["proj_dim"])
    p.add_argument("--c-text-encoder", type=str, default=CONTRASTIVE["text_encoder"])
    p.add_argument("--c-spatial-encoder", type=str, default=CONTRASTIVE["spatial_encoder"])
    p.add_argument("--c-freeze", action="store_true", default=CONTRASTIVE["freeze_text"])
    p.add_argument("--c-wtext", type=float, default=1.0)
    p.add_argument("--c-wspatial", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # override globals for downstream modules
    import Indexing as IDX, DataEmbedding as DE
    λ = args.l; DE.λ = args.l
    INDEX_TYPE = args.idx; IDX.INDEX_TYPE = args.idx
    INPUT_CSV = args.input
    EXPERIMENT = args.exp; SOURCE = args.so; K_VALUES = args.k

    # build a unique checkpoint path based on encoder + dim
    ckpt_path = args.c_ckpt
    if ckpt_path == "./contrastive/contrastive_model.pt":
        ckpt_path = f"./contrastive/model_{args.c_spatial_encoder}_d{args.c_proj_dim}.pt"

    CONTRASTIVE.update({
        "ckpt": ckpt_path,
        "proj_dim": args.c_proj_dim,
        "text_encoder": args.c_text_encoder,
        "spatial_encoder": args.c_spatial_encoder,
        "freeze_text": args.c_freeze,
        "w_text": args.c_wtext,
        "w_spatial": args.c_wspatial
    })
    CONTRASTIVE_DIM = CONTRASTIVE["proj_dim"]


    print("=== Benchmark Config ===")
    print(f"Scenario     : {args.sce}")
    print(f"Input CSV    : {args.input}")
    print(f"K Values     : {args.k}")
    print(f"Lambda (λ)   : {args.l}")
    print(f"Index Type   : {args.idx}")
    print(f"Experiment   : {args.exp}")
    print(f"Source       : {args.so}")
    print(f"Contrastive  : {CONTRASTIVE}")
    print("========================")

    scenario_funcs = SCENARIOS[args.sce]
    benchmark(args.input, args.k, scenario_funcs, output_dir="./results/" + args.exp)