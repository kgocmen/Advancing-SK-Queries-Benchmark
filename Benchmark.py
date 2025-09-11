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
import time

# ---------------------------------------------------------------------
# Scenarios: choose which indexes to create for each run
# ---------------------------------------------------------------------
SCENARIOS = {
    "concat":   {"concat_embedding": create_concat_pgvector_index},
    "contrast": {"contrastive_embedding": create_contrastive_pgvector_index},
    "fused":    {
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
        "concat_embedding": create_concat_pgvector_index,
        "contrastive_embedding": create_contrastive_pgvector_index
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_parse_json(tag_str):
    if isinstance(tag_str, str):
        try:
            return ast.literal_eval(tag_str)
        except (ValueError, SyntaxError):
            return {}
    return {}

def load_csv_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["tags"] = df["tags"].fillna("{}").apply(safe_parse_json)
    return df

def load_queries_for_k(query_dir: str, k: int, logical_prefix: str, source: str) -> List[str]:
    """
    logical_prefix in {"", "embedded_", "concat_embedded_", "contrastive_embedded_"}
    source in {"custom", "dataset"}
    """
    sql_basename = f"{source}_{logical_prefix}queries_k{k}.sql"
    query_file = os.path.join(query_dir, sql_basename)
    if os.path.exists(query_file):
        with open(query_file, "r") as f:
            return [q for q in f.read().splitlines() if q.strip()]
    return []

def finish_embedding_and_setup_database(func, file, length):
    """
    Builds (or opens) the embeddings consistent with the chosen index creator.
    Returns the embeddings list/array to pass into insert_data.
    """
    embeddings = []

    # semantic-only / spatial+semantic (existing)
    if func in (create_pgvector_index, create_embedding_spatial_index):
        setup_database()
        semantic_path = "./semantic/semantic_embeddings_" + os.path.basename(file).replace(".csv", ".npy")
        sem = SemanticEmbedder.SemanticEmbedder(file, semantic_path)
        sem.run()
        print(sem.elapsed_time, "seconds for semantic embedding.")
        embeddings = open_embedding(semantic_path)

    # concatenation (SE ⊕ spatial features)
    elif func is create_concat_pgvector_index:
        total_dim = VECTOR_DIM + 4 * λ
        setup_database(total_dim)
        semantic_path = "./semantic/semantic_embeddings_" + os.path.basename(file).replace(".csv", ".npy")
        sem = SemanticEmbedder.SemanticEmbedder(file, semantic_path)
        sem.run()
        print(sem.elapsed_time, "seconds for semantic embedding.")
        spatial_path = "./spatial/spatial_embeddings_" + os.path.basename(file).replace(".csv", ".npy")
        spa = SpatialEmbedder.SpatialEmbedder(file)
        spa.run(spatial_path)
        print(spa.elapsed_time, "seconds for spatial embedding.")
        embeddings = generate_concat_embedding(
            semantic_path=semantic_path,
            spatial_path=spatial_path,
            dim=total_dim
        )

    # contrastive fused (text + coords via sinusoidal-only model)
    elif func is create_contrastive_pgvector_index:
        proj_dim = CONTRASTIVE_DIM
        setup_database(proj_dim)
        print("Building contrastive embeddings…")
        embeddings = generate_contrastive_embedding(
            input_csv=file,
            ckpt=CONTRASTIVE["ckpt"],
            proj_dim=proj_dim,
            freeze_text=CONTRASTIVE["freeze_text"],
            w_text=CONTRASTIVE.get("w_text", 1.0),
            w_spatial=CONTRASTIVE.get("w_spatial", 1.0),
        )

    # non-embedded baselines
    else:
        setup_database()
        embeddings = [None] * length

    print("Table 'PoIs' constructed.")
    return embeddings

def run_scenario(records: pd.DataFrame, queries: List[str], index_creation_func, csv_file: str):
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
    input_len = len(records)
    print(f"{csv_file} has {input_len} rows.")
    os.makedirs(output_dir, exist_ok=True)

    for index_name, index_func in index_functions.items():
        # choose the right workload family prefix
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
                with open(result_file, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Results saved to {result_file}\n")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run benchmark scenarios.")
    p.add_argument("--sce", choices=list(SCENARIOS.keys()), default="embedded")
    p.add_argument("--exp", type=str, default=EXPERIMENT)
    p.add_argument("--so", nargs="+", default=SOURCE)
    p.add_argument("--k", nargs="+", type=int, default=K_VALUES)
    p.add_argument("--l", type=int, default=λ)
    p.add_argument("--idx", choices=["hnsw","ivfflat"], type=str, default=INDEX_TYPE)
    p.add_argument("--input", type=str, default=INPUT_CSV)
    # contrastive
    p.add_argument("--c-ckpt", type=str, default=CONTRASTIVE["ckpt"])
    p.add_argument("--c-proj-dim", type=int, default=CONTRASTIVE["proj_dim"])
    p.add_argument("--c-freeze", action="store_true", default=CONTRASTIVE["freeze_text"])
    p.add_argument("--c-wtext", type=float, default=CONTRASTIVE.get("w_text", 1.0))
    p.add_argument("--c-wspatial", type=float, default=CONTRASTIVE.get("w_spatial", 1.0))
    return p.parse_args()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # propagate overrides for downstream modules
    import Indexing as IDX, DataEmbedding as DE
    λ = args.l; DE.λ = args.l
    INDEX_TYPE = args.idx; IDX.INDEX_TYPE = args.idx
    INPUT_CSV = args.input
    EXPERIMENT = args.exp
    SOURCE = args.so
    K_VALUES = args.k

    # build a simple checkpoint path based on dim if placeholder is used
    ckpt_path = args.c_ckpt
    if ckpt_path == "./contrastive/contrastive_model.pt":
        ckpt_path = f"./contrastive/d{args.c_proj_dim}.pt"

    CONTRASTIVE.update({
        "ckpt": ckpt_path,
        "proj_dim": args.c_proj_dim,
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