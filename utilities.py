import psycopg2
import time
import json
from typing import List, Union
from constants import *
from tqdm import tqdm


def connect_db():
    return psycopg2.connect(**DB_PARAMS)

def setup_database(dim=VECTOR_DIM):
    print("Setting up database...")
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    conn.commit()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cur.execute("DROP TABLE IF EXISTS PoIs CASCADE;")
    conn.commit()
    cur.execute(f"""
        CREATE TABLE PoIs (
            id BIGINT PRIMARY KEY,
            lat DOUBLE PRECISION NOT NULL,
            lon DOUBLE PRECISION NOT NULL,
            geom GEOMETRY(POINT, 4326) NOT NULL,
            tags JSONB,
            embedding VECTOR({dim})
        );
    """)
    conn.commit()
    cur.execute("SELECT to_regclass('public.PoIs');")
    if cur.fetchone()[0] is None:
        raise RuntimeError("Table 'PoIs' was not created successfully.")
    cur.close()
    conn.close()

    print("Table 'PoIs' is ready.")


def insert_data(records, embeddings):
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("SELECT to_regclass('public.PoIs');")
    if cur.fetchone()[0] is None:
        raise RuntimeError("Table 'PoIs' was not created successfully.")

    insert_query = """ INSERT INTO PoIs (id, lat, lon, geom, tags, embedding) 
        VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s)
        ON CONFLICT (id) DO NOTHING; """
    
    start_time = time.time()

    records = list(records.itertuples(index=False, name=None))
    total = len(records)
    assert total == len(embeddings)

    for record, embedding in tqdm(zip(records, embeddings), total=total):
        vec = (embedding.tolist() if hasattr(embedding, "tolist") else embedding) 
        cur.execute(insert_query, (
            record[0],record[1],record[2],  #id, lat, lon
            record[2], record[1],  # (lon,lat) in ST_MakePoint
            json.dumps(record[3]),  # tags as JSON
            vec #embeddings
        ))
    conn.commit()

    insertion_time = time.time() - start_time
    print(f"Inserted {total} records in {insertion_time:.2f} seconds ({total/insertion_time:.2f} records/s)")

    cur.close()
    conn.close()

    return insertion_time

def run_queries(queries: Union[str, List[str]]):
    queries = [queries] if isinstance(queries, str) else queries

    conn = connect_db()
    cur = conn.cursor()
    
    results = {}
    for i, query in enumerate(queries):
        start_time = time.time()
        cur.execute(query)
        rows = cur.fetchall()
        execution_time = time.time() - start_time

        results[i] = {
            "execution_time": execution_time,
            "results": []
        }

        ids   = [row[0] for row in rows]
        tags  = [row[1] for row in rows]
        dists = [row[2] for row in rows]

        # detect if we selected similarity (embedded/concat)
        has_sims = (len(rows) > 0 and len(rows[0]) >= 4)

        for j in range(len(ids)):
            item = {"id": ids[j], "distance": dists[j], "tags": tags[j]}
            if has_sims:
                item["sims"] = rows[j][3]  # 4th column = similarity
            results[i]["results"].append(item)

    
    cur.close()
    conn.close()
    
    return results