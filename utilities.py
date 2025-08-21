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
    results = {}

    for i, query in enumerate(queries):
        conn = connect_db()
        conn.autocommit = True  # so DISCARD works cleanly
        cur = conn.cursor()
        try:
            # keep plans deterministic and avoid JIT warmups
            cur.execute("SET local jit = off;")
            cur.execute("DISCARD ALL;")  # clear session-local caches

            start_time = time.time()
            cur.execute(query)
            rows = cur.fetchall()
            execution_time = time.time() - start_time

            # ... same result packaging as before ...
            has_sims = (len(rows) > 0 and len(rows[0]) >= 4)
            results[i] = {"execution_time": execution_time, "results": []}
            for row in rows:
                item = {"id": row[0], "tags": row[1], "distance": row[2]}
                if has_sims:
                    item["sims"] = row[3]
                results[i]["results"].append(item)
        finally:
            cur.close()
            conn.close()

    return results
