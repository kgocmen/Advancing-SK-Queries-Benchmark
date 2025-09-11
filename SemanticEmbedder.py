import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from constants import *

class SemanticEmbedder:
    def __init__(
        self,
        input_csv: str,
        output_npy: str,
        chunk_size: int = 100_000,
        model_name = SEMANTIC_MODEL,
    ):
        self.input_csv = input_csv
        self.output_npy = output_npy
        self.chunk_size = chunk_size
        self.model = model_name

        self.elapsed_time = 0

        # Make sure target directory exists
        os.makedirs(os.path.dirname(self.output_npy), exist_ok=True)

        # Resume support -----------------------------------------------------
        if os.path.exists(self.output_npy):
            self._embeddings = np.load(self.output_npy, mmap_mode="r")
            self.processed_rows = self._embeddings.shape[0]
            self._first_write = False
        else:
            self._embeddings = None
            self.processed_rows = 0
            self._first_write = True
            
    def run(self):
        if os.path.exists(self.output_npy):
            self._embeddings = np.load(self.output_npy, mmap_mode='r')
            csv_rows = sum(1 for _ in open(self.input_csv)) - 1  # header hariç
            if self._embeddings.shape[0] == csv_rows:
                print("Semantic Embeddings:", self._embeddings.shape[0] == csv_rows)
                return

        reader = pd.read_csv(self.input_csv, chunksize=self.chunk_size)
        current_row = 0
        total_rows = self.processed_rows

        start_time = time.time()
        for i, chunk in enumerate(reader, start=1):
            chunk_start = current_row
            chunk_end = current_row + len(chunk)
            current_row = chunk_end

            # Skip chunks that are already done
            if chunk_end <= self.processed_rows:
                print(f"Skipping chunk {i} (rows {chunk_start}-{chunk_end})")
                continue

            print(f"\nProcessing chunk {i} (rows {chunk_start}-{chunk_end})")

            # --------------------------------------------------------------
            # 1. Parse tags column safely
            chunk["tags"] = chunk["tags"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str)
                else (x if isinstance(x, dict) else {})
            )
            chunk = chunk[chunk["tags"].apply(lambda d: isinstance(d, dict) and len(d) > 0)]

            # --------------------------------------------------------------
            # 2. Embed each tag dict
            embeddings = self._embed_chunk(chunk["tags"], i)

            # --------------------------------------------------------------
            # 3. Persist to disk
            self._append_embeddings(embeddings)

            total_rows += len(embeddings)
            print(f"Appended {len(embeddings)} embeddings (total written: {total_rows})")
        
        self.elapsed_time += time.time() - start_time
        print(f"\n\tDone! All embeddings saved to '{self.output_npy}'")
        return
    
    def _embed_chunk(self, tags_series, chunk_idx):
        texts = []
        for tags in tags_series:
            if isinstance(tags, dict) and tags:
                tag_text = "; ".join(f"{k}: {v}" for k, v in tags.items())
            else:
                tag_text = "poi"
            texts.append(tag_text)
        vecs = self.model.encode(texts, batch_size=1024, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


    def _append_embeddings(self, new_vecs: np.ndarray):
        """Append or create the .npy file in an idempotent way."""
        if self._first_write:
            np.save(self.output_npy, new_vecs)
            self._first_write = False
        else:
            # Memory-map existing, concatenate, then overwrite
            existing = np.load(self.output_npy, mmap_mode="r")
            combined = np.concatenate((existing, new_vecs), axis=0)
            np.save(self.output_npy, combined)