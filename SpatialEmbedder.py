import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

class SpatialEmbedder:
    def __init__(self, input_csv: str, chunk_size: int = 500_000):
        self.input_csv = input_csv
        self.chunk_size = chunk_size
        self.elapsed_time = 0

        self.df = pd.read_csv(self.input_csv)
        self.total_rows = len(self.df)
        self.southing = self.df["lat"].min()
        self.northing = self.df["lat"].max()
        self.westing = self.df["lon"].min()
        self.easting = self.df["lon"].max()
        mean_lat = (self.southing + self.northing) / 2
        self.lat_range = self.northing - self.southing
        self.lon_range = (self.easting - self.westing) * np.cos(np.radians(mean_lat))
        self.spatial_dim = 4

    def run(self, output_npy):
        if os.path.exists(output_npy):
            embedding = np.load(output_npy, mmap_mode='r')
            csv_rows = sum(1 for _ in open(self.input_csv)) - 1  # header hariç
            if embedding.shape[0] == csv_rows:
                print("Spatial Embeddings:", embedding.shape[0] == csv_rows)
                return
        os.makedirs(os.path.dirname(output_npy), exist_ok=True)
        print("🔧 Generating spatial-only embeddings")
        fused_array = np.lib.format.open_memmap(
            output_npy,
            dtype=np.float32,
            mode='w+',
            shape=(self.total_rows, self.spatial_dim)
        )
        start_time = time.time()

        for start in range(0, self.total_rows, self.chunk_size):
            end = min(start + self.chunk_size, self.total_rows)
            print(f"➡️ Processing rows {start} to {end}")

            for j, (_, row) in tqdm(
                enumerate(self.df.iloc[start:end].iterrows()),
                total=(end - start),
                desc=f"Chunk {start // self.chunk_size + 1}/{self.total_rows // self.chunk_size + 1}",
                leave=False
            ):
                spatial_enc = self._encode_relative(row["lat"], row["lon"])
                #NORMALIZE
                spatial_enc /= np.linalg.norm(spatial_enc)
                fused_array[start + j] = spatial_enc
        
        self.elapsed_time += time.time() - start_time

        return fused_array

    def _encode_relative(self, lat, lon):
        coslat = np.cos(np.radians(lat))
        north = 1 - (lat - self.southing) / self.lat_range
        south = 1 - (self.northing - lat) / self.lat_range
        east = 1 - coslat * (lon - self.westing) / self.lon_range
        west = 1 - coslat * (self.easting - lon) / self.lon_range
        v = np.array([north, south, east, west], dtype=np.float32)
        return v
    
# Example usage
if __name__ == "__main__":
    embedder = SpatialEmbedder(
        input_csv="./data/melbourne_cleaned.csv",
        chunk_size=500_000
    )
    embedder.run(output_npy="./spatial/spatial_embeddings.npy")
