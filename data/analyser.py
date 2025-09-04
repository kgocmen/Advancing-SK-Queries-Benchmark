#!/usr/bin/env python3
import sys
import os
import ast
from collections import Counter
import polars as pl
import matplotlib.pyplot as plt
import math

# -----------------------
# Helpers
# -----------------------
def safe_parse_dict(x):
    try:
        if isinstance(x, str) and x.strip().startswith("{"):
            return ast.literal_eval(x)
    except Exception:
        pass
    return {}

def plot_top(counter: Counter, title: str, out_path: str, top_n: int = 50, fmt=None):
    items = counter.most_common(top_n)
    if not items:
        print(f"(No data to plot for {title})")
        return
    labels, counts = zip(*items)
    if fmt:
        labels = [fmt(l) for l in labels]
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 Saved plot → {out_path}")

def basic(df: pl.DataFrame):
    print("Data Shape:", df.shape)
    print("\nColumns & Types:")
    for n, d in df.schema.items():
        print(f"{n}: {d}")

    numeric_cols = [c for c, d in df.schema.items() if d == pl.Float64]
    print("\nSummary")
    if numeric_cols:
        stats = df.select(
            [pl.col(c).mean().alias(f"{c}_mean") for c in numeric_cols]
            + [pl.col(c).median().alias(f"{c}_median") for c in numeric_cols]
            + [pl.col(c).min().alias(f"{c}_min") for c in numeric_cols]
            + [pl.col(c).max().alias(f"{c}_max") for c in numeric_cols]
        )
        print(stats)
    else:
        print("No numeric columns found.")

    # -----------------------------
    # Custom dataset facts
    # -----------------------------
    row_count = df.height
    south = df["lat"].min()
    north = df["lat"].max()
    west  = df["lon"].min()
    east  = df["lon"].max()
    mean_lat = (south + north) / 2

    # Distances
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(mean_lat))
    ns_dist = (north - south) * km_per_deg_lat
    ew_dist = (east - west) * km_per_deg_lon
    area = ns_dist * ew_dist
    density = row_count / area if area > 0 else float("nan")

    print("\n--- Dataset Facts ---")
    print(f"Row count: {row_count:,}")
    print(f"Southmost latitude: {south:.4f}")
    print(f"Northmost latitude: {north:.4f}")
    print(f"Westmost longitude: {west:.4f}")
    print(f"Eastmost longitude: {east:.4f}")
    print(f"Mean latitude: {mean_lat:.4f}")
    print(f"N-S distance: {ns_dist:.2f} km")
    print(f"E-W distance: {ew_dist:.2f} km")
    print(f"Approx. area: {area:.2f} km²")
    print(f"Density: {density:.2f} rows/km²")


def distinct_txt(df: pl.DataFrame):
    if "tags" not in df.columns:
        print("(No 'tags' column; skipping tag frequency plots)")
        return

    key_count, value_count, pair_count = Counter(), Counter(), Counter()
    for i, v in enumerate(df["tags"].to_list(), 1):
        d = safe_parse_dict(v)
        if isinstance(d, dict):
            for k, val in d.items():
                key_count[k] += 1
                value_count[val] += 1
                pair_count[(k, val)] += 1

    plot_top(key_count, "Top 50 tag keys", "top_keys_50.png")
    plot_top(pair_count, "Top 50 key:value pairs", "top_pairs_50.png",
             fmt=lambda kv: f"{kv[0]}={kv[1]}")

# -----------------------
# Main
# -----------------------
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <input.csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    df = pl.read_csv(input_csv)
    basic(df)
    distinct_txt(df)

if __name__ == "__main__":
    main()
