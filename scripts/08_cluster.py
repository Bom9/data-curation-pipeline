#!/usr/bin/env python3
"""
Run PCA + KMeans clustering on DINOv2 embeddings to assign cluster IDs.

Reads:  config.yaml -> clustering.pca.*, clustering.kmeans.*
Reads:  data/output/embeddings.npy, data/output/image_filenames.npy
Writes: data/output/cluster_labels.json  {stem: cluster_id, ...}
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.utils import ensure_dir


def main():
    cfg = load_config()
    output_dir = ensure_dir(cfg["paths"]["output_dir"])

    emb_path = output_dir / "embeddings.npy"
    fnames_path = output_dir / "image_filenames.npy"

    if not emb_path.exists():
        print(f"Error: {emb_path} not found. Run script 07 first.")
        sys.exit(1)

    var_threshold = cfg["clustering"]["pca"]["variance_threshold"]
    k = cfg["clustering"]["kmeans"]["k"]
    seed = cfg["clustering"]["kmeans"]["seed"]

    print("Loading embeddings...")
    embeddings = np.load(emb_path)
    fnames = np.load(fnames_path)
    print(f"  Embeddings: {embeddings.shape}")

    print("Standardizing...")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    print("Running PCA...")
    pca = PCA()
    scores = pca.fit_transform(emb_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum_var, var_threshold) + 1)
    X = scores[:, :n_comp]
    print(f"  {n_comp} components for {var_threshold*100:.0f}% variance (actual: {cum_var[n_comp-1]*100:.1f}%)")

    print(f"Running K-means with k={k}...")
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)

    mapping = {}
    for i, fpath in enumerate(fnames):
        stem = os.path.splitext(os.path.basename(str(fpath)))[0]
        mapping[stem] = int(labels[i])

    out_path = output_dir / "cluster_labels.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)

    dist = Counter(mapping.values())
    print(f"\nDone! {len(mapping):,} images in {len(dist)} clusters")
    print(f"  Min: {min(dist.values())}, Max: {max(dist.values())}, Avg: {len(mapping)/len(dist):.1f}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
