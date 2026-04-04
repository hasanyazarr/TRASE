"""
Visualize learned Gaussian features from a saved point_cloud.ply.

Outputs two colored .ply files next to the input:
  - *_pca.ply    : PCA of 32-dim features → RGB (continuous)
  - *_kmeans.ply : K-means cluster IDs → distinct colors (discrete)

Usage:
    python self_supervised_scripts/visualize_features.py \
        output/8abe732a-1/point_cloud/iteration_20000_features/point_cloud.ply \
        --n_clusters 8
"""

import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


CLUSTER_PALETTE = np.array([
    [230,  25,  75], [60,  180,  75], [ 67, 99, 216], [255, 225,  25],
    [245, 130,  49], [145,  30, 180], [ 66, 212, 244], [240,  50, 230],
    [188, 246,  12], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [154,  99,  36], [255, 250, 200], [128,   0,   0], [170, 255, 195],
    [128, 128,   0], [255, 216, 177], [  0,   0, 117], [128, 128, 128],
], dtype=np.uint8)


def load_ply_features(path):
    plydata = PlyData.read(path)
    el = plydata.elements[0]

    xyz = np.stack([np.asarray(el['x']),
                    np.asarray(el['y']),
                    np.asarray(el['z'])], axis=1)

    feat_dim = sum(1 for p in el.properties if p.name.startswith('gaussian_feats_'))
    feats = np.stack([np.asarray(el[f'gaussian_feats_{i}'])
                      for i in range(feat_dim)], axis=1)   # [N, D]

    print(f"Loaded {xyz.shape[0]:,} Gaussians, feature dim={feat_dim}")
    return xyz, feats


def write_colored_ply(path, xyz, rgb):
    """rgb: [N, 3] uint8"""
    elements = np.empty(xyz.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red']   = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue']  = rgb[:, 2]
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    print(f"Saved: {path}")


def feats_to_pca_rgb(feats):
    """Project 32-dim features to 3-dim via PCA, normalize to [0, 255]."""
    f_norm = normalize(feats, norm='l2')
    pca = PCA(n_components=3, random_state=0)
    proj = pca.fit_transform(f_norm)                    # [N, 3]
    proj -= proj.min(axis=0)
    proj /= proj.max(axis=0).clip(min=1e-6)
    return (proj * 255).astype(np.uint8)


def feats_to_kmeans_rgb(feats, n_clusters):
    f_norm = normalize(feats, norm='l2')
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = km.fit_predict(f_norm)
    colors = CLUSTER_PALETTE[labels % len(CLUSTER_PALETTE)]
    counts = np.bincount(labels, minlength=n_clusters)
    print(f"K-means cluster sizes: {sorted(counts, reverse=True)}")
    return colors.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_path', type=str,
                        help='Path to point_cloud.ply with gaussian_feats_* columns')
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='Number of K-means clusters')
    args = parser.parse_args()

    xyz, feats = load_ply_features(args.ply_path)
    base = os.path.splitext(args.ply_path)[0]

    print("\n--- PCA visualization ---")
    pca_rgb = feats_to_pca_rgb(feats)
    write_colored_ply(base + '_pca.ply', xyz, pca_rgb)

    print(f"\n--- K-means (k={args.n_clusters}) ---")
    km_rgb = feats_to_kmeans_rgb(feats, args.n_clusters)
    write_colored_ply(base + f'_kmeans_k{args.n_clusters}.ply', xyz, km_rgb)


if __name__ == '__main__':
    main()
