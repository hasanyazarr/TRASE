"""
Render learned feature clusters onto the original scene.

Runs K-means on the saved Gaussian features, assigns cluster colors,
and renders all training views using the existing Gaussian renderer.

Usage:
    python self_supervised_scripts/render_clusters.py \
        -s data/HyperNeRF/americano \
        --model_path output/8abe732a-1 \
        --deform_path /okyanus/users/mtuncel/TRASE \
        --load_iteration 20000 \
        --n_clusters 8
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import torchvision
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize

from scene import Scene, GaussianModel, DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from gaussian_renderer import render


CLUSTER_PALETTE = torch.tensor([
    [230,  25,  75], [60,  180,  75], [ 67,  99, 216], [255, 225,  25],
    [245, 130,  49], [145,  30, 180], [ 66, 212, 244], [240,  50, 230],
    [188, 246,  12], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [154,  99,  36], [255, 250, 200], [128,   0,   0], [170, 255, 195],
], dtype=torch.float32) / 255.0  # [P, 3]


@torch.no_grad()
def main(dataset, opt, pipe, args):
    # ------------------------------------------------------------------
    # Load Stage 1 checkpoint
    # ------------------------------------------------------------------
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration)

    deform = DeformModel(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_type=opt.deform_type,
    )
    deform_path = args.deform_path if args.deform_path else dataset.model_path
    deform.load_weights(deform_path, iteration=args.load_iteration)

    # ------------------------------------------------------------------
    # Load saved features
    # ------------------------------------------------------------------
    feat_ply = os.path.join(
        dataset.model_path, "point_cloud",
        f"iteration_{args.load_iteration}_features", "point_cloud.ply"
    )
    print(f"Loading features from: {feat_ply}")

    # Re-load gaussian features from the feature .ply
    gaussians.load_ply(feat_ply)
    feats = gaussians.get_gaussian_features.squeeze(1)  # [N, 32]
    print(f"Features: {feats.shape}")

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    f_np = normalize(feats.cpu().numpy(), norm='l2')

    if args.cluster_method == 'dbscan':
        print(f"Running DBSCAN (eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples})...")
        db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, metric='euclidean', n_jobs=-1)
        labels = db.fit_predict(f_np)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        print(f"Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(labels):.1f}%)")
        counts = np.bincount(labels[labels >= 0], minlength=n_clusters)
        print(f"Cluster sizes: {sorted(counts, reverse=True)}")
        out_suffix = f"dbscan_eps{args.dbscan_eps}_min{args.dbscan_min_samples}"
    else:
        print(f"Running K-means (k={args.n_clusters})...")
        km = KMeans(n_clusters=args.n_clusters, random_state=0, n_init='auto')
        labels = km.fit_predict(f_np)
        counts = np.bincount(labels, minlength=args.n_clusters)
        print(f"Cluster sizes: {sorted(counts, reverse=True)}")
        out_suffix = f"k{args.n_clusters}"

    # Assign cluster colors — noise points (label=-1) → grey
    noise_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    colors_list = []
    for lbl in labels:
        if lbl == -1:
            colors_list.append(noise_color)
        else:
            colors_list.append(CLUSTER_PALETTE[lbl % len(CLUSTER_PALETTE)])
    cluster_colors = torch.stack(colors_list).cuda()  # [N, 3]

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()
    out_dir = os.path.join(dataset.model_path, "cluster_renders",
                           f"iteration_{args.load_iteration}_{out_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {len(views)} views to: {out_dir}\n")

    for idx, view in enumerate(tqdm(views, desc="Rendering")):
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        d_xyz, d_rotation, d_scaling = deform.step(
            xyz.detach(), time_input
        ) if opt.deform_type == 'DeformNetwork' else deform.step(
            xyz.detach(), time_input, gaussians.get_gaussian_features.squeeze(1)
        )

        result = render(view, gaussians, pipe, background,
                        d_xyz, d_rotation, d_scaling,
                        is_6dof=dataset.is_6dof,
                        override_color=cluster_colors)

        torchvision.utils.save_image(
            result["render"].cpu(),
            os.path.join(out_dir, f"{idx:05d}.png")
        )

    print(f"\nDone. Images saved to: {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--load_iteration", type=int, default=20000)
    parser.add_argument("--deform_path", type=str, default="")
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "dbscan"])
    parser.add_argument("--dbscan_eps", type=float, default=0.3)
    parser.add_argument("--dbscan_min_samples", type=int, default=10)

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet if hasattr(args, 'quiet') else False)

    main(lp.extract(args), op.extract(args), pp.extract(args), args)
