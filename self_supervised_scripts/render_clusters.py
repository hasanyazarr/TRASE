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
import torch
import numpy as np
import torchvision
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.cluster import KMeans
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
    # K-means clustering
    # ------------------------------------------------------------------
    print(f"Running K-means (k={args.n_clusters})...")
    f_np = normalize(feats.cpu().numpy(), norm='l2')
    km = KMeans(n_clusters=args.n_clusters, random_state=0, n_init='auto')
    labels = km.fit_predict(f_np)
    counts = np.bincount(labels, minlength=args.n_clusters)
    print(f"Cluster sizes: {sorted(counts, reverse=True)}")

    # Assign cluster colors [N, 3]
    cluster_colors = CLUSTER_PALETTE[labels % len(CLUSTER_PALETTE)].cuda()  # [N, 3]

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()
    out_dir = os.path.join(dataset.model_path, "cluster_renders",
                           f"iteration_{args.load_iteration}_k{args.n_clusters}")
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

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet if hasattr(args, 'quiet') else False)

    main(lp.extract(args), op.extract(args), pp.extract(args), args)
