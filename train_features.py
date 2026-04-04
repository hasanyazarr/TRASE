import os
import sys
import math
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from scene import Scene, GaussianModel, DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state

from self_supervised_scripts.gaussian_descriptor import GaussianDescriptor
from self_supervised_scripts.segmentation_mlp import SegmentationMLP
from self_supervised_scripts.self_supervised_losses import motion_affinity_loss, spatial_coherence_loss, rendering_coherence_loss, MOTION_SAMPLING_MODES
from gaussian_renderer import render as gs_render


def training(dataset, opt, pipe, args):
    # ------------------------------------------------------------------
    # 1. Load Stage 1 checkpoint (frozen)
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

    # Freeze everything — backbone is never updated
    for p in deform.deform.parameters():
        p.requires_grad_(False)
    gaussians._xyz.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)

    # ------------------------------------------------------------------
    # 2. Build descriptor once (trajectory is expensive, cache it)
    # ------------------------------------------------------------------
    print(f"\nBuilding Gaussian descriptors (T={args.T} timestamps)...")
    descriptor = GaussianDescriptor(gaussians, deform, T=args.T)
    h, trajectories = descriptor.build()
    print(f"Descriptor shape: {h.shape}  |  Trajectories shape: {trajectories.shape}")

    # ------------------------------------------------------------------
    # 3. Init Segmentation MLP
    # ------------------------------------------------------------------
    dim_in = h.shape[1]   # 9 + T*3
    mlp = SegmentationMLP(dim_in=dim_in, dim_out=32).cuda()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting feature training for {args.iterations} iterations")
    print(f"Motion sampling mode: {args.motion_sampling}\n")

    positions = gaussians._xyz.detach()

    # Fixed random projection 32 → render_feature_dim for feature rendering (not trained).
    # renderer only supports 3-channel passes, so we use ceil(D/3) passes and concatenate.
    D = args.render_feature_dim
    n_passes = math.ceil(D / 3)
    proj = torch.nn.functional.normalize(
        torch.randn(32, n_passes * 3, device='cuda'), dim=0
    )  # [32, n_passes*3]

    # Background and training views for rendering coherence
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    train_views = scene.getTrainCameras()

    progress = tqdm(range(1, args.feature_iterations + 1), desc="Feature Training")
    for iteration in progress:

        # Forward: descriptor → MLP → features
        f = mlp(h)   # [N, 32]

        # Motion affinity loss
        loss_motion = motion_affinity_loss(
            f=f,
            trajectories=trajectories,
            positions=positions,
            mode=args.motion_sampling,
            num_pairs=args.num_pairs,
            spatial_radius=args.spatial_radius,
            K=args.knn_K,
            tau_pos=args.tau_pos,
            tau_neg=args.tau_neg,
            margin=args.margin,
        )

        # Spatial coherence loss
        loss_spatial = spatial_coherence_loss(
            f=f,
            positions=positions,
            num_pairs=args.num_pairs,
            spatial_radius=args.spatial_radius,
            sigma=args.spatial_sigma,
        )

        # Rendering coherence loss (every render_interval iterations)
        loss_render = torch.tensor(0.0, device='cuda')
        if args.render_weight > 0 and iteration % args.render_interval == 0:
            view = train_views[torch.randint(len(train_views), (1,)).item()]
            fid = view.fid
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

            with torch.no_grad():
                d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                rgb_result = gs_render(view, gaussians, pipe, background,
                                       d_xyz, d_rotation, d_scaling,
                                       is_6dof=dataset.is_6dof)
                rgb_map   = rgb_result["render"].detach()    # [3, H, W]
                depth_map = rgb_result["depth"].detach()     # [1, H, W]

            # Project features 32→D via n_passes of 3-channel renders (differentiable).
            f_proj = f @ proj   # [N, n_passes*3]
            feat_chunks = []
            for p in range(n_passes):
                f_chunk = f_proj[:, p * 3:(p + 1) * 3]              # [N, 3]
                f_min = f_chunk.min(dim=0).values
                f_max = f_chunk.max(dim=0).values
                f_chunk = (f_chunk - f_min) / (f_max - f_min + 1e-6)  # [N, 3] in [0,1]
                chunk_rendered = gs_render(view, gaussians, pipe, background,
                                           d_xyz, d_rotation, d_scaling,
                                           is_6dof=dataset.is_6dof,
                                           override_color=f_chunk)["render"]   # [3, H, W]
                feat_chunks.append(chunk_rendered)

            feat_rendered = torch.cat(feat_chunks, dim=0)[:D]   # [D, H, W]

            loss_render = rendering_coherence_loss(
                feat_rendered, rgb_map, depth_map,
                alpha=args.render_alpha, beta=args.render_beta,
                edge_percentile=args.render_edge_percentile,
                num_pairs=args.render_num_pairs,
                margin=args.render_margin,
            )

        loss = loss_motion + args.spatial_weight * loss_spatial + args.render_weight * loss_render

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.set_postfix({"loss": f"{loss.item():.4f}", "mot": f"{loss_motion.item():.4f}", "spa": f"{loss_spatial.item():.4f}", "rnd": f"{loss_render.item():.4f}"})

        if iteration % args.log_interval == 0:
            print(f"[iter {iteration:05d}] loss={loss.item():.4f}  mot={loss_motion.item():.4f}  spa={loss_spatial.item():.4f}  rnd={loss_render.item():.4f}")

    # ------------------------------------------------------------------
    # 5. Save learned features into Gaussians and write .ply
    # ------------------------------------------------------------------
    print("\nSaving features...")
    with torch.no_grad():
        f_final = mlp(h)   # [N, 32]
        # Store back into gaussians for DBSCAN / evaluation
        gaussians._gaussian_features = torch.nn.Parameter(
            f_final.unsqueeze(1),   # [N, 1, 32] to match GaussianModel format
            requires_grad=False,
        )

    run_suffix = f"_{args.run_name}" if args.run_name else ""
    save_path = os.path.join(dataset.model_path, "point_cloud",
                             f"iteration_{args.load_iteration}_features{run_suffix}")
    os.makedirs(save_path, exist_ok=True)
    gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
    torch.save(mlp.state_dict(), os.path.join(save_path, "segmentation_mlp.pth"))
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Self-supervised feature training (SAM-free)")

    # Scene / model
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Stage 1 checkpoint to load
    parser.add_argument("--load_iteration", type=int, default=-1,
                        help="Stage 1 checkpoint iteration to load (-1 = latest)")
    parser.add_argument("--deform_path", type=str, default="",
                        help="Path to deform weights folder (if separate from model_path)")

    # Descriptor
    parser.add_argument("--T", type=int, default=8,
                        help="Number of trajectory timestamps")

    # Training
    parser.add_argument("--feature_iterations", type=int, default=10000,
                        help="Number of feature training iterations")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Segmentation MLP")
    parser.add_argument("--log_interval", type=int, default=500)

    # Motion loss
    parser.add_argument("--motion_sampling", type=str, default="random",
                        choices=MOTION_SAMPLING_MODES,
                        help="Pair sampling strategy: 'random' or 'knn'")
    parser.add_argument("--num_pairs", type=int, default=4096,
                        help="(random mode) number of pairs per iteration")
    parser.add_argument("--spatial_radius", type=float, default=0.5,
                        help="(random mode) max 3D distance between pairs")
    parser.add_argument("--knn_K", type=int, default=64,
                        help="(knn mode) number of neighbors per Gaussian")
    parser.add_argument("--tau_pos", type=float, default=0.8,
                        help="Rigid score threshold for positive pairs")
    parser.add_argument("--tau_neg", type=float, default=0.2,
                        help="Rigid score threshold for negative pairs")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Hinge margin for negative pairs")

    # Spatial coherence loss
    parser.add_argument("--spatial_weight", type=float, default=1.0,
                        help="Weight for spatial coherence loss (λ)")
    parser.add_argument("--spatial_sigma", type=float, default=None,
                        help="Gaussian bandwidth for spatial weights (default: spatial_radius/2)")

    # Rendering coherence loss
    parser.add_argument("--render_weight", type=float, default=1.0,
                        help="Weight for rendering coherence loss (λ)")
    parser.add_argument("--render_interval", type=int, default=10,
                        help="Run rendering coherence every N iterations (expensive)")
    parser.add_argument("--render_alpha", type=float, default=1.0,
                        help="Edge weight for RGB gradient")
    parser.add_argument("--render_beta", type=float, default=0.5,
                        help="Edge weight for depth gradient")
    parser.add_argument("--render_feature_dim", type=int, default=3,
                        help="Dimension of projected features for render loss (must be >= 3; "
                             "uses ceil(D/3) render passes, so multiples of 3 are most efficient)")
    parser.add_argument("--render_edge_percentile", type=float, default=0.7,
                        help="Quantile threshold for edge detection (e.g. 0.7 → top 30%% = edges)")
    parser.add_argument("--render_num_pairs", type=int, default=2048,
                        help="Max number of negative/positive pairs per render loss call")
    parser.add_argument("--render_margin", type=float, default=0.3,
                        help="Hinge margin for negative pairs in render loss")

    # Run identification
    parser.add_argument("--run_name", type=str, default="",
                        help="Optional name suffix for the output folder (e.g. 'rw2_sw03')")

    args = parser.parse_args(sys.argv[1:])

    print("Feature training — SAM-free self-supervised segmentation")
    safe_state(args.quiet if hasattr(args, 'quiet') else False)

    training(lp.extract(args), op.extract(args), pp.extract(args), args)
