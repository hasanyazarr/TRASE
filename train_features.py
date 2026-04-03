import os
import sys
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from scene import Scene, GaussianModel, DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state

from self_supervised_scripts.gaussian_descriptor import GaussianDescriptor
from self_supervised_scripts.segmentation_mlp import SegmentationMLP
from self_supervised_scripts.self_supervised_losses import motion_affinity_loss, MOTION_SAMPLING_MODES


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
    deform.load_weights(dataset.model_path, iteration=args.load_iteration)

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

    progress = tqdm(range(1, args.iterations + 1), desc="Feature Training")
    for iteration in progress:

        # Forward: descriptor → MLP → features
        f = mlp(h)   # [N, 32]

        # Motion affinity loss
        loss = motion_affinity_loss(
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.set_postfix({"loss": f"{loss.item():.4f}"})

        if iteration % args.log_interval == 0:
            print(f"[iter {iteration:05d}] loss={loss.item():.4f}")

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

    save_path = os.path.join(dataset.model_path, "point_cloud",
                             f"iteration_{args.load_iteration}_features")
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

    # Descriptor
    parser.add_argument("--T", type=int, default=8,
                        help="Number of trajectory timestamps")

    # Training
    parser.add_argument("--iterations", type=int, default=10000,
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

    args = parser.parse_args(sys.argv[1:])

    print("Feature training — SAM-free self-supervised segmentation")
    safe_state(args.quiet if hasattr(args, 'quiet') else False)

    training(lp.extract(args), op.extract(args), pp.extract(args), args)
