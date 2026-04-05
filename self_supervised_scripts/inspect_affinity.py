"""
Inspect affinity graph components before running spectral clustering.

Prints per-component statistics and saves diagnostic plots to verify
that each affinity signal (Apos, Acolor, Aorient, Ascale, W) is
well-distributed and not degenerate.

Usage:
    python self_supervised_scripts/inspect_affinity.py \
        -s data/HyperNeRF/americano \
        --model_path output/8abe732a-1 \
        --load_iteration 20000
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless server
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state

from self_supervised_scripts.affinity_graph import AffinityGraph


@torch.no_grad()
def main(dataset, opt, args):
    # ── Load Stage 1 checkpoint ───────────────────────────────────────────
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration,
                  shuffle=False)

    N_total = gaussians.get_xyz.shape[0]
    print(f"\nTotal Gaussians: {N_total:,}")

    # ── Build affinity graph ──────────────────────────────────────────────
    print(f"\nBuilding affinity graph (k={args.k})...")
    graph = AffinityGraph(
        gaussians,
        k=args.k,
        opacity_thresh=args.opacity_thresh,
        sigma_pos=args.sigma_pos,
        sigma_color=args.sigma_color,
        sigma_scale=args.sigma_scale,
    )
    edge_index, W, valid, components = graph.build(return_components=True)

    N_valid = valid.sum().item()
    E = W.shape[0]
    print(f"Gaussians after opacity filter: {N_valid:,} / {N_total:,} "
          f"({100 * N_valid / N_total:.1f}%)")
    print(f"Edges: {E:,}  (avg {E / N_valid:.1f} per node)")

    # ── Data-driven sigma diagnostics ─────────────────────────────────────
    pos_all = gaussians.get_xyz[valid]
    color_all = gaussians._features_dc[valid].squeeze(1)
    i_idx, j_idx = edge_index[0], edge_index[1]

    dists = ((pos_all[i_idx] - pos_all[j_idx]) ** 2).sum(dim=1).sqrt().cpu()
    color_diffs = ((color_all[i_idx] - color_all[j_idx]) ** 2).sum(dim=1).sqrt().cpu()

    print(f"\nSigma calibration hints (based on actual k-NN pairs):")
    print(f"  k-NN pos dist : mean={dists.mean():.4f}  median={dists.median():.4f}  p95={dists.quantile(0.95):.4f}  → suggest sigma_pos ≈ {dists.median():.4f}")
    print(f"  color diff    : mean={color_diffs.mean():.4f}  median={color_diffs.median():.4f}  p95={color_diffs.quantile(0.95):.4f}  → suggest sigma_color ≈ {color_diffs.median():.4f}")

    # ── Per-component statistics ──────────────────────────────────────────
    all_tensors = {**components, 'W': W}
    print(f"\n{'Component':<12} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  {'~0% (dead)':>12}  {'~1% (sat)':>12}")
    print("-" * 78)
    for name, t in all_tensors.items():
        t_cpu = t.float().cpu()
        print(
            f"{name:<12} "
            f"{t_cpu.mean().item():>8.4f} "
            f"{t_cpu.std().item():>8.4f} "
            f"{t_cpu.min().item():>8.4f} "
            f"{t_cpu.max().item():>8.4f}  "
            f"{(t_cpu < 0.05).float().mean().item() * 100:>10.1f}%  "
            f"{(t_cpu > 0.95).float().mean().item() * 100:>10.1f}%"
        )

    # ── Plots ─────────────────────────────────────────────────────────────
    out_dir = os.path.join(dataset.model_path, "affinity_inspect")
    os.makedirs(out_dir, exist_ok=True)

    _plot_histograms(all_tensors, out_dir)
    _plot_spatial(gaussians, valid, W, edge_index, out_dir, args.scatter_subsample)

    print(f"\nPlots saved to: {out_dir}")


def _plot_histograms(all_tensors, out_dir):
    """One subplot per component showing value distribution."""
    names = list(all_tensors.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    fig.suptitle("Affinity Component Distributions", fontsize=13)

    for ax, name in zip(axes, names):
        vals = all_tensors[name].float().cpu().numpy()
        ax.hist(vals, bins=50, color='steelblue', edgecolor='none', alpha=0.85)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Edge count")
        ax.axvline(vals.mean(), color='red', linewidth=1.2, linestyle='--',
                   label=f"mean={vals.mean():.3f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "component_histograms.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_spatial(gaussians, valid, W, edge_index, out_dir, subsample):
    """
    Top-down scatter of Gaussian positions (XZ plane) colored by
    per-node mean affinity W. Reveals spatial structure of the graph.
    """
    pos = gaussians.get_xyz[valid].cpu().float()  # [N', 3]
    N = pos.shape[0]

    # Compute per-node mean affinity
    src = edge_index[0].cpu()
    node_mean_W = torch.zeros(N)
    node_count  = torch.zeros(N)
    node_mean_W.scatter_add_(0, src, W.cpu().float())
    node_count.scatter_add_(0, src, torch.ones(W.shape[0]))
    node_count = node_count.clamp(min=1)
    node_mean_W = node_mean_W / node_count  # [N']

    # Subsample for scatter plot
    if N > subsample:
        idx = torch.randperm(N)[:subsample]
        pos_plot = pos[idx]
        w_plot   = node_mean_W[idx].numpy()
    else:
        pos_plot = pos
        w_plot   = node_mean_W.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Gaussian Positions Colored by Mean Neighbor Affinity W",
                 fontsize=12)

    planes = [
        ('XY', pos_plot[:, 0].numpy(), pos_plot[:, 1].numpy()),
        ('XZ', pos_plot[:, 0].numpy(), pos_plot[:, 2].numpy()),
        ('YZ', pos_plot[:, 1].numpy(), pos_plot[:, 2].numpy()),
    ]
    for ax, (label, x, y) in zip(axes, planes):
        sc = ax.scatter(x, y, c=w_plot, cmap='coolwarm', s=0.3,
                        vmin=0, vmax=1, alpha=0.6)
        ax.set_title(f"{label} plane")
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, fraction=0.046, label='mean W')

    plt.tight_layout()
    path = os.path.join(out_dir, "spatial_affinity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--load_iteration",   type=int,   default=20000)
    parser.add_argument("--k",                type=int,   default=20)
    parser.add_argument("--opacity_thresh",   type=float, default=0.05)
    parser.add_argument("--sigma_pos",        type=float, default=0.1)
    parser.add_argument("--sigma_color",      type=float, default=0.3)
    parser.add_argument("--sigma_scale",      type=float, default=1.0)
    parser.add_argument("--scatter_subsample",type=int,   default=50000,
                        help="Max Gaussians to plot in scatter (performance)")

    args = parser.parse_args(sys.argv[1:])
    safe_state(False)

    main(lp.extract(args), op.extract(args), args)
