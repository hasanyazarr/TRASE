"""
Spectral clustering on the Gaussian affinity graph (Phase 1 / Mode A).

Pipeline:
  1. Build Ageo affinity graph (reuses AffinityGraph)
  2. Symmetrize the directed k-NN graph
  3. Compute normalized graph Laplacian
  4. Extract bottom-k eigenvectors via ARPACK (sparse, CPU)
  5. K-means on eigenvector embedding → cluster labels
  6. Save labels as .npy and annotate the .ply checkpoint
  7. Render all training views with cluster colours

Usage:
    python self_supervised_scripts/spectral_cluster.py \\
        -s data/HyperNeRF/americano \\
        --model_path output/8abe732a-1 \\
        --load_iteration 20000 \\
        --n_clusters 5 \\
        --sigma_pos 0.0036 \\
        --sigma_color 0.5160
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import torchvision
from plyfile import PlyData, PlyElement

from scene import Scene, GaussianModel, DeformModel
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from gaussian_renderer import render

from self_supervised_scripts.affinity_graph import AffinityGraph


CLUSTER_PALETTE = torch.tensor([
    [  0,   0,   0],  # index 0 — filtered-out / invalid Gaussians (black = invisible)
    [230,  25,  75], [ 60, 180,  75], [ 67,  99, 216], [255, 225,  25],
    [245, 130,  49], [145,  30, 180], [ 66, 212, 244], [240,  50, 230],
    [188, 246,  12], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [154,  99,  36], [255, 250, 200], [128,   0,   0], [170, 255, 195],
], dtype=torch.float32) / 255.0  # [P, 3]  — index 0 reserved for invalid


# ── Spectral clustering ────────────────────────────────────────────────────────

def symmetrize(edge_index, W, N):
    """
    Convert directed k-NN edges to a symmetric sparse matrix.
    W_sym[i,j] = W_sym[j,i] = max(W[i,j], W[j,i]) via scipy COO.
    """
    i = edge_index[0].cpu().numpy()
    j = edge_index[1].cpu().numpy()
    w = W.cpu().float().numpy()

    # Stack both directions; scipy will sum duplicates → we use max via two-pass
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    vals = np.concatenate([w, w])

    # Build with sum then clip to [0,1] — summing duplicates overestimates but
    # max(a,b) ≤ a+b; dividing by count of duplicates gives average.
    # Simpler: build two matrices and take elementwise max.
    W_ij = sp.csr_matrix((w, (i, j)), shape=(N, N))
    W_ji = sp.csr_matrix((w, (j, i)), shape=(N, N))
    W_sym = W_ij.maximum(W_ji)
    W_sym.eliminate_zeros()
    return W_sym


def normalized_laplacian(W_sym):
    """
    L_sym = I - D^{-1/2} W D^{-1/2}
    Returns the normalized affinity A_norm = D^{-1/2} W D^{-1/2}
    (eigenvectors of L_sym bottom-k  ↔  eigenvectors of A_norm top-k).
    """
    d = np.asarray(W_sym.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ W_sym @ D_inv_sqrt
    return A_norm


def _eigsh_arpack(A_norm, k):
    """CPU fallback: ARPACK via scipy."""
    print(f"  Computing top-{k} eigenvectors (ARPACK / CPU) ...")
    eigenvalues, eigenvectors = spla.eigsh(A_norm, k=k, which='LM')
    return eigenvalues, eigenvectors


def _eigsh_randomized(A_norm, k, n_iter=10, random_state=42):
    """
    Approximate top-k eigenvectors via randomized SVD (sklearn).
    No new dependencies. Runs on CPU in ~5-30 seconds for 687k nodes.

    For a symmetric PSD matrix A_norm:
      top-k singular vectors == top-k eigenvectors
      singular values         == eigenvalues (all ≥ 0)

    n_iter=10 gives high-quality approximation (sklearn default is 4;
    more iterations → better accuracy for clustered eigenvalues).
    """
    from sklearn.utils.extmath import randomized_svd
    print(f"  Computing top-{k} eigenvectors (randomized SVD / CPU) ...")
    # n_oversamples=20 (default 10) improves accuracy for clustered spectrum
    U, S, _ = randomized_svd(
        A_norm, n_components=k, n_iter=n_iter,
        n_oversamples=20, random_state=random_state
    )
    return S, U   # (eigenvalues, eigenvectors)


def _eigsh_lobpcg(A_norm, k, device='cuda'):
    """
    GPU solver: torch.lobpcg on a sparse-CSR matrix (no new dependencies).

    Memory footprint for 687k nodes, k=15:
      sparse matrix  ~112 MB  (14M edges × float32 × val+col)
      eigvec buffers ~ 40 MB  (687k × 15 × float32)
      LOBPCG work    ~160 MB  (≈4k buffer columns)
      ─────────────────────────────
      total          ~312 MB  (well within 32 GB V100)
    """
    print(f"  Computing top-{k} eigenvectors (torch.lobpcg / GPU) ...")
    N = A_norm.shape[0]

    # scipy CSR → torch sparse_csr on GPU
    M    = A_norm.tocsr().astype(np.float32)
    crow = torch.from_numpy(M.indptr.copy().astype(np.int64)).to(device)
    col  = torch.from_numpy(M.indices.copy().astype(np.int64)).to(device)
    val  = torch.from_numpy(M.data.copy().astype(np.float32)).to(device)
    A_t  = torch.sparse_csr_tensor(crow, col, val, size=(N, N), device=device)

    # Random initial guess (LOBPCG is sensitive to init; randn is fine)
    X0 = torch.randn(N, k, dtype=torch.float32, device=device)

    # largest=True → top-k eigenpairs of symmetric PSD A_norm
    eigenvalues_t, eigenvectors_t = torch.lobpcg(
        A_t, k=k, X=X0, largest=True, niter=1000, tol=1e-5
    )

    return eigenvalues_t.cpu().numpy(), eigenvectors_t.cpu().numpy()


def _eigsh_cupy(A_norm, k, maxiter=3000, tol=1e-3):
    """
    GPU solver: cupy drop-in for scipy eigsh (ARPACK on GPU).
    Requires: pip install cupy-cuda118   (match your CUDA version)

    maxiter/tol: safety nets to prevent infinite looping on near-flat spectra.
    With power sharpening applied upstream, 3000 iterations is generous.
    tol=1e-3 is coarser than default (1e-10) but sufficient for k-means input.
    """
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cpsla
    except ImportError:
        raise ImportError(
            "cupy not found. Install with:  pip install cupy-cuda118\n"
            "Or use --solver arpack (CPU)."
        )
    print(f"  Computing top-{k} eigenvectors (cupyx eigsh / GPU, maxiter={maxiter}, tol={tol}) ...")
    A_cp = cpsp.csr_matrix(A_norm.astype(np.float32))
    eigenvalues, eigenvectors = cpsla.eigsh(A_cp, k=k, which='LM',
                                            maxiter=maxiter, tol=tol)
    return eigenvalues, eigenvectors.get()   # move back to numpy


def spectral_embed(A_norm, n_clusters, eigengap_k=15, solver='lobpcg'):
    """
    Top-k eigenvectors of A_norm = bottom-k of L_sym.

    solver choices:
      'lobpcg'  — torch.lobpcg on GPU (default, no new deps)
      'cupy'    — cupyx.scipy eigsh on GPU (needs cupy-cuda118)
      'arpack'  — scipy ARPACK on CPU (original, slowest)

    Always computes eigengap_k eigenvectors (≥ n_clusters) so we can
    recommend the optimal k via the eigengap heuristic (Proposition 5).
    Returns [N, n_clusters] float32 array, row-normalised.
    """
    k_compute = max(n_clusters, eigengap_k)

    if solver == 'lobpcg':
        eigenvalues, eigenvectors = _eigsh_lobpcg(A_norm, k_compute)
    elif solver == 'cupy':
        eigenvalues, eigenvectors = _eigsh_cupy(A_norm, k_compute)
    elif solver == 'randomized':
        eigenvalues, eigenvectors = _eigsh_randomized(A_norm, k_compute)
    elif solver == 'arpack':
        eigenvalues, eigenvectors = _eigsh_arpack(A_norm, k_compute)
    else:
        raise ValueError(f"Unknown solver '{solver}'. Choose: lobpcg | cupy | randomized | arpack")

    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Eigengap heuristic: largest drop between consecutive eigenvalues
    gaps          = eigenvalues[:-1] - eigenvalues[1:]   # λ_i - λ_{i+1}
    suggested_k   = int(np.argmax(gaps)) + 1             # gap before index → k clusters
    print(f"  Eigenvalues (top-{k_compute}): {eigenvalues.round(4).tolist()}")
    print(f"  Eigengaps:                     {gaps.round(4).tolist()}")
    print(f"  Eigengap suggests k = {suggested_k}  (requested k = {n_clusters})")

    # Save eigengap plot
    _plot_eigengap(eigenvalues, gaps, suggested_k, n_clusters)

    # Row-normalise the first n_clusters eigenvectors for k-means
    embedding = normalize(eigenvectors[:, :n_clusters], norm='l2')
    return embedding.astype(np.float32)


def _plot_eigengap(eigenvalues, gaps, suggested_k, requested_k):
    """Saved to /tmp for quick inspection — path printed to stdout."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Eigengap Heuristic", fontsize=12)

    ks = np.arange(1, len(eigenvalues) + 1)
    ax1.plot(ks, eigenvalues, 'o-', markersize=4)
    ax1.axvline(suggested_k, color='red',    linestyle='--', label=f'suggested k={suggested_k}')
    ax1.axvline(requested_k, color='orange', linestyle='--', label=f'requested k={requested_k}')
    ax1.set_xlabel('k'); ax1.set_ylabel('Eigenvalue'); ax1.set_title('Eigenvalues')
    ax1.legend(fontsize=8)

    gap_ks = np.arange(1, len(gaps) + 1)
    ax2.bar(gap_ks, gaps, color='steelblue', alpha=0.8)
    ax2.axvline(suggested_k, color='red',    linestyle='--', label=f'suggested k={suggested_k}')
    ax2.axvline(requested_k, color='orange', linestyle='--', label=f'requested k={requested_k}')
    ax2.set_xlabel('k'); ax2.set_ylabel('Gap (λ_k − λ_{k+1})'); ax2.set_title('Eigengaps')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = '/tmp/eigengap.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Eigengap plot: {path}")


def run_kmeans(embedding, n_clusters, seed=0):
    print(f"  Running k-means (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto', max_iter=300)
    labels = km.fit_predict(embedding)
    counts = np.bincount(labels, minlength=n_clusters)
    print(f"  Cluster sizes: {sorted(counts.tolist(), reverse=True)}")
    return labels


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_labels(labels_full, out_dir, n_clusters):
    """Save full-N label array as .npy (invalid Gaussians → label 0)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"spectral_labels_k{n_clusters}.npy")
    np.save(path, labels_full)
    print(f"  Saved labels: {path}")
    return path


def annotate_ply(ply_path, out_ply_path, labels_full):
    """
    Read original .ply, append/overwrite the 'cls' property, write new .ply.
    labels_full: [N] int array aligned to all Gaussians in the .ply.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]
    data   = vertex.data

    # Rebuild as structured array with cls column
    old_names  = data.dtype.names
    new_dtype  = [(n, data.dtype[n]) for n in old_names if n != 'cls']
    new_dtype += [('cls', 'f4')]

    new_data = np.empty(len(data), dtype=new_dtype)
    for n in old_names:
        if n != 'cls':
            new_data[n] = data[n]
    new_data['cls'] = labels_full.astype(np.float32)

    new_vertex = PlyElement.describe(new_data, 'vertex')
    PlyData([new_vertex], text=False).write(out_ply_path)
    print(f"  Saved annotated .ply: {out_ply_path}")


def save_cluster_scatter(gaussians, valid, labels_full, out_dir, n_clusters, subsample=50000):
    """Quick 2-D scatter coloured by cluster label."""
    pos = gaussians.get_xyz[valid].cpu().float().numpy()
    labels = labels_full[valid.cpu().numpy()]

    N = pos.shape[0]
    if N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        pos    = pos[idx]
        labels = labels[idx]

    palette = (CLUSTER_PALETTE.numpy() * 255).astype(np.uint8)
    colors  = np.array([palette[l % len(palette)] for l in labels]) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Spectral Clusters (k={n_clusters})", fontsize=12)
    planes = [('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2)]
    for ax, (label, a, b) in zip(axes, planes):
        ax.scatter(pos[:, a], pos[:, b], c=colors, s=0.4, alpha=0.6)
        ax.set_title(label); ax.set_xlabel(label[0]); ax.set_ylabel(label[1])
        ax.set_aspect('equal')
    plt.tight_layout()
    path = os.path.join(out_dir, f"cluster_scatter_k{n_clusters}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved scatter: {path}")


# ── Rendering ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def render_clusters(gaussians, scene, deform, dataset, opt, pipe, labels_full, n_clusters, out_dir):
    labels_t = torch.from_numpy(labels_full).long().cuda()  # [N]
    palette  = CLUSTER_PALETTE.cuda()                        # [P, 3]
    cluster_colors = palette[labels_t % len(palette)]        # [N, 3]

    bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    views = scene.getTrainCameras()
    render_dir = os.path.join(out_dir, f"renders_k{n_clusters}")
    os.makedirs(render_dir, exist_ok=True)
    print(f"\nRendering {len(views)} views → {render_dir}")

    for view in tqdm(views, desc="Rendering"):
        fid        = view.fid
        xyz        = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        if opt.deform_type == 'DeformNetwork':
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        else:
            d_xyz, d_rotation, d_scaling = deform.step(
                xyz.detach(), time_input,
                gaussians.get_gaussian_features.squeeze(1)
            )

        result = render(view, gaussians, pipe, background,
                        d_xyz, d_rotation, d_scaling,
                        is_6dof=dataset.is_6dof,
                        override_color=cluster_colors)

        img_name = os.path.splitext(view.image_name)[0]
        torchvision.utils.save_image(
            result['render'].cpu(),
            os.path.join(render_dir, f"{img_name}.png")
        )


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main(dataset, opt, pipe, args):
    # ── 1. Load checkpoint ────────────────────────────────────────────────
    gaussians = GaussianModel(dataset.sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=args.load_iteration,
                      shuffle=False)

    deform      = DeformModel(is_blender=dataset.is_blender,
                              is_6dof=dataset.is_6dof,
                              model_type=opt.deform_type)
    deform_path = args.deform_path if args.deform_path else dataset.model_path
    deform.load_weights(deform_path, iteration=args.load_iteration)

    N_total = gaussians.get_xyz.shape[0]
    print(f"\nTotal Gaussians: {N_total:,}")

    # ── 2. Build affinity graph ───────────────────────────────────────────
    print(f"\nBuilding affinity graph (k={args.k})...")
    graph = AffinityGraph(
        gaussians,
        k=args.k,
        opacity_thresh=args.opacity_thresh,
        sigma_pos=args.sigma_pos,
        sigma_color=args.sigma_color,
        sigma_scale=args.sigma_scale,
        power=args.power,
    )
    edge_index, W, valid = graph.build(return_components=False)

    N_valid = valid.sum().item()
    print(f"Gaussians after opacity filter: {N_valid:,} / {N_total:,} "
          f"({100 * N_valid / N_total:.1f}%)")
    print(f"Edges: {W.shape[0]:,}")

    # ── 3. Symmetrize → normalized affinity matrix ────────────────────────
    print("\nBuilding symmetric normalized Laplacian...")
    W_sym  = symmetrize(edge_index, W, N_valid)
    A_norm = normalized_laplacian(W_sym)
    print(f"  Sparse matrix: {W_sym.shape}, nnz={W_sym.nnz:,}")

    # ── 4. Spectral embedding ─────────────────────────────────────────────
    embedding = spectral_embed(A_norm, args.n_clusters, solver=args.solver)  # [N_valid, k]

    # ── 5. K-means ───────────────────────────────────────────────────────
    labels_valid = run_kmeans(embedding, args.n_clusters)  # [N_valid]

    # Map back to all N Gaussians.
    # labels_valid is 0-indexed from KMeans → shift by +1 so valid clusters
    # occupy indices 1..k. Index 0 is reserved for filtered-out Gaussians,
    # which map to the black entry in CLUSTER_PALETTE and are invisible in renders.
    labels_full = np.zeros(N_total, dtype=np.int32)
    valid_np    = valid.cpu().numpy()
    labels_full[valid_np] = labels_valid + 1

    # ── 6. Save outputs ───────────────────────────────────────────────────
    out_dir = os.path.join(dataset.model_path, "spectral")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving outputs to: {out_dir}")

    save_labels(labels_full, out_dir, args.n_clusters)
    save_cluster_scatter(gaussians, valid, labels_full, out_dir, args.n_clusters)

    # Annotate the original .ply with cluster ids
    ply_in  = os.path.join(dataset.model_path, "point_cloud",
                           f"iteration_{args.load_iteration}", "point_cloud.ply")
    ply_out = os.path.join(out_dir,
                           f"point_cloud_spectral_k{args.n_clusters}.ply")
    if os.path.exists(ply_in):
        annotate_ply(ply_in, ply_out, labels_full)
    else:
        print(f"  [WARN] .ply not found at {ply_in}, skipping annotation")

    # ── 7. Render ─────────────────────────────────────────────────────────
    if not args.no_render:
        render_clusters(gaussians, scene, deform, dataset, opt, pipe,
                        labels_full, args.n_clusters, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--load_iteration",    type=int,   default=20000)
    parser.add_argument("--deform_path",       type=str,   default="")
    parser.add_argument("--n_clusters",        type=int,   default=5)
    parser.add_argument("--k",                 type=int,   default=20)
    parser.add_argument("--opacity_thresh",    type=float, default=0.05)
    parser.add_argument("--sigma_pos",         type=float, default=0.0036)
    parser.add_argument("--sigma_color",       type=float, default=0.5160)
    parser.add_argument("--sigma_scale",       type=float, default=1.0)
    parser.add_argument("--power",             type=float, default=1.0,
                        help="Sharpening exponent on W (p>1 boosts eigengap; try 4 or 8)")
    parser.add_argument("--solver",             type=str,   default="lobpcg",
                        choices=["lobpcg", "cupy", "randomized", "arpack"],
                        help="Eigensolver: lobpcg=GPU/no-new-deps (default), "
                             "cupy=GPU/needs cupy-cuda118, arpack=CPU/scipy")
    parser.add_argument("--no_render",         action="store_true",
                        help="Skip rendering, only save labels and scatter")

    args = parser.parse_args(sys.argv[1:])
    safe_state(False)

    main(lp.extract(args), op.extract(args), pp.extract(args), args)
