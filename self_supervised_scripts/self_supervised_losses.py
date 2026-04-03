import torch
import torch.nn.functional as F

try:
    import pytorch3d.ops as pt3d_ops
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _compute_rigid_score(traj_i, traj_j):
    """
    Args:
        traj_i, traj_j: [P, T, 3]  deformation offsets for each pair
    Returns:
        rigid_score: [P]  in [0, 1], 1 = perfectly rigid (same object)
    """
    rel_motion_sq = torch.sum((traj_i - traj_j) ** 2, dim=-1)  # [P, T]
    motion_var = rel_motion_sq.var(dim=-1)                      # [P]
    sigma = motion_var.median().clamp(min=1e-6)
    return torch.exp(-motion_var / sigma)


def _contrastive_loss(sim, pos_mask, neg_mask, margin, device):
    """
    Args:
        sim:      [P]  cosine similarity for each pair
        pos_mask: [P]  bool, positive pairs
        neg_mask: [P]  bool, negative pairs
    Returns:
        scalar loss
    """
    loss = torch.tensor(0.0, device=device)
    n_terms = 0

    if pos_mask.sum() > 0:
        loss = loss + (1.0 - sim[pos_mask]).mean()
        n_terms += 1

    if neg_mask.sum() > 0:
        loss = loss + F.relu(sim[neg_mask] - margin).mean()
        n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss / n_terms


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _motion_loss_random(f, trajectories, positions, num_pairs, spatial_radius,
                        tau_pos, tau_neg, margin):
    """
    Random pair sampling with guaranteed negatives.

    Near pairs  (dist < spatial_radius): use rigid_score to decide pos/neg.
    Far pairs   (dist > spatial_radius): guaranteed negatives — objects far
                apart are almost certainly different, regardless of motion.
    Half of num_pairs budget goes to near pairs, half to far pairs.
    """
    N = f.shape[0]
    device = f.device
    half = num_pairs * 2  # half budget (will be trimmed to num_pairs//2 each)

    # --- Near pairs: rigid score decides ---
    idx_i = torch.randint(0, N, (num_pairs * 4,), device=device)
    idx_j = torch.randint(0, N, (num_pairs * 4,), device=device)
    valid = (idx_i != idx_j)
    dist = torch.norm(positions[idx_i] - positions[idx_j], dim=-1)
    near_mask = valid & (dist < spatial_radius)
    near_i = idx_i[near_mask][:half]
    near_j = idx_j[near_mask][:half]

    # --- Far pairs: guaranteed negatives ---
    idx_i2 = torch.randint(0, N, (num_pairs * 4,), device=device)
    idx_j2 = torch.randint(0, N, (num_pairs * 4,), device=device)
    valid2 = (idx_i2 != idx_j2)
    dist2 = torch.norm(positions[idx_i2] - positions[idx_j2], dim=-1)
    far_mask = valid2 & (dist2 > spatial_radius)
    far_i = idx_i2[far_mask][:half]
    far_j = idx_j2[far_mask][:half]

    if near_i.shape[0] < 2 and far_i.shape[0] < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = torch.tensor(0.0, device=device)
    n_terms = 0

    # Near pairs
    if near_i.shape[0] >= 2:
        rigid_score = _compute_rigid_score(trajectories[near_i], trajectories[near_j])
        sim_near = (f[near_i] * f[near_j]).sum(dim=-1)
        near_loss = _contrastive_loss(sim_near, rigid_score > tau_pos,
                                      rigid_score < tau_neg, margin, device)
        loss = loss + near_loss
        n_terms += 1

    # Far pairs — always negative
    if far_i.shape[0] >= 2:
        sim_far = (f[far_i] * f[far_j]).sum(dim=-1)
        far_loss = F.relu(sim_far).mean()
        loss = loss + far_loss
        n_terms += 1

    return loss / n_terms


def _motion_loss_knn(f, trajectories, positions, K, tau_pos, tau_neg, margin):
    """
    KNN-based pair sampling.
    For each Gaussian, finds K nearest neighbors in 3D.
    Positives come from close neighbors, semi-hard negatives from farther ones.
    Avoids trivial negatives and mode collapse.
    """
    if not PYTORCH3D_AVAILABLE:
        raise RuntimeError("pytorch3d is required for knn mode but could not be imported.")

    device = f.device
    N = f.shape[0]

    # KNN in 3D: [1, N, K] indices
    knn_result = pt3d_ops.knn_points(
        positions.unsqueeze(0),
        positions.unsqueeze(0),
        K=K + 1,  # +1 because the point itself is included
    )
    knn_idx = knn_result.idx.squeeze(0)  # [N, K+1]
    knn_idx = knn_idx[:, 1:]             # [N, K]  remove self

    # Build pairs: each Gaussian i paired with all K neighbors
    idx_i = torch.arange(N, device=device).unsqueeze(1).expand(N, K).reshape(-1)  # [N*K]
    idx_j = knn_idx.reshape(-1)                                                     # [N*K]

    rigid_score = _compute_rigid_score(trajectories[idx_i], trajectories[idx_j])
    sim = (f[idx_i] * f[idx_j]).sum(dim=-1)

    return _contrastive_loss(sim, rigid_score > tau_pos, rigid_score < tau_neg,
                             margin, device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MOTION_SAMPLING_MODES = ['random', 'knn']


def motion_affinity_loss(
    f,
    trajectories,
    positions,
    mode='random',
    # random mode
    num_pairs=4096,
    spatial_radius=0.5,
    # knn mode
    K=64,
    # shared
    tau_pos=0.8,
    tau_neg=0.2,
    margin=0.5,
):
    """
    Motion affinity loss: Gaussians that move together should have similar features.

    Args:
        f:             [N, 32]    L2-normalized feature vectors
        trajectories:  [N, T, 3]  deformation offsets at T timestamps
        positions:     [N, 3]     canonical 3D positions
        mode:          'random' or 'knn'
        num_pairs:     (random mode) number of pairs to sample
        spatial_radius:(random mode) max 3D distance between pairs
        K:             (knn mode) number of neighbors per Gaussian
        tau_pos:       rigid_score threshold → positive pair
        tau_neg:       rigid_score threshold → negative pair
        margin:        hinge margin for negative pairs

    Returns:
        scalar loss
    """
    if mode == 'random':
        return _motion_loss_random(f, trajectories, positions, num_pairs,
                                   spatial_radius, tau_pos, tau_neg, margin)
    elif mode == 'knn':
        return _motion_loss_knn(f, trajectories, positions, K,
                                tau_pos, tau_neg, margin)
    else:
        raise ValueError(f"Unknown motion sampling mode: '{mode}'. "
                         f"Choose from {MOTION_SAMPLING_MODES}")
