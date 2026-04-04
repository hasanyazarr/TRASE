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
# Rendering coherence loss
# ---------------------------------------------------------------------------

def _image_gradient(x):
    """
    x: [C, H, W]
    returns: [H, W] mean gradient magnitude across channels
    """
    dx = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=0)   # [H, W-1]
    dy = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=0)   # [H-1, W]
    dx = F.pad(dx, (0, 1))          # [H, W]
    dy = F.pad(dy, (0, 0, 0, 1))   # [H, W]
    return (dx + dy) / 2


def rendering_coherence_loss(feat_map, rgb_map, depth_map=None,
                              alpha=1.0, beta=0.5,
                              edge_percentile=0.7,
                              num_pairs=2048, margin=0.3):
    """
    2D boundary-aware contrastive loss on rendered feature maps.

    Negative pairs: pixel straddling a detected RGB/depth edge → push features apart.
    Positive pairs: nearby pixels both in smooth region → pull features together.

    The key signal here is the negative pairs — boundary pixels must have different
    features. Positive pairs complement spatial coherence loss in 2D.

    Args:
        feat_map:        [C, H, W]  rendered feature map (differentiable)
        rgb_map:         [3, H, W]  rendered RGB          (detached)
        depth_map:       [1, H, W]  rendered depth        (detached, optional)
        alpha:           weight for RGB gradient in edge detection
        beta:            weight for depth gradient in edge detection
        edge_percentile: pixels above this quantile of edge strength are edges
                         (e.g. 0.7 → top 30% = boundaries, bottom 70% = smooth)
        num_pairs:       max number of pairs per type (neg / pos)
        margin:          hinge margin for negative pairs
    """
    C, H, W = feat_map.shape
    device = feat_map.device

    # --- Edge map [H, W] ---
    edge = alpha * _image_gradient(rgb_map.detach())
    if depth_map is not None:
        edge = edge + beta * _image_gradient(depth_map.detach())

    # Percentile threshold — robust to scene scale variation
    threshold = torch.quantile(edge.reshape(-1), edge_percentile)
    edge_mask = edge > threshold   # [H, W] bool: True = boundary

    # --- Per-pixel features: [H*W, C], L2-normalized ---
    feat_flat = feat_map.permute(1, 2, 0).reshape(-1, C)
    feat_flat = F.normalize(feat_flat, dim=-1)

    def flat_idx(y, x):
        return y * W + x

    loss = torch.tensor(0.0, device=device)
    n_terms = 0

    # -------------------------------------------------------------------
    # Negative pairs: edge pixel ↔ smooth 4-connected neighbor
    # Each such pair straddles a boundary → features should differ.
    # -------------------------------------------------------------------
    edge_pixels = edge_mask.nonzero()   # [E, 2]  (y, x)
    if edge_pixels.shape[0] > 0:
        perm = torch.randperm(edge_pixels.shape[0], device=device)[:num_pairs]
        ep = edge_pixels[perm]          # [S, 2]

        neg_i_list, neg_j_list = [], []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny = ep[:, 0] + dy
            nx = ep[:, 1] + dx
            in_bounds = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            ny_v = ny[in_bounds]
            nx_v = nx[in_bounds]
            ep_v  = ep[in_bounds]
            # Neighbor must be in the smooth region (non-edge)
            is_smooth = ~edge_mask[ny_v, nx_v]
            if is_smooth.sum() > 0:
                neg_i_list.append(flat_idx(ep_v[is_smooth, 0], ep_v[is_smooth, 1]))
                neg_j_list.append(flat_idx(ny_v[is_smooth], nx_v[is_smooth]))

        if neg_i_list:
            ni = torch.cat(neg_i_list)[:num_pairs]
            nj = torch.cat(neg_j_list)[:num_pairs]
            sim_neg = (feat_flat[ni] * feat_flat[nj]).sum(dim=-1)
            loss_neg = F.relu(sim_neg - margin).mean()
            loss = loss + loss_neg
            n_terms += 1

    # -------------------------------------------------------------------
    # Positive pairs: smooth pixel ↔ nearby smooth pixel (±2 px offset)
    # Reinforces spatial coherence in 2D image space.
    # -------------------------------------------------------------------
    smooth_pixels = (~edge_mask).nonzero()   # [S, 2]
    if smooth_pixels.shape[0] >= 2:
        perm = torch.randperm(smooth_pixels.shape[0], device=device)[:num_pairs]
        sp = smooth_pixels[perm]    # [S, 2]

        offsets = torch.randint(-2, 3, (sp.shape[0], 2), device=device)
        ny = (sp[:, 0] + offsets[:, 0]).clamp(0, H - 1)
        nx = (sp[:, 1] + offsets[:, 1]).clamp(0, W - 1)

        # Neighbor must also be smooth
        is_smooth_j = ~edge_mask[ny, nx]
        if is_smooth_j.sum() > 0:
            pi = flat_idx(sp[is_smooth_j, 0], sp[is_smooth_j, 1])
            pj = flat_idx(ny[is_smooth_j], nx[is_smooth_j])
            sim_pos = (feat_flat[pi] * feat_flat[pj]).sum(dim=-1)
            loss = loss + (1.0 - sim_pos).mean()
            n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss / n_terms


# ---------------------------------------------------------------------------
# Spatial coherence loss
# ---------------------------------------------------------------------------

def spatial_coherence_loss(f, positions, num_pairs, spatial_radius, sigma=None):
    """
    Spatially close Gaussians should have similar features.

    w_ij = exp(-d_ij² / 2σ²)  — closer pairs get higher weight.
    Loss  = Σ w_ij · (1 - cos_sim(f_i, f_j)) / Σ w_ij

    Pure pull loss — relies on motion loss far-pair negatives to prevent collapse.

    Args:
        f:              [N, 32]  L2-normalized feature vectors
        positions:      [N, 3]   canonical 3D positions
        num_pairs:      number of near pairs to sample
        spatial_radius: max 3D distance between pairs
        sigma:          Gaussian bandwidth (default: spatial_radius / 2)
    """
    N = f.shape[0]
    device = f.device

    if sigma is None:
        sigma = spatial_radius * 0.5

    idx_i = torch.randint(0, N, (num_pairs * 4,), device=device)
    idx_j = torch.randint(0, N, (num_pairs * 4,), device=device)
    valid = (idx_i != idx_j)
    dist = torch.norm(positions[idx_i] - positions[idx_j], dim=-1)
    near_mask = valid & (dist < spatial_radius)

    idx_i = idx_i[near_mask][:num_pairs]
    idx_j = idx_j[near_mask][:num_pairs]

    if idx_i.shape[0] < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    d = torch.norm(positions[idx_i] - positions[idx_j], dim=-1)
    w = torch.exp(-d ** 2 / (2 * sigma ** 2))

    sim = (f[idx_i] * f[idx_j]).sum(dim=-1)
    loss = (w * (1.0 - sim)).sum() / w.sum().clamp(min=1e-6)

    return loss


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
