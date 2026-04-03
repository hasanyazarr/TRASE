import torch


class GaussianDescriptor:
    """
    Builds a descriptor vector h_i for each Gaussian by concatenating
    static properties (position, scale, color) and motion trajectory
    sampled from the frozen deformation MLP.

    Output:
        h:           [N, 9 + T*3]  normalized descriptor
        trajectories:[N, T, 3]     raw deformation offsets (for motion loss)
    """

    def __init__(self, gaussians, deform, T=8, batch_size=10000):
        """
        Args:
            gaussians:  GaussianModel  (Stage 1 checkpoint, frozen)
            deform:     DeformModel    (Stage 1 checkpoint, frozen)
            T:          number of evenly spaced timestamps to sample
            batch_size: number of Gaussians to process per deform.step call
        """
        self.gaussians = gaussians
        self.deform = deform
        self.T = T
        self.batch_size = batch_size

    @torch.no_grad()
    def build(self):
        """
        Returns:
            h:            [N, 9 + T*3]  normalized descriptor
            trajectories: [N, T, 3]    raw deformation offsets
        """
        gaussians = self.gaussians

        # --- 1. Static properties ---
        pos   = gaussians._xyz.detach()                              # [N, 3]
        scale = gaussians._scaling.detach()                          # [N, 3]  log scale
        color = gaussians._features_dc.detach().squeeze(1)          # [N, 3]  DC component

        N = pos.shape[0]
        device = pos.device

        # --- 2. Trajectory from deformation MLP ---
        timestamps = torch.linspace(0, 1, self.T, device=device)    # [T]
        trajectories = torch.zeros(N, self.T, 3, device=device)     # [N, T, 3]

        for t_idx, t_val in enumerate(timestamps):
            time_input = torch.full((N, 1), t_val.item(), device=device)

            # Batch to avoid OOM on large scenes
            d_xyz_parts = []
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                d_xyz, _, _ = self.deform.step(pos[start:end], time_input[start:end])
                d_xyz_parts.append(d_xyz)

            trajectories[:, t_idx, :] = torch.cat(d_xyz_parts, dim=0)

        traj_flat = trajectories.reshape(N, self.T * 3)             # [N, T*3]

        # --- 3. Normalize each component independently ---
        pos_n   = self._normalize(pos)
        scale_n = self._normalize(scale)
        color_n = self._normalize(color)
        traj_n  = self._normalize(traj_flat)

        # --- 4. Concatenate ---
        h = torch.cat([pos_n, scale_n, color_n, traj_n], dim=-1)   # [N, 9 + T*3]

        return h, trajectories

    @staticmethod
    def _normalize(x):
        """Zero mean, unit std across N dimension. x: [N, D]"""
        mean = x.mean(dim=0, keepdim=True)
        std  = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std
