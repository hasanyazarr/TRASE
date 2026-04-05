import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch_cluster import knn as torch_knn

from utils.general_utils import build_rotation


class AffinityGraph:
    """
    Builds a sparse Gaussian affinity graph from frozen 4DGS parameters.

    Ageo-only mode (Phase 1):
        W(i,j) = Apos · Acolor · Aorient · Ascale

    No rendering, no deformation MLP queries, no training required.
    Amotion and boundary suppression B are added in Phase 3.
    """

    def __init__(
        self,
        gaussians,
        k: int = 20,
        opacity_thresh: float = 0.1,
        sigma_pos: float = 0.1,
        sigma_color: float = 0.3,
        sigma_scale: float = 1.0,
    ):
        """
        Args:
            gaussians:      GaussianModel (Stage 1 checkpoint, frozen)
            k:              number of nearest neighbors per Gaussian
            opacity_thresh: Gaussians below this opacity are excluded
            sigma_pos:      bandwidth for spatial proximity kernel
            sigma_color:    bandwidth for color similarity kernel
            sigma_scale:    bandwidth for scale ratio kernel
        """
        self.gaussians = gaussians
        self.k = k
        self.opacity_thresh = opacity_thresh
        self.sigma_pos = sigma_pos
        self.sigma_color = sigma_color
        self.sigma_scale = sigma_scale

    @torch.no_grad()
    def build(self, return_components=False):
        """
        Build the affinity graph.

        Args:
            return_components: if True, also return dict of individual A_x tensors

        Returns:
            edge_index:  [2, E] LongTensor  — (source, target) index pairs
            weights:     [E]    FloatTensor — W(i,j) = Ageo per edge
            valid_mask:  [N]    BoolTensor  — which original Gaussians passed opacity filter
            components:  dict with keys Apos, Acolor, Aorient, Ascale  (only if return_components=True)
        """
        g = self.gaussians

        # ── 1. Opacity filter ─────────────────────────────────────────────
        valid = g.get_opacity.squeeze(1) > self.opacity_thresh      # [N]
        pos   = g.get_xyz[valid]                                    # [N', 3]
        scale = g.get_scaling[valid]                                # [N', 3]  exp already applied
        color = g._features_dc[valid].squeeze(1)                   # [N', 3]  DC SH component
        rot_q = g.get_rotation[valid]                               # [N', 4]  normalized quaternions

        N = pos.shape[0]
        device = pos.device

        # ── 2. Principal axis from rotation + scale ───────────────────────
        # Gaussian covariance: Σ = R · diag(s²) · R^T
        # Eigenvectors = columns of R, eigenvalues = s²
        # Principal axis = column of R with largest scale value
        R     = build_rotation(rot_q)                               # [N', 3, 3]
        max_s = scale.argmax(dim=1)                                 # [N']
        v1    = R[torch.arange(N, device=device), :, max_s]        # [N', 3]
        v1    = F.normalize(v1, dim=1)

        # ── 3. k-NN graph in canonical space (GPU) ────────────────────────
        # torch_cluster.knn(x, y, k): for each point in y find k nearest in x
        # k+1 to account for self-loops, which are removed below
        edge_index = torch_knn(pos, pos, k=self.k + 1)             # [2, N'*(k+1)]
        self_loop  = edge_index[0] == edge_index[1]
        edge_index = edge_index[:, ~self_loop]                      # [2, E]
        i, j = edge_index[0], edge_index[1]                        # E each

        # ── 4. Affinity components ────────────────────────────────────────

        # Apos: spatial proximity
        Apos = torch.exp(
            -((pos[i] - pos[j]) ** 2).sum(dim=1)
            / (2 * self.sigma_pos ** 2)
        )

        # Acolor: DC SH appearance similarity
        Acolor = torch.exp(
            -((color[i] - color[j]) ** 2).sum(dim=1)
            / (2 * self.sigma_color ** 2)
        )

        # Aorient: principal axis alignment — key signal for static scenes
        # |v_i · v_j| = 1 if axes aligned, 0 if perpendicular
        Aorient = (v1[i] * v1[j]).sum(dim=1).abs()

        # Ascale: Gaussian size similarity via log scale ratio
        norm_i = scale[i].norm(dim=1).clamp(min=1e-6)
        norm_j = scale[j].norm(dim=1).clamp(min=1e-6)
        Ascale = torch.exp(
            -(norm_i / norm_j).log() ** 2
            / (2 * self.sigma_scale ** 2)
        )

        # ── 5. Final weight (geometric mean) ─────────────────────────────
        # Geometric mean instead of product to avoid collapse when any
        # single component is low — each term contributes equally.
        W = (Apos * Acolor * Aorient * Ascale) ** 0.25              # [E]

        if return_components:
            components = {
                'Apos':    Apos,
                'Acolor':  Acolor,
                'Aorient': Aorient,
                'Ascale':  Ascale,
            }
            return edge_index, W, valid, components

        return edge_index, W, valid
