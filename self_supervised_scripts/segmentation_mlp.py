import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationMLP(nn.Module):
    """
    Maps Gaussian descriptor h_i to a 32-dim feature f_i on the unit sphere.

    Architecture:
        Linear(dim_in → 256) + ReLU
        Linear(256 → 128)    + ReLU
        Linear(128 → 32)
        L2-normalize

    The L2 normalization puts all features on the unit sphere, making
    cosine similarity equivalent to dot product — required for the
    contrastive losses in self_supervised_losses.py.
    """

    def __init__(self, dim_in, dim_out=32):
        """
        Args:
            dim_in:  input descriptor dimension (9 + T*3, e.g. 33 for T=8)
            dim_out: output feature dimension (32 to match TRASE)
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, dim_out),
        )

    def forward(self, h):
        """
        Args:
            h: [N, dim_in]  normalized Gaussian descriptors
        Returns:
            f: [N, dim_out] L2-normalized feature vectors
        """
        f = self.net(h)
        return F.normalize(f, p=2, dim=-1)
