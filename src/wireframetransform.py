# wireframetransform.py

import numpy as np
import torch
from torch.utils.data import Dataset

class WireframeTransform:
    """Apply random augmentations and then
       a two-phase corner+edge prediction on the PC."""
    def __init__(
        self,
        noise_std: float = 0.01,
        max_rotation: float = 15,
        min_scale: float = 0.9,
        max_scale: float = 1.1,
        num_corners: int = 50,
        k_neighbors: int = 4
    ):
        # augmentation params
        self.noise_std      = noise_std
        self.max_rotation   = max_rotation
        self.scale_range    = (min_scale, max_scale)

        # corner->edge params
        self.num_corners    = num_corners
        self.k_neighbors    = k_neighbors

    def __call__(self, pc: torch.Tensor, gt_edges: torch.Tensor):
        # --- existing augmentations ---
        angle   = np.deg2rad(np.random.uniform(-self.max_rotation, self.max_rotation))
        rot_mat = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ], dtype=pc.dtype, device=pc.device)
        scale   = np.random.uniform(*self.scale_range)

        pc      = (pc @ rot_mat) * scale
        gt_edges= (gt_edges @ rot_mat) * scale
        if self.noise_std > 0:
            pc += torch.randn_like(pc) * self.noise_std

        # --- Phase 1: predict 'corners' via farthest point sampling ---
        pred_corners = self._farthest_point_sampling(pc, self.num_corners)

        # --- Phase 2: connect each corner to its k nearest neighbour corners ---
        pred_edges   = self._connect_corners(pred_corners, self.k_neighbors)

        # return augmented PC, GT edges, plus our new preds
        return pc, gt_edges, pred_corners, pred_edges

    def _farthest_point_sampling(self, pc: torch.Tensor, k: int) -> torch.Tensor:
        """Iterative farthest point sampling on a (N,3) tensor → (k,3)."""
        pts = pc.detach().cpu().numpy()
        N   = pts.shape[0]
        centroids = np.zeros((k,), dtype=int)
        dists     = np.full((N,), np.inf)

        farthest = np.random.randint(0, N)
        for i in range(k):
            centroids[i] = farthest
            centroid_pt  = pts[farthest]
            dist_to_new  = np.sum((pts - centroid_pt) ** 2, axis=1)
            dists        = np.minimum(dists, dist_to_new)
            farthest     = np.argmax(dists)
        return torch.from_numpy(pts[centroids]).to(pc.device).float()

    def _connect_corners(self, corners: torch.Tensor, k: int) -> torch.Tensor:
        """Connect each corner to its k nearest neighbours → Tensor of shape (k * C, 2, 3)."""
        # compute pairwise distances
        dmat = torch.cdist(corners, corners)  # (C, C)
        edges = []
        for i in range(corners.size(0)):
            # get top-(k+1) (includes self), then skip index 0
            idxs = torch.topk(dmat[i], k+1, largest=False).indices.tolist()[1:]
            for j in idxs:
                edges.append(torch.stack([corners[i], corners[j]], dim=0))
        return torch.stack(edges, dim=0)
