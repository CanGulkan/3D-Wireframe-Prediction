import os
import random
import numpy as np
from scipy.spatial import cKDTree
from dataset import normalize_wireframe

# ─── Deterministic Setup ──────────────────────────────────────────────────────
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── I/O & Normalization ──────────────────────────────────────────────────────
def load_wireframe_obj(path: str):
    """
    Parse a Wavefront .obj file, normalize vertices and edge indices,
    and return normalized edges as an (M, 2, 3) array.
    """
    verts, edges = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'l':
                idxs = [int(p) - 1 for p in parts[1:]]
                for a, b in zip(idxs, idxs[1:]):
                    edges.append((a, b))

    verts = np.array(verts, dtype=np.float32)
    edges_idx = np.array(edges, dtype=int)

    # center & scale to unit sphere
    verts_norm, edges_idx_norm = normalize_wireframe(verts, edges_idx)
    return verts_norm[edges_idx_norm]  # (M, 2, 3)

# ─── Sampling-Based RMS ────────────────────────────────────────────────────────
def sample_wireframe_edges(edges, samples_per_edge=20):
    """
    edges: (M, 2, 3)
    returns: (M * samples_per_edge, 3) uniformly sampled points along all edges.
    """
    t = np.linspace(0, 1, samples_per_edge).reshape(-1, 1)
    samples = []
    for v0, v1 in edges:
        samples.append(v0 + t * (v1 - v0))
    return np.vstack(samples)

def rms_distance_sampling(point_cloud, edges, samples_per_edge=20):
    """
    Compute RMS distance by sampling points on edges and querying nearest neighbor.
    """
    pts = sample_wireframe_edges(edges, samples_per_edge)
    tree = cKDTree(pts)
    dists, _ = tree.query(point_cloud, k=1)
    return np.sqrt(np.mean(dists**2))

# ─── Analytic (Exact) RMS ────────────────────────────────────────────────────

def rms_distance_exact(pred_edges: np.ndarray, gt_edges: np.ndarray) -> float:
    """
    Compute the exact RMS distance from every point on each predicted edge
    (we treat the edge-endpoints as a point cloud) to the nearest
    ground-truth line segment.
    
    pred_edges: (E_pred, 2, 3)
    gt_edges:   (E_gt,   2, 3)
    """
    # 1) Flatten predicted-edge endpoints into a point set P of shape (N,3)
    P = pred_edges.reshape(-1, 3)  # N = E_pred * 2

    # 2) Unpack ground-truth segments A→B
    A = gt_edges[:, 0, :]  # (E_gt, 3)
    B = gt_edges[:, 1, :]  # (E_gt, 3)
    AB = B - A             # (E_gt, 3)

    # 3) Broadcast so we can compute all point-to-segment distances in one shot
    P_exp = P[:,   None, :]  # (N, 1, 3)
    A_exp = A[None, :, :]    # (1, E_gt, 3)
    AB_exp = AB[None, :, :]  # (1, E_gt, 3)

    # 4) Projection parameter t = ((P–A)·AB) / (AB·AB)
    numer  = np.sum((P_exp - A_exp) * AB_exp, axis=-1)  # (N, E_gt)
    denom  = np.sum(AB_exp * AB_exp,     axis=-1)       # (1, E_gt)
    t      = np.clip(numer / denom, 0.0, 1.0)          # (N, E_gt)

    # 5) Closest point on each segment & squared distances
    proj   = A_exp + t[..., None] * AB_exp  # (N, E_gt, 3)
    d2_all = np.sum((P_exp - proj)**2, axis=-1)  # (N, E_gt)

    # 6) For each P_i, take the min over segments
    min_d2 = np.min(d2_all, axis=1)  # (N,)

    # 7) RMS = sqrt(mean(d²))
    return float(np.sqrt(np.mean(min_d2)))

# ─── Main Script ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load and normalize point cloud
    P_full = np.loadtxt("1.xyz")
    if P_full.ndim != 2 or P_full.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in 1.xyz, got {P_full.shape}")
    P = P_full[:, :3]

    # Apply the same normalization as the wireframe
    # (assuming normalize_wireframe returns both verts_norm and scale info;
    # if not, you may need to extract centroid & scale from normalize_wireframe)
    # Here we assume P is already in the same normalized space.

    print("Point-cloud shape:", P.shape)

    # 2) Load normalized wireframe edges
    E = load_wireframe_obj("1.obj")
    print("Wireframe-edges shape:", E.shape)

    # 3) Compute RMS via sampling
    rms_samp = rms_distance_sampling(P, E, samples_per_edge=50)
    print(f"RMS distance (sampling, 50 pts/edge): {rms_samp:.8f}")

    # 4) Compute exact RMS via analytic distance
    rms_ex = rms_distance_exact(P, E)
    print(f"RMS distance (exact analytic):     {rms_ex:.8f}")
