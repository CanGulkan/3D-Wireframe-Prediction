# ─── Import Libraries ───────────────────────────────────────────────────────
import os
import json
import time
import torch                            # PyTorch core library for tensor operations
import torch.nn.functional as F         # Functional interface: activations, loss functions, etc.
from torch.utils.data import Dataset, DataLoader  # For creating datasets and batching
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm for optimal matching
import numpy as np                      # NumPy for array operations and I/O

# ─── Part 1: Utility Functions ──────────────────────────────────────────────

def sample_edge_pts(verts, num_samples=10):
    """
    Uniformly sample points along an edge in 3D space.
    verts: Tensor shape [2,3] — two end-points of an edge.
    num_samples: number of samples (including endpoints).
    returns: Tensor shape [num_samples, 3]
    """
    t = torch.linspace(0, 1, num_samples, device=verts.device).unsqueeze(1)
    return verts[0][None] * (1 - t) + verts[1][None] * t  # Linear interpolation

def directed_hausdorff(A, B):
    """
    Compute the directed Hausdorff distance from set A to B.
    A, B: Tensors of shape [S, 3] — sampled edge points.
    Returns: maximum of minimum distances from A to B.
    """
    dists = torch.cdist(A, B)  # Pairwise distance matrix [S, S]
    return torch.max(torch.min(dists, dim=1).values)

def edge_similarity(e_pred, e_gt, alpha=1.0, beta=1.0, gamma=1.0, samples=10):
    """
    Compute dissimilarity between predicted and ground-truth edge using:
    1. Symmetric Hausdorff distance (geometry)
    2. Directional cosine similarity (angle)
    3. Relative length difference
    """
    A = sample_edge_pts(e_pred, samples)
    B = sample_edge_pts(e_gt, samples)

    Hd = max(directed_hausdorff(A, B), directed_hausdorff(B, A))  # Symmetric

    v_p = e_pred[1] - e_pred[0]  # Vector of predicted edge
    v_g = e_gt[1] - e_gt[0]      # Vector of ground truth edge

    cos_sim = torch.dot(v_p, v_g) / (v_p.norm() * v_g.norm() + 1e-8)  # Cosine similarity
    Dir_sim = 1 - torch.abs(cos_sim)  # Angle difference (0 = aligned)

    lp, lg = v_p.norm(), v_g.norm()  # Edge lengths
    Len_sim = 1 - (torch.min(lp, lg) / torch.max(lp, lg))  # Relative length diff

    return alpha * Hd + beta * Dir_sim + gamma * Len_sim  # Weighted sum

# ─── Part 2: Loss Function ──────────────────────────────────────────────────

def wireframe_loss(pred_edges, gt_edges, p_conf, p_quad_logits, *,
                   alpha=1.0, beta=1.0, gamma=1.0,
                   λ_mid=1e-4, λ_comp=1e-4, λ_con=1.0, λ_quad=1.0, λ_sim=1.0,
                   num_samples=20):
    """
    Compute composite loss between predicted edges and ground-truth.
    Includes Hungarian matching, midpoint/component errors, confidence, quadrant, and similarity.
    """
    E_pred, E_gt = pred_edges.shape[0], gt_edges.shape[0]

    # 1) Compute cost matrix [E_pred, E_gt]
    C = torch.zeros(E_pred, E_gt, device=pred_edges.device)
    for i in range(E_pred):
        for j in range(E_gt):
            C[i, j] = edge_similarity(pred_edges[i], gt_edges[j], alpha, beta, gamma, samples=num_samples)

    # 2) Match using Hungarian algorithm
    rows, cols = linear_sum_assignment(C.detach().cpu().numpy())
    matches = list(zip(rows, cols))

    # 3) Midpoint and component-wise L1 losses
    mid, comp = 0.0, 0.0
    for i, j in matches:
        p_mid = pred_edges[i].mean(dim=0)
        g_mid = gt_edges[j].mean(dim=0)
        mid += (p_mid - g_mid).abs().sum()

        v_p = pred_edges[i][1] - pred_edges[i][0]
        v_g = gt_edges[j][1] - gt_edges[j][0]
        comp += (v_p - v_g).abs().sum()

    mid_loss = mid / max(len(matches), 1)
    comp_loss = comp / max(len(matches), 1)

    # 4) Confidence loss — target confidence = 1 - dissimilarity
    g_con = torch.zeros(E_pred, device=C.device)
    for i, j in matches:
        g_con[i] = torch.clamp(1 - C[i, j], min=0.0)
    L_con = F.binary_cross_entropy_with_logits(p_conf, g_con)

    # 5) Quadrant classification loss (currently dummy targets = 0)
    g_quad = torch.zeros(E_pred, dtype=torch.long, device=C.device)
    L_quad = F.cross_entropy(p_quad_logits, g_quad)

    # 6) Edge similarity average loss
    if matches:
        sim_vals = torch.stack([C[i, j] for i, j in matches])
        L_sim = sim_vals.mean()
    else:
        L_sim = torch.tensor(0.0, device=C.device)

    # 7) Final weighted loss
    total = (
        λ_mid  * mid_loss +
        λ_comp * comp_loss +
        λ_con  * L_con +
        λ_quad * L_quad +
        λ_sim  * L_sim
    )
    return total

# ─── Part 3: Dataset Handling ───────────────────────────────────────────────

def load_xyz(path: str) -> np.ndarray:
    """
    Load XYZ point cloud (first 3 columns).
    """
    data = np.loadtxt(path)
    return data[:, :3]

def load_wireframe_obj(path: str):
    """
    Parse a Wavefront .obj file to extract vertices and edge pairs.
    """
    verts, edges = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == 'v':
                x, y, z = map(float, parts[1:4])
                verts.append((x, y, z))
            elif parts[0] == 'l':
                idxs = [int(p) - 1 for p in parts[1:]]
                for a, b in zip(idxs, idxs[1:]):
                    edges.append((a, b))
    return np.array(verts, dtype=np.float32), edges

class WireframeDataset(Dataset):
    """
    PyTorch Dataset class for point clouds and wireframes.
    """
    def __init__(self, xyz_paths, obj_paths):
        self.xyz_paths = xyz_paths
        self.obj_paths = obj_paths

    def __len__(self):
        return len(self.xyz_paths)

    def __getitem__(self, idx):
        pc = torch.from_numpy(load_xyz(self.xyz_paths[idx])).float()
        verts, eidx = load_wireframe_obj(self.obj_paths[idx])
        gt_edges = torch.stack([
            torch.from_numpy(np.stack([verts[i], verts[j]], 0)).float()
            for i, j in eidx
        ], dim=0)
        return pc, gt_edges

# ─── Part 4: Dummy Model for Testing ────────────────────────────────────────

class DummyWireframeNet(torch.nn.Module):
    """
    A dummy network that learns to regress fixed-length edges from point cloud input.
    """
    def __init__(self, E_pred):
        super().__init__()
        self.E_pred = E_pred
        self.fc = torch.nn.Linear(3, 3)  # dummy linear mapping

    def forward(self, pc):
        pts = pc[: self.E_pred * 2]     # select first 2*E_pred points
        out = self.fc(pts)              # apply linear layer
        edges = out.view(self.E_pred, 2, 3)  # reshape into edges
        conf = torch.zeros(self.E_pred, device=pc.device)       # dummy confidence
        quad = torch.zeros(self.E_pred, 4, device=pc.device)    # dummy quadrant logits
        return edges, conf, quad

# ─── Part 5: Training Loop ──────────────────────────────────────────────────

def main():
    # Load training history from JSON file
    history_path = "denemeler.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: '{history_path}' contains invalid JSON. Overwriting.")
            history = []
    else:
        history = []

    # Dataset paths (replace with your own files)
    xyz_paths = ["1.xyz"]
    obj_paths = ["1.obj"]

    # Load dataset and prepare data loader
    ds = WireframeDataset(xyz_paths, obj_paths)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine number of predicted edges from GT
    _, first_gt = ds[0]
    E_pred = first_gt.shape[0]

    model = DummyWireframeNet(E_pred).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    a = True
    first_epoch = -1
    epoch_max_range = 6

    for epoch in range(1, epoch_max_range):
        if device.type == "cuda": torch.cuda.synchronize()
        start_time = time.time()

        total_loss = 0.0
        for pc_batch, gt_batch in loader:
            pc = pc_batch[0].to(device)
            gt_edges = gt_batch[0].to(device)

            pred_edges, p_conf, p_quad = model(pc)
            loss = wireframe_loss(pred_edges, gt_edges, p_conf, p_quad)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if device.type == "cuda": torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)

        # Peak memory monitoring (CUDA only)
        peak_mem = (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

        if a:
            first_epoch = avg_loss
            a = False

        print(f"Epoch {epoch:02d} — avg loss {avg_loss:.4f} — time {epoch_time:.3f}s — peak_mem {peak_mem:.1f}MB")
        history.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time_s": epoch_time,
            "peak_mem_mb": peak_mem
        })

    # Final productivity stats
    print(f"First one is {first_epoch:.4f} ")
    print(f"Last one is {avg_loss:.4f} ")
    diff = first_epoch - avg_loss
    print(f"Differenceses between first and last is {diff:.4f} ")
    b = diff / (epoch_max_range - 1)
    print(f"Productivity increase average per epoch {b:.4f} ")

    # Save training history to JSON
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
