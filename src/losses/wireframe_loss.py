import torch
import torch.nn.functional as F     
from scipy.optimize import linear_sum_assignment

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