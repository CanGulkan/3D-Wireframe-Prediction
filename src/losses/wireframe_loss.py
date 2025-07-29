import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def sample_edge_pts(verts, num_samples=10):
    """Uniformly sample points along an edge with adaptive sampling based on edge length."""
    length = torch.norm(verts[1] - verts[0])
    num_samples = max(2, int(num_samples * (length / 0.1)))  # Scale samples by length
    t = torch.linspace(0, 1, num_samples, device=verts.device).unsqueeze(1)
    return verts[0][None] * (1 - t) + verts[1][None] * t

def chamfer_distance(A, B):
    """Compute symmetric Chamfer distance between point sets A and B."""
    dist_A_B = torch.min(torch.cdist(A, B), dim=1).values
    dist_B_A = torch.min(torch.cdist(B, A), dim=1).values
    return (dist_A_B.mean() + dist_B_A.mean()) / 2

def edge_angle_loss(v1, v2):
    """Compute angle difference loss between two edge vectors."""
    cos_sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
    return 1 - torch.abs(cos_sim)

def edge_length_loss(l_pred, l_gt):
    """Compute length-aware loss with log-scale sensitivity."""
    ratio = (l_pred + 1e-8) / (l_gt + 1e-8)
    return torch.abs(torch.log(ratio))

def edge_similarity(e_pred, e_gt, alpha=1.0, beta=1.0, gamma=1.0, samples=10):
    """Enhanced edge dissimilarity metric with Chamfer distance."""
    A = sample_edge_pts(e_pred, samples)
    B = sample_edge_pts(e_gt, samples)
    
    # Geometry similarity using Chamfer distance (more robust than Hausdorff)
    geom_dist = chamfer_distance(A, B)
    
    # Direction similarity
    v_p = e_pred[1] - e_pred[0]
    v_g = e_gt[1] - e_gt[0]
    dir_dist = edge_angle_loss(v_p, v_g)
    
    # Length similarity (log-scale)
    len_dist = edge_length_loss(v_p.norm(), v_g.norm())
    
    return alpha * geom_dist + beta * dir_dist + gamma * len_dist

def wireframe_loss(pred_edges, gt_edges, p_conf, *,
                  alpha=1.0, beta=1.0, gamma=1.0,
                  λ_mid=1e-4, λ_comp=1e-4, λ_con=1.0, λ_sim=1.0,
                  λ_chamf=0.5, num_samples=20, temp=0.1):
    """
    Wireframe loss without quadrant classification.
    """
    E_pred, E_gt = pred_edges.shape[0], gt_edges.shape[0]
    
    # 1) Compute cost matrix
    C = torch.zeros(E_pred, E_gt, device=pred_edges.device)
    for i in range(E_pred):
        for j in range(E_gt):
            C[i, j] = edge_similarity(pred_edges[i], gt_edges[j], 
                                    alpha, beta, gamma, samples=num_samples)
    
    # 2) Hungarian matching
    with torch.no_grad():
        rows, cols = linear_sum_assignment(C.detach().cpu().numpy())
        matches = list(zip(rows, cols))
    
    # 3) Geometric losses
    mid_loss, comp_loss, chamf_loss = 0.0, 0.0, 0.0
    if matches:
        mid_diffs = []
        comp_diffs = []
        chamf_dists = []
        
        for i, j in matches:
            # Midpoint loss
            p_mid = pred_edges[i].mean(0)
            g_mid = gt_edges[j].mean(0)
            mid_diffs.append((p_mid - g_mid).abs().sum())
            
            # Component-wise loss
            v_p = pred_edges[i][1] - pred_edges[i][0]
            v_g = gt_edges[j][1] - gt_edges[j][0]
            comp_diffs.append((v_p - v_g).abs().sum())
            
            # Chamfer distance
            A = sample_edge_pts(pred_edges[i], num_samples)
            B = sample_edge_pts(gt_edges[j], num_samples)
            chamf_dists.append(chamfer_distance(A, B))
        
        mid_loss = torch.stack(mid_diffs).mean()
        comp_loss = torch.stack(comp_diffs).mean()
        chamf_loss = torch.stack(chamf_dists).mean()
    
    # 4) Confidence loss
    g_con = torch.zeros(E_pred, device=C.device)
    if matches:
        for i, j in matches:
            g_con[i] = torch.exp(-C[i, j] / temp)
    L_con = F.binary_cross_entropy_with_logits(p_conf, g_con)
    
    # 5) Similarity loss
    L_sim = C[torch.tensor(rows), torch.tensor(cols)].mean() if matches else torch.tensor(0.0, device=C.device)
    
    # 6) Existence loss for unmatched predictions
    unmatched_mask = torch.ones(E_pred, dtype=torch.bool, device=C.device)
    if matches:
        unmatched_mask[torch.tensor(rows)] = False
    L_exist = p_conf[unmatched_mask].sigmoid().mean() if unmatched_mask.any() else torch.tensor(0.0)
    
    # Final loss
    total = (
        λ_mid * mid_loss +
        λ_comp * comp_loss +
        λ_con * L_con +
        λ_sim * L_sim +
        λ_chamf * chamf_loss +
        0.1 * L_exist
    )
    
    return total, {
        'total': total.item(),
        'mid_loss': mid_loss.item(),
        'comp_loss': comp_loss.item(),
        'con_loss': L_con.item(),
        'sim_loss': L_sim.item(),
        'chamf_loss': chamf_loss.item(),
        'exist_loss': L_exist.item()
    }