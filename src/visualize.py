import open3d as o3d
import numpy as np
import torch

def visualize_wireframe_open3d(pred_edges: torch.Tensor, gt_edges: torch.Tensor = None):
    """
    Visualize predicted and ground-truth edges as wireframes using Open3D.
    
    Args:
        pred_edges (torch.Tensor): Tensor of shape [E1, 2, 3] for predicted edges.
        gt_edges (torch.Tensor, optional): Tensor of shape [E2, 2, 3] for ground-truth edges.
    """
    pred_edges_np = pred_edges.detach().cpu().numpy()  # [E1, 2, 3]
    gt_edges_np = gt_edges.detach().cpu().numpy() if gt_edges is not None else None  # [E2, 2, 3]

    # --- helper to extract points and line indices ---
    def extract_edges_and_points(edges_np):
        points = []
        lines = []
        index_map = {}
        for edge in edges_np:
            v0 = tuple(edge[0])
            v1 = tuple(edge[1])
            for v in (v0, v1):
                if v not in index_map:
                    index_map[v] = len(points)
                    points.append(v)
            lines.append([index_map[v0], index_map[v1]])
        return np.array(points), np.array(lines)

    # --- predicted ---
    pred_points, pred_lines = extract_edges_and_points(pred_edges_np)

    # --- ground truth (optional) ---
    if gt_edges_np is not None:
        gt_points, gt_lines = extract_edges_and_points(gt_edges_np)
        # Offset GT point indices to avoid conflict with pred point indices
        gt_lines_offset = gt_lines + len(pred_points)
        all_points = np.concatenate([pred_points, gt_points], axis=0)
        all_lines = np.concatenate([pred_lines, gt_lines_offset], axis=0)
        colors = (
            [[0.1, 0.8, 0.1]] * len(pred_lines) +  # green for predicted
            [[0.1, 0.1, 1.0]] * len(gt_lines)      # blue for ground truth
        )
    else:
        all_points = pred_points
        all_lines = pred_lines
        colors = [[0.1, 0.8, 0.1]] * len(pred_lines)  # only green lines

    # --- LineSet for edges ---
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(all_lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # --- PointCloud for vertices (all shown in red) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for _ in all_points])

    # --- visualize ---
    o3d.visualization.draw_geometries([line_set, pcd])
