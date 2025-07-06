import open3d as o3d
import numpy as np
import torch

def visualize_wireframe_open3d(pred_edges: torch.Tensor):
    """
    Visualize predicted edges as a wireframe using Open3D,
    with edges in green and vertices in red.
    
    Args:
        pred_edges (torch.Tensor): Tensor of shape [E, 2, 3] where E is number of edges.
    """
    pred_edges_np = pred_edges.detach().cpu().numpy()  # [E, 2, 3]

    # --- collect unique points and line segments ---
    points = []
    lines = []
    index_map = {}

    for edge in pred_edges_np:
        v0 = tuple(edge[0])
        v1 = tuple(edge[1])
        for v in (v0, v1):
            if v not in index_map:
                index_map[v] = len(points)
                points.append(v)
        lines.append([index_map[v0], index_map[v1]])

    pts_array = np.array(points)
    lines_array = np.array(lines)

    # --- build an Open3D LineSet for edges ---
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_array),
        lines=o3d.utility.Vector2iVector(lines_array)
    )
    # green for all edge segments
    line_set.colors = o3d.utility.Vector3dVector(
        [[0.1, 0.8, 0.1] for _ in lines_array]
    )

    # --- build an Open3D PointCloud for the vertices ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_array)
    # red for all points
    pcd.colors = o3d.utility.Vector3dVector(
        [[1.0, 0.0, 0.0] for _ in pts_array]
    )

    # --- visualize both together ---
    o3d.visualization.draw_geometries([line_set, pcd])
