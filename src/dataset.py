import numpy as np
import torch
import torch.nn as nn

def normalize_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    """
    Center a (Nx3) point cloud at the origin and scale so that its
    farthest point lies on the unit sphere.
    
    Args:
        xyz: np.ndarray of shape (N, 3)
    Returns:
        np.ndarray of shape (N, 3) normalized to unit sphere.
    """
    # 1) Compute centroid
    centroid = xyz.mean(axis=0)              # shape (3,)
    # 2) Shift to center at (0,0,0)
    xyz_centered = xyz - centroid            # shape (N,3)
    # 3) Compute max distance from origin
    max_radius = np.linalg.norm(xyz_centered, axis=1).max()
    # 4) Scale onto unit sphere
    return xyz_centered / max_radius


def normalize_wireframe(vertices: np.ndarray, lines=None):
    """
    Center and scale a 3D wireframe so that its farthest vertex lies on the unit sphere.

    Args:
        vertices: np.ndarray of shape (N, 3), the X/Y/Z coords of each vertex.
        lines: optional list of line definitions (e.g. ['l 1 2', 'l 2 3', ...]).
               These will be returned unchanged.

    Returns:
        norm_vertices: np.ndarray of shape (N, 3), normalized coords.
        lines: the same `lines` input, if provided (else None).
    """
    # 1) Compute centroid (mean across vertices)
    centroid = vertices.mean(axis=0)            # shape (3,)

    # 2) Translate so centroid moves to origin
    centered = vertices - centroid              # shape (N,3)

    # 3) Compute the max distance from origin
    max_radius = np.linalg.norm(centered, axis=1).max()

    # 4) Scale so that farthest point lies on the unit sphere
    norm_vertices = centered / max_radius       # shape (N,3)

    return norm_vertices, lines

def load_xyz(path: str) -> np.ndarray:
    """
    Load XYZ point cloud (first 3 columns).
    """
    data = np.loadtxt(path)
    return normalize_unit_sphere(data[:, :3])

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
    

    return normalize_wireframe(np.array(verts, dtype=np.float32), edges)