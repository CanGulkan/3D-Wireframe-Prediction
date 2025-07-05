import numpy as np
from scipy.spatial import cKDTree
from dataset import normalize_wireframe

def load_wireframe_obj(path: str):
    """
    Parse a Wavefront .obj file, normalize vertices and edge indices,
    and return an (M, 2, 3) array of edge endpoint coordinates.
    """
    verts, edges = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                x, y, z = map(float, parts[1:4])
                verts.append((x, y, z))
            elif parts[0] == 'l':
                idxs = [int(p) - 1 for p in parts[1:]]
                for a, b in zip(idxs, idxs[1:]):
                    edges.append((a, b))

    verts = np.array(verts, dtype=np.float32)    # (N,3)
    edges_idx = np.array(edges, dtype=int)       # (M,2)

    # normalize_wireframe → (verts_norm, edges_idx_norm)
    verts_norm, edges_idx_norm = normalize_wireframe(verts, edges_idx)

    # Build (M,2,3) array of endpoint coordinates
    wireframe_edges = verts_norm[edges_idx_norm]  # (M,2,3)
    return wireframe_edges

def sample_wireframe_edges(edges, samples_per_edge=10):
    """
    edges: (M, 2, 3)
    returns: (M * samples_per_edge, 3)
    """
    t = np.linspace(0, 1, samples_per_edge).reshape(samples_per_edge, 1)
    samples = []
    for v0, v1 in edges:
        pts = v0 + t * (v1 - v0)  # (samples_per_edge,3)
        samples.append(pts)
    return np.vstack(samples)

def rms_distance_to_wireframe(point_cloud, wireframe_edges, samples_per_edge=10):
    """
    point_cloud: (N, 3)
    wireframe_edges: (M, 2, 3)
    """
    sampled_pts = sample_wireframe_edges(wireframe_edges, samples_per_edge)
    tree = cKDTree(sampled_pts)
    dists, _ = tree.query(point_cloud, k=1)  # (N,)
    return np.sqrt(np.sum(dists**2) / dists.shape[0])

if __name__ == "__main__":
    # 1) Load only XYZ columns (first three) from your .xyz file
    P_full = np.loadtxt("1.xyz")
    if P_full.ndim != 2 or P_full.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in 1.xyz, got shape {P_full.shape}")
    P = P_full[:, :3]
    print("Point-cloud shape:", P.shape)  # should be (N, 3)

    # 2) Load wireframe edges
    E = load_wireframe_obj("1.obj")
    print("Wireframe-edges shape:", E.shape)  # should be (M, 2, 3)

    # 3) Compute RMS distance
    rms = rms_distance_to_wireframe(P, E, samples_per_edge=20)
    print(f"RMS distance (sampling): {rms:.8f}")
