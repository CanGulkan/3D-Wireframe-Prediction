
import os
import json
import time
import torch
import numpy as np       
import torch.nn as nn                                
from torch.utils.data import Dataset, DataLoader  
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm for optimal matching           
from models import TransformerWireframeNet
from models.dummy_net import DummyWireframeNet
from losses.wireframe_loss import wireframe_loss    
from dataset import load_wireframe_obj , load_xyz        
from visualize import visualize_wireframe_open3d 
from test import rms_distance_exact
from test import graph_edit_distance

def extract_vertices_and_edges(edge_tensor):
    """
    From edge tensor [E, 2, 3], extract unique vertices and edge indices.
    Returns:
        vertices: np.ndarray [N, 3]
        edges: np.ndarray [E, 2] (index into vertices)
    """
    edge_np = edge_tensor.reshape(-1, 3)  # Flatten to [2E, 3]
    vertices, inverse_indices = np.unique(edge_np, axis=0, return_inverse=True)
    edges = inverse_indices.reshape(-1, 2)
    return vertices, edges


# ─── Part 3: Dataset Handling ───────────────────────────────────────────────

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
    epoch_max_range = 4

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

        print(f"Epoch {epoch:02d} — avg loss {avg_loss:.8f} — time {epoch_time:.6f}s — peak_mem {peak_mem:.1f}MB")
        history.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time_s": epoch_time,
            "peak_mem_mb": peak_mem
        })

    # Final productivity stats
    print(f"First one is {first_epoch:.8f} ")
    print(f"Last one is {avg_loss:.8f} ")
    diff = first_epoch - avg_loss
    print(f"Differenceses between first and last is {diff:.8f} ")
    b = diff / (epoch_max_range - 1)
    print(f"Productivity increase average per epoch {b:.8f} ")


    pred_np = pred_edges.detach().cpu().numpy()   # shape [E_pred, 2, 3]
    gt_np   = gt_edges.detach().cpu().numpy()     # shape [E_gt,   2, 3]


    pd_vertices, pd_edges = extract_vertices_and_edges(pred_np)
    gt_vertices, gt_edges = extract_vertices_and_edges(gt_np)

    rms = rms_distance_exact(pred_np, gt_np)
    print(f"RMS distance (exact): {rms:.8f}")
    
    wde = graph_edit_distance(pd_vertices, pd_edges, gt_vertices, gt_edges, wed_v=0) # For simplicity, using RMS as GED
    print(f"Graph_edit_distance : {wde:.8f}")


    # visualize_wireframe_open3d(pred_edges)

    # Save training history to JSON
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
