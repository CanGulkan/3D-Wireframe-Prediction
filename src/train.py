import os
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from losses.wireframe_loss import wireframe_loss    # Custom loss for wireframe prediction
from dataset import load_wireframe_obj, load_xyz      # Utilities to load point clouds and .obj wireframe data
from visualize import visualize_wireframe_open3d      # Function to display a wireframe in Open3D
from test import rms_distance_exact, graph_edit_distance  # Evaluation metrics
from models.bwformer import BWFormer                 # Transformer-based wireframe model


from torch.optim.lr_scheduler import StepLR # Learning rate scheduler for adaptive learning rate control



def extract_vertices_and_edges(edge_tensor):
    """
    From a tensor of edges (shape [E, 2, 3]), extract:
      - unique vertices as an (N, 3) array
      - edge indices as an (E, 2) array that indexes into vertices
    Returns:
        vertices: np.ndarray of shape [N, 3]
        edges: np.ndarray of shape [E, 2]
    """
    # Flatten edge endpoints to a list of points [2E, 3]
    edge_np = edge_tensor.reshape(-1, 3)
    # Find unique vertices and get indices mapping back to edge points
    vertices, inverse_indices = np.unique(edge_np, axis=0, return_inverse=True)
    # Reshape the flat indices into pairs for each edge
    edges = inverse_indices.reshape(-1, 2)
    return vertices, edges

def normalize_by_gt(pc: torch.Tensor, gt_edges: torch.Tensor, eps: float = 1e-8):
    """
    Center & scale both `pc` and `gt_edges` so that all GT vertices
    lie within the unit sphere.
    
    Args:
        pc:        Tensor [N_points, 3]
        gt_edges:  Tensor [E_gt, 2, 3]
        eps:       small constant to avoid div by zero
    Returns:
        pc_norm:       Tensor [N_points, 3]
        gt_edges_norm: Tensor [E_gt,   2, 3]
    """
    # 1) flatten GT edge verts to [2*E_gt, 3]
    pts_gt = gt_edges.view(-1, 3)                  # [2*E_gt,3]
    # 2) centroid of GT
    centroid = pts_gt.mean(dim=0, keepdim=True)    # [1,3]
    # 3) max radius
    max_rad = (pts_gt - centroid).norm(dim=1).max()# scalar
    # 4) apply same transform
    pc_norm       = (pc       - centroid) / (max_rad + eps)
    gt_edges_norm = (gt_edges - centroid) / (max_rad + eps)
    return pc_norm, gt_edges_norm


# ─── Part 3: Dataset Handling ───────────────────────────────────────────────

class WireframeDataset(Dataset):
    """
    PyTorch Dataset for loading point cloud samples and their ground-truth wireframe edges.
    Each item returns:
      - pc: Tensor of shape [N_points, 3]
      - gt_edges: Tensor of shape [E_gt, 2, 3]
    """
    def __init__(self, xyz_paths, obj_paths):
        # Store lists of file paths for .xyz point clouds and .obj wireframes
        self.xyz_paths = xyz_paths
        self.obj_paths = obj_paths

    def __len__(self):
        # Number of samples in the dataset
        return len(self.xyz_paths)

    def __getitem__(self, idx):
        # Load the point cloud from .xyz file and convert to float Tensor
        pc = torch.from_numpy(load_xyz(self.xyz_paths[idx])).float()
        # Load vertices and edge index pairs from .obj file
        verts, eidx = load_wireframe_obj(self.obj_paths[idx])
        # Build ground-truth edge tensor by stacking endpoint coordinates
        gt_edges = torch.stack([
            torch.from_numpy(np.stack([verts[i], verts[j]], axis=0)).float()
            for i, j in eidx
        ], dim=0)
        # 2) normalize both so that GT fits in unit sphere at origin
        pc, gt_edges = normalize_by_gt(pc, gt_edges)

        return pc, gt_edges


# ─── Part 5: Training Loop ──────────────────────────────────────────────────

def main():
    # Path to JSON file that stores training history between runs
    history_path = "denemeler.json"
    if os.path.exists(history_path):
        try:
            # Load existing history
            with open(history_path, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted, warn and start fresh
            print(f"Warning: '{history_path}' contains invalid JSON. Overwriting.")
            history = []
    else:
        # No history file yet
        history = []

    # Define dataset file paths (replace with your own file list)
    xyz_paths = ["1.xyz"]
    obj_paths = ["1.obj"]

    # Instantiate dataset and data loader
    ds = WireframeDataset(xyz_paths, obj_paths)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # Speeds up host to GPU transfers if CUDA is used
    )

    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Initialize the model and move it to the chosen device
    model = BWFormer().to(device)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 1) Define the scheduler right after your optimizer:
    scheduler = StepLR(optimizer, step_size=10, gamma=1.0) # defult 0.5

    # Variables to track loss improvements
    first_epoch_loss = None
    max_epochs = 16

    # Training loop
    for epoch in range(1, max_epochs + 1):
        # If using CUDA, synchronize to get accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        total_loss = 0.0
        # Iterate over batches
        for pc_batch, gt_batch in loader:
            # Move data to device
            pc = pc_batch.to(device)
            gt_edges = gt_batch[0].to(device)

            # Forward pass: predict edges, confidences, and quadrant logits
            pred_edges, p_conf, p_quad = model(pc)

            # Squeeze batch dimension from confidences
            p_conf = p_conf.squeeze(0)
            # Compute wireframe loss
            loss = wireframe_loss(pred_edges, gt_edges, p_conf, p_quad)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        # 2) Step the scheduler once per epoch:
        scheduler.step()

        # 3) (Optional) Log the current LR:
        current_lr = scheduler.get_last_lr()[0]

        # Synchronize again if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)

        # Track peak GPU memory (if applicable)
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
            torch.cuda.reset_peak_memory_stats(device)
        else:
            peak_mem = None

        # Record the first epoch's loss for productivity stats
        if first_epoch_loss is None:
            first_epoch_loss = avg_loss

        # Print training progress
        print(f"Epoch {epoch:02d} — avg loss {avg_loss:.8f} — LR: {current_lr:.5f} — time {epoch_time:.6f}s — peak_mem {peak_mem:.1f}MB")

        # Append to history for saving later
        history.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time_s": epoch_time,
            "peak_mem_mb": peak_mem
        })

    # After training, compute productivity improvement
    print(f"First one is {first_epoch_loss:.8f}")
    print(f"Last one is {avg_loss:.8f}")
    diff = first_epoch_loss - avg_loss
    print(f"Differences between first and last is {diff:.8f}")
    avg_improvement = diff / (max_epochs - 1)
    print(f"Productivity increase average per epoch {avg_improvement:.8f}")

    # Convert predictions and GT to NumPy for evaluation
    pred_np = pred_edges.detach().cpu().numpy()   # [E_pred, 2, 3]
    gt_np   = gt_edges.detach().cpu().numpy()     # [E_gt,   2, 3]

    visualize_wireframe_open3d(pred_edges)

    # Extract vertices and edge indices
    pd_vertices, pd_edges = extract_vertices_and_edges(pred_np)
    gt_vertices, gt_edges = extract_vertices_and_edges(gt_np)

    # Compute RMS distance between predicted and GT edges
    rms = rms_distance_exact(pred_np, gt_np)
    print(f"RMS distance (exact): {rms:.8f}")

    # Compute graph edit distance as another metric
    wde = graph_edit_distance(pd_vertices, pd_edges, gt_vertices, gt_edges, wed_v=0)
    print(f"Graph_edit_distance : {wde:.8f}")

    # Optional visualization (uncomment to view)
    

    

    # Save the training history back to JSON file
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
