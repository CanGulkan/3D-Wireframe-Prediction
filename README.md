# 3D Wireframe Prediction Training

This repository provides an end-to-end pipeline for training a 3D wireframe prediction model using PyTorch. It implements two recent transformer-based building-wireframe reconstruction methods—PBWR and BWFormer—to sample and match edges in aerial LiDAR point clouds.

- **PBWR**: Parametric-Building-Wireframe Reconstruction from Aerial LiDAR Point Clouds :contentReference[oaicite:0]{index=0}  
- **BWFormer**: Building Wireframe Reconstruction from Airborne LiDAR Point Cloud with Transformer :contentReference[oaicite:1]{index=1}


## Features

* **Edge Sampling & Similarity**: Uniform sampling along edges and computation of a weighted combination of Hausdorff distance, directional cosine similarity, and relative length difference.
* **Wireframe Loss**: Composite loss incorporating midpoint and component errors, confidence, quadrant classification, and edge similarity, matched with the Hungarian algorithm.
* **Dataset Handling**: Load point clouds from `.xyz` files and wireframes from Wavefront `.obj` files, packaged into a PyTorch `Dataset` and `DataLoader`.
* **Dummy Model**: A simple linear model (`DummyWireframeNet`) to test the training pipeline.
* **Training Loop**: Multi-epoch training with average loss reporting, time and (optional) CUDA memory profiling, and history saved to `denemeler.json`.

## Requirements

* Python 3.7+
* PyTorch
* NumPy
* SciPy

Install dependencies via:

```bash
pip install torch numpy scipy
```

## Repository Structure

```plaintext
.
├── README.md             # Project overview and instructions
├── denemeler.json        # Training history (created during training)
├── 1.xyz                 # Example point cloud data (replace with your own)
├── 1.obj                 # Example wireframe data (replace with your own)
└── wireframe_train.py    # Main Python script containing all code sections
```

## Usage

1. **Prepare Data**: Place your point cloud files (`.xyz`) and corresponding wireframe files (`.obj`) in the project directory. Update the `xyz_paths` and `obj_paths` lists in `wireframe_train.py` if necessary.

2. **Run Training**:

   ```bash
   python wireframe_train.py
   ```

   This will:

   * Load or initialize `denemeler.json` for logging.
   * Instantiate the dataset, model, and optimizer.
   * Perform training for 5 epochs by default.
   * Print epoch statistics (average loss, time, peak memory).
   * Save updated training history to `denemeler.json`.

3. **Inspect Results**: Open `denemeler.json` to visualize the training loss over epochs or integrate with other tools.

## Code Structure

* **Part 1: Utility Functions** (`sample_edge_pts`, `directed_hausdorff`, `edge_similarity`)
* **Part 2: Loss Function** (`wireframe_loss` with Hungarian matching and weighted components)
* **Part 3: Dataset Handling** (`load_xyz`, `load_wireframe_obj`, `WireframeDataset`)
* **Part 4: Dummy Model** (`DummyWireframeNet`)
* **Part 5: Training Loop** (`main` function)

## Customization

* Adjust loss weights (`λ_mid`, `λ_comp`, `λ_con`, `λ_quad`, `λ_sim`) and sampling density (`num_samples`) in `wireframe_loss` arguments.
* Replace `DummyWireframeNet` with your own architecture for real-world experiments.
* Modify dataset paths, batch size, learning rate, and number of epochs in the `main` function.

---

Feel free to contribute improvements or report issues!