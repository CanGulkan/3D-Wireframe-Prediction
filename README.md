# 3D Wireframe Prediction

This repository provides a modular PyTorch pipeline for training a 3D wireframe prediction model from point cloud data. It supports transformer-based and dummy models to reconstruct building wireframes from aerial LiDAR scans.

The implementation is inspired by two recent papers:
- **PBWR** – Parametric-Building-Wireframe Reconstruction from Aerial LiDAR Point Clouds  
- **BWFormer** – Building Wireframe Reconstruction from Airborne LiDAR Point Cloud with Transformer

---

## 📌 Features

- **Edge Sampling & Similarity**  
  Compute a composite similarity score using Hausdorff distance, directional cosine, and length ratios.
  
- **Hungarian Matching Loss**  
  Matches predicted and ground-truth edges optimally, and computes loss using multiple weighted terms (midpoint, components, confidence, quadrant, similarity).

- **Modular Architecture**  
  Includes both a simple baseline model (`DummyWireframeNet`) and a transformer-based encoder-decoder (`TransformerWireframeNet`).

- **Point Cloud Dataset Loader**  
  Loads `.xyz` and `.obj` files into PyTorch `Dataset` format.

- **Training Loop**  
  Logs average loss, time, and optional CUDA memory for each epoch; results are saved to `denemeler.json`.

---

## 📁 Project Structure

```plaintext
3D-Wireframe-Prediction/
├── src/
│   ├── models/
│   │   ├── dummy_net.py              # Simple linear model
│   │   ├── transformer_net.py        # Transformer-based model
│   │   └── __init__.py               # For model imports
│   ├── losses/
│   │   └── wireframe_loss.py         # Loss function with matching
│   ├── dataset.py                    # Dataset class and data loading utils
│   ├── train.py                      # Main training script
├── data/
│   ├── 1.xyz                         # Sample point cloud
│   └── 1.obj                         # Sample wireframe edges
├── denemeler.json                    # Training history (generated during training)
├── README.md                         # You are here!


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