import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from demo_dataset.PointCloudWireframeDataset import PointCloudWireframeDataset
from models.PointCloudToWireframe import PointCloudToWireframe
from main import load_and_preprocess_data

def visualize_point_cloud(point_cloud, title="Point Cloud", max_points=1000):
    """Visualize point cloud data"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points if too many
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        sampled_points = point_cloud[indices]
    else:
        sampled_points = point_cloud
    
    # Extract coordinates and colors
    x, y, z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    colors = sampled_points[:, 3:7] / 255.0  # RGB values
    
    # Plot points
    ax.scatter(x, y, z, c=colors[:, :3], s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig, ax

def visualize_wireframe(vertices, edges, title="Wireframe", color='blue'):
    """Visualize wireframe structure"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='Vertices')
    
    # Plot edges
    for edge in edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [vertices[v1, 0], vertices[v2, 0]]
        y_vals = [vertices[v1, 1], vertices[v2, 1]]
        z_vals = [vertices[v1, 2], vertices[v2, 2]]
        ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    return fig, ax

def visualize_prediction_comparison(dataset, model, device):
    """Compare predicted vs actual wireframes"""
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        point_cloud_tensor = torch.FloatTensor(dataset.normalized_point_cloud).unsqueeze(0).to(device)
        predictions = model(point_cloud_tensor)
        
        # Extract predictions
        pred_vertices = predictions['vertices'].cpu().numpy()[0]
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
        
        # Convert back to original scale
        pred_vertices_original = dataset.spatial_scaler.inverse_transform(pred_vertices)
        
        # Create predicted edges (threshold at 0.5)
        pred_edges = []
        for idx, (i, j) in enumerate(edge_indices):
            if pred_edge_probs[idx] > 0.5:
                pred_edges.append([i, j])
        pred_edges = np.array(pred_edges) if pred_edges else np.array([]).reshape(0, 2)
    
    # Create comparison visualization
    fig = plt.figure(figsize=(20, 8))
    
    # Original wireframe
    ax1 = fig.add_subplot(131, projection='3d')
    vertices = dataset.vertices
    edges = dataset.edges
    
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='Vertices')
    
    for edge in edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [vertices[v1, 0], vertices[v2, 0]]
        y_vals = [vertices[v1, 1], vertices[v2, 1]]
        z_vals = [vertices[v1, 2], vertices[v2, 2]]
        ax1.plot(x_vals, y_vals, z_vals, color='blue', linewidth=2, alpha=0.7)
    
    ax1.set_title('Ground Truth Wireframe')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Predicted wireframe
    ax2 = fig.add_subplot(132, projection='3d')
    
    ax2.scatter(pred_vertices_original[:, 0], pred_vertices_original[:, 1], pred_vertices_original[:, 2], 
               c='red', s=50, alpha=0.8, label='Vertices')
    
    for edge in pred_edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [pred_vertices_original[v1, 0], pred_vertices_original[v2, 0]]
        y_vals = [pred_vertices_original[v1, 1], pred_vertices_original[v2, 1]]
        z_vals = [pred_vertices_original[v1, 2], pred_vertices_original[v2, 2]]
        ax2.plot(x_vals, y_vals, z_vals, color='green', linewidth=2, alpha=0.7)
    
    ax2.set_title('Predicted Wireframe')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Overlay comparison
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot both sets of vertices
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='True Vertices')
    ax3.scatter(pred_vertices_original[:, 0], pred_vertices_original[:, 1], pred_vertices_original[:, 2], 
               c='orange', s=30, alpha=0.6, label='Pred Vertices')
    
    # Plot true edges in blue
    for edge in edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [vertices[v1, 0], vertices[v2, 0]]
        y_vals = [vertices[v1, 1], vertices[v2, 1]]
        z_vals = [vertices[v1, 2], vertices[v2, 2]]
        ax3.plot(x_vals, y_vals, z_vals, color='blue', linewidth=3, alpha=0.7, label='True Edges' if edge is edges[0] else "")
    
    # Plot predicted edges in green
    for i, edge in enumerate(pred_edges):
        v1, v2 = edge[0], edge[1]
        x_vals = [pred_vertices_original[v1, 0], pred_vertices_original[v2, 0]]
        y_vals = [pred_vertices_original[v1, 1], pred_vertices_original[v2, 1]]
        z_vals = [pred_vertices_original[v1, 2], pred_vertices_original[v2, 2]]
        ax3.plot(x_vals, y_vals, z_vals, color='green', linewidth=2, alpha=0.7, 
                linestyle='--', label='Pred Edges' if i == 0 else "")
    
    ax3.set_title('Comparison Overlay')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    return fig

def plot_training_loss(loss_history):
    """Plot training loss curve"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(loss_history, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    # Add smoothed curve
    if len(loss_history) > 50:
        window_size = max(1, len(loss_history) // 50)
        smoothed = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        ax.plot(range(window_size//2, len(loss_history) - window_size//2), smoothed, 'r-', 
                linewidth=3, alpha=0.7, label='Smoothed')
        ax.legend()
    
    return fig

def visualize_edge_probabilities(edge_probs, edge_indices, threshold=0.5):
    """Visualize edge probability distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram of edge probabilities
    ax1.hist(edge_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Edge Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Edge Probabilities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Edge probabilities sorted
    sorted_probs = np.sort(edge_probs)[::-1]
    # Use 1-based indexing for logarithmic scale (log(0) is undefined)
    x_indices = np.arange(1, len(sorted_probs) + 1)
    ax2.plot(x_indices, sorted_probs, 'b-', linewidth=2)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax2.set_xlabel('Edge Index (sorted by probability) - Log Scale')
    ax2.set_ylabel('Edge Probability')
    ax2.set_title('Sorted Edge Probabilities')
    ax2.set_xscale('log', base=2)  # Make x-axis logarithmic with base 100
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load dataset
    dataset = load_and_preprocess_data()
    
    # Visualize point cloud
    print("Creating point cloud visualization...")
    fig1 = visualize_point_cloud(dataset.point_cloud, "Original Point Cloud")
    plt.savefig('point_cloud_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Visualize original wireframe
    print("Creating wireframe visualization...")
    fig2 = visualize_wireframe(dataset.vertices, dataset.edges, "Original Wireframe")
    plt.savefig('wireframe_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved!")
    print("To use prediction comparison, first train the model using main.py") 