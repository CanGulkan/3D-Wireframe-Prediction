import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import open3d as o3d
from main import load_and_preprocess_data
from models.PointCloudToWireframe import PointCloudToWireframe
import time

def load_trained_model():
    """Load the trained model"""
    print("Loading trained model...")
    
    # Load dataset to get model parameters
    dataset = load_and_preprocess_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with same architecture
    num_vertices = len(dataset.vertices)
    model = PointCloudToWireframe(input_dim=8, num_vertices=num_vertices).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model.eval()
    
    print(f"✓ Model loaded successfully on {device}")
    return model, dataset, device

def create_point_cloud_o3d(points, colors=None, point_size=2.0):
    """Create Open3D point cloud from numpy array"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if colors is not None:
        # Normalize colors to [0,1] if needed
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Use RGB only
    else:
        # Default color: light blue
        pcd.paint_uniform_color([0.7, 0.8, 1.0])
    
    return pcd

def create_wireframe_o3d(vertices, edges, color=[1.0, 0.0, 0.0], line_width=5.0):
    """Create Open3D line set from vertices and edges"""
    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    
    # Set colors for all lines
    colors = [color for _ in range(len(edges))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def create_vertex_spheres(vertices, color=[1.0, 0.0, 0.0], radius=0.5):
    """Create spheres at vertex locations for better visibility"""
    spheres = []
    for vertex in vertices:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(vertex)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    return spheres

def visualize_point_cloud_only():
    """Visualize just the point cloud data"""
    print("Creating point cloud visualization...")
    
    dataset = load_and_preprocess_data()
    
    # Create point cloud
    point_cloud = dataset.point_cloud
    colors = point_cloud[:, 3:7]  # RGBA colors
    
    # Sample points if too many (for performance)
    max_points = 5000
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        sampled_points = point_cloud[indices]
        sampled_colors = colors[indices]
    else:
        sampled_points = point_cloud
        sampled_colors = colors
    
    pcd = create_point_cloud_o3d(sampled_points, sampled_colors)
    
    # Visualize
    print("Opening point cloud visualization...")
    print("Controls: Mouse to rotate, scroll to zoom, drag to pan")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud Visualization",
        width=1200,
        height=800,
        left=50,
        top=50
    )

def visualize_ground_truth_wireframe():
    """Visualize ground truth wireframe only"""
    print("Creating ground truth wireframe visualization...")
    
    dataset = load_and_preprocess_data()
    
    # Create wireframe
    wireframe = create_wireframe_o3d(
        dataset.vertices, 
        dataset.edges, 
        color=[0.0, 0.0, 1.0],  # Blue for ground truth
        line_width=5.0
    )
    
    # Create vertex spheres
    vertex_spheres = create_vertex_spheres(
        dataset.vertices, 
        color=[0.0, 0.0, 1.0], 
        radius=0.3
    )
    
    # Combine geometries
    geometries = [wireframe] + vertex_spheres
    
    # Visualize
    print("Opening ground truth wireframe visualization...")
    print("Controls: Mouse to rotate, scroll to zoom, drag to pan")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Ground Truth Wireframe",
        width=1200,
        height=800,
        left=50,
        top=50
    )

def visualize_predicted_wireframe():
    """Visualize predicted wireframe only"""
    print("Creating predicted wireframe visualization...")
    
    model, dataset, device = load_trained_model()
    
    # Get predictions
    with torch.no_grad():
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
    
    # Create wireframe
    wireframe = create_wireframe_o3d(
        pred_vertices_original, 
        pred_edges, 
        color=[0.0, 1.0, 0.0],  # Green for predictions
        line_width=5.0
    )
    
    # Create vertex spheres
    vertex_spheres = create_vertex_spheres(
        pred_vertices_original, 
        color=[0.0, 1.0, 0.0], 
        radius=0.3
    )
    
    # Combine geometries
    geometries = [wireframe] + vertex_spheres
    
    # Visualize
    print("Opening predicted wireframe visualization...")
    print("Controls: Mouse to rotate, scroll to zoom, drag to pan")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Predicted Wireframe",
        width=1200,
        height=800,
        left=50,
        top=50
    )

def visualize_comparison_overlay():
    """Visualize ground truth vs predicted wireframes in overlay"""
    print("Creating comparison overlay visualization...")
    
    model, dataset, device = load_trained_model()
    
    # Get predictions
    with torch.no_grad():
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
    
    # Create ground truth wireframe (blue)
    gt_wireframe = create_wireframe_o3d(
        dataset.vertices, 
        dataset.edges, 
        color=[0.0, 0.0, 1.0],  # Blue
        line_width=6.0
    )
    
    # Create predicted wireframe (green)
    pred_wireframe = create_wireframe_o3d(
        pred_vertices_original, 
        pred_edges, 
        color=[0.0, 1.0, 0.0],  # Green
        line_width=4.0
    )
    
    # Create vertex spheres for ground truth (blue)
    gt_vertex_spheres = create_vertex_spheres(
        dataset.vertices, 
        color=[0.0, 0.0, 1.0], 
        radius=0.4
    )
    
    # Create vertex spheres for predictions (green)
    pred_vertex_spheres = create_vertex_spheres(
        pred_vertices_original, 
        color=[0.0, 1.0, 0.0], 
        radius=0.3
    )
    
    # Combine all geometries
    geometries = [gt_wireframe, pred_wireframe] + gt_vertex_spheres + pred_vertex_spheres
    
    # Visualize
    print("Opening comparison overlay visualization...")
    print("Blue = Ground Truth | Green = Predicted")
    print("Controls: Mouse to rotate, scroll to zoom, drag to pan")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Wireframe Comparison: Blue=Ground Truth, Green=Predicted",
        width=1400,
        height=900,
        left=50,
        top=50
    )

def visualize_comprehensive():
    """Comprehensive visualization with point cloud + wireframes"""
    print("Creating comprehensive visualization...")
    
    model, dataset, device = load_trained_model()
    
    # Get predictions
    with torch.no_grad():
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
    
    # Create point cloud (light gray, transparent)
    point_cloud = dataset.point_cloud
    max_points = 3000  # Reduced for performance
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        sampled_points = point_cloud[indices]
    else:
        sampled_points = point_cloud
    
    pcd = create_point_cloud_o3d(sampled_points)
    pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    
    # Create ground truth wireframe (blue, thick)
    gt_wireframe = create_wireframe_o3d(
        dataset.vertices, 
        dataset.edges, 
        color=[0.0, 0.0, 1.0],  # Blue
        line_width=8.0
    )
    
    # Create predicted wireframe (red, slightly thinner)
    pred_wireframe = create_wireframe_o3d(
        pred_vertices_original, 
        pred_edges, 
        color=[1.0, 0.0, 0.0],  # Red for contrast
        line_width=6.0
    )
    
    # Create vertex spheres
    gt_vertex_spheres = create_vertex_spheres(
        dataset.vertices, 
        color=[0.0, 0.0, 1.0], 
        radius=0.5
    )
    
    pred_vertex_spheres = create_vertex_spheres(
        pred_vertices_original, 
        color=[1.0, 0.0, 0.0], 
        radius=0.4
    )
    
    # Combine all geometries
    geometries = [pcd, gt_wireframe, pred_wireframe] + gt_vertex_spheres + pred_vertex_spheres
    
    # Visualize
    print("Opening comprehensive visualization...")
    print("Gray = Point Cloud | Blue = Ground Truth | Red = Predicted")
    print("Controls: Mouse to rotate, scroll to zoom, drag to pan")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Complete View: Point Cloud + Wireframe Comparison",
        width=1600,
        height=1000,
        left=50,
        top=50
    )

def save_high_quality_images():
    """Save high-quality rendered images"""
    print("Saving high-quality Open3D renderings...")
    
    model, dataset, device = load_trained_model()
    
    # Get predictions
    with torch.no_grad():
        point_cloud_tensor = torch.FloatTensor(dataset.normalized_point_cloud).unsqueeze(0).to(device)
        predictions = model(point_cloud_tensor)
        
        pred_vertices = predictions['vertices'].cpu().numpy()[0]
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
        
        pred_vertices_original = dataset.spatial_scaler.inverse_transform(pred_vertices)
        
        pred_edges = []
        for idx, (i, j) in enumerate(edge_indices):
            if pred_edge_probs[idx] > 0.5:
                pred_edges.append([i, j])
        pred_edges = np.array(pred_edges) if pred_edges else np.array([]).reshape(0, 2)
    
    # Create geometries for comparison
    gt_wireframe = create_wireframe_o3d(dataset.vertices, dataset.edges, [0.0, 0.0, 1.0], 6.0)
    pred_wireframe = create_wireframe_o3d(pred_vertices_original, pred_edges, [0.0, 1.0, 0.0], 4.0)
    gt_spheres = create_vertex_spheres(dataset.vertices, [0.0, 0.0, 1.0], 0.4)
    pred_spheres = create_vertex_spheres(pred_vertices_original, [0.0, 1.0, 0.0], 0.3)
    
    geometries = [gt_wireframe, pred_wireframe] + gt_spheres + pred_spheres
    
    # Create visualizer for high-quality rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background
    render_option.line_width = 10.0
    render_option.point_size = 8.0
    
    # Capture and save image
    vis.capture_screen_image("open3d_wireframe_comparison.png")
    vis.destroy_window()
    
    print("✓ Saved: open3d_wireframe_comparison.png")

if __name__ == "__main__":
    print("="*70)
    print("OPEN3D INTERACTIVE 3D WIREFRAME VISUALIZATION")
    print("="*70)
    
    while True:
        print("\nChoose visualization:")
        print("1. Point Cloud Only")
        print("2. Ground Truth Wireframe Only") 
        print("3. Predicted Wireframe Only")
        print("4. Comparison Overlay (Ground Truth vs Predicted)")
        print("5. Comprehensive View (Point Cloud + Both Wireframes)")
        print("6. Save High-Quality Images")
        print("0. Exit")
        
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                visualize_point_cloud_only()
            elif choice == "2":
                visualize_ground_truth_wireframe()
            elif choice == "3":
                visualize_predicted_wireframe()
            elif choice == "4":
                visualize_comparison_overlay()
            elif choice == "5":
                visualize_comprehensive()
            elif choice == "6":
                save_high_quality_images()
            else:
                print("Invalid choice. Please enter 0-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Make sure you have trained the model first by running 'python main.py'") 