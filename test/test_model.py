import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from main import load_and_preprocess_data
from models.PointCloudToWireframe import PointCloudToWireframe
from visualize.visualize_wireframe import visualize_point_cloud, visualize_wireframe

def test_data_loading():
    """Test basic data loading and preprocessing"""
    print("Testing data loading...")
    
    dataset = load_and_preprocess_data()
    
    print(f"✓ Point cloud loaded: {dataset.point_cloud.shape}")
    print(f"✓ Vertices loaded: {dataset.vertices.shape}")
    print(f"✓ Edges loaded: {dataset.edges.shape}")
    print(f"✓ Adjacency matrix: {dataset.edge_adjacency_matrix.shape}")
    print(f"✓ Normalized point cloud: {dataset.normalized_point_cloud.shape}")
    
    # Check data ranges
    print(f"Point cloud range: X[{dataset.point_cloud[:, 0].min():.1f}, {dataset.point_cloud[:, 0].max():.1f}]")
    print(f"                  Y[{dataset.point_cloud[:, 1].min():.1f}, {dataset.point_cloud[:, 1].max():.1f}]")
    print(f"                  Z[{dataset.point_cloud[:, 2].min():.1f}, {dataset.point_cloud[:, 2].max():.1f}]")
    
    print(f"Vertices range:   X[{dataset.vertices[:, 0].min():.1f}, {dataset.vertices[:, 0].max():.1f}]")
    print(f"                  Y[{dataset.vertices[:, 1].min():.1f}, {dataset.vertices[:, 1].max():.1f}]")
    print(f"                  Z[{dataset.vertices[:, 2].min():.1f}, {dataset.vertices[:, 2].max():.1f}]")
    
    print(f"Number of edges: {len(dataset.edges)}")
    print(f"Edge connectivity: {dataset.edge_adjacency_matrix.sum() / 2:.0f} connections")
    
    return dataset

def test_model_forward_pass():
    """Test model forward pass without training"""
    print("\nTesting model forward pass...")
    
    dataset = load_and_preprocess_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    num_vertices = len(dataset.vertices)
    model = PointCloudToWireframe(input_dim=8, num_vertices=num_vertices).to(device)
    
    # Test forward pass
    point_cloud_tensor = torch.FloatTensor(dataset.normalized_point_cloud).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(point_cloud_tensor)
        
        print(f"✓ Model forward pass successful")
        print(f"✓ Predicted vertices shape: {output['vertices'].shape}")
        print(f"✓ Predicted edge probabilities shape: {output['edge_probs'].shape}")
        print(f"✓ Number of possible edges: {len(output['edge_indices'])}")
        print(f"✓ Global features shape: {output['global_features'].shape}")
        
        # Check output ranges
        vertices = output['vertices'].cpu().numpy()[0]
        edge_probs = output['edge_probs'].cpu().numpy()[0]
        
        print(f"Vertex predictions range: X[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
        print(f"                         Y[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
        print(f"                         Z[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
        print(f"Edge probabilities range: [{edge_probs.min():.3f}, {edge_probs.max():.3f}]")
        print(f"Edges with >0.5 probability: {(edge_probs > 0.5).sum()}")
    
    return model, dataset

def create_basic_visualizations():
    """Create basic visualizations of the data"""
    print("\nCreating basic visualizations...")
    
    dataset = load_and_preprocess_data()
    
    # Visualize point cloud (save without showing to avoid GUI issues)
    fig1, ax1 = visualize_point_cloud(dataset.point_cloud, "Original Point Cloud")
    plt.savefig('test_point_cloud.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("✓ Point cloud visualization saved as 'test_point_cloud.png'")
    
    # Visualize wireframe
    fig2, ax2 = visualize_wireframe(dataset.vertices, dataset.edges, "Original Wireframe")
    plt.savefig('test_wireframe.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("✓ Wireframe visualization saved as 'test_wireframe.png'")

def test_loss_computation():
    """Test loss computation"""
    print("\nTesting loss computation...")
    
    from losses.WireframeLoss import WireframeLoss
    from train import create_edge_labels_from_adjacency
    
    dataset = load_and_preprocess_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and get predictions
    num_vertices = len(dataset.vertices)
    model = PointCloudToWireframe(input_dim=8, num_vertices=num_vertices).to(device)
    
    point_cloud_tensor = torch.FloatTensor(dataset.normalized_point_cloud).unsqueeze(0).to(device)
    target_vertices = torch.FloatTensor(dataset.normalized_vertices).unsqueeze(0).to(device)
    
    # Create edge labels
    edge_indices = [(i, j) for i in range(num_vertices) for j in range(i+1, num_vertices)]
    edge_labels = create_edge_labels_from_adjacency(dataset.edge_adjacency_matrix, edge_indices).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(point_cloud_tensor)
        
        # Compute loss
        criterion = WireframeLoss(vertex_weight=1.0, edge_weight=2.0)
        targets = {
            'vertices': target_vertices,
            'edge_labels': edge_labels
        }
        loss_dict = criterion(predictions, targets)
        
        print(f"✓ Loss computation successful")
        print(f"✓ Total loss: {loss_dict['total_loss'].item():.6f}")
        print(f"✓ Vertex loss: {loss_dict['vertex_loss'].item():.6f}")
        print(f"✓ Edge loss: {loss_dict['edge_loss'].item():.6f}")

if __name__ == "__main__":
    print("="*60)
    print("POINT CLOUD TO WIREFRAME - MODEL TEST")
    print("="*60)
    
    try:
        # Test data loading
        dataset = test_data_loading()
        
        # Test model forward pass
        model, dataset = test_model_forward_pass()
        
        # Test loss computation
        test_loss_computation()
        
        # Create visualizations
        create_basic_visualizations()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print("\nThe model architecture is working correctly.")
        print("You can now run 'python main.py' to start overtraining the model.")
        print("Or run 'python visualize_wireframe.py' to see visualizations.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error and fix the issue.") 