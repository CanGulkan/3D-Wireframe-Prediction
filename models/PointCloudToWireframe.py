
import torch.nn as nn
from models.EdgePredictor import EdgePredictor
from models.PointNetEncoder import PointNetEncoder
from models.VertexPredictor import VertexPredictor


class PointCloudToWireframe(nn.Module):
    """Complete model for point cloud to wireframe prediction"""
    
    def __init__(self, input_dim=8, num_vertices=32):
        super(PointCloudToWireframe, self).__init__()
        
        self.num_vertices = num_vertices
        
        # Point cloud encoder
        self.encoder = PointNetEncoder(input_dim=input_dim)
        
        # Vertex predictor
        self.vertex_predictor = VertexPredictor(
            global_feature_dim=512, 
            num_vertices=num_vertices
        )
        
        # Edge predictor
        self.edge_predictor = EdgePredictor(vertex_dim=3)
        
    def forward(self, point_cloud):
        # Encode point cloud
        global_features, point_features = self.encoder(point_cloud)
        
        # Predict vertices
        predicted_vertices = self.vertex_predictor(global_features)
        
        # Predict edges
        edge_probs, edge_indices = self.edge_predictor(predicted_vertices)
        
        return {
            'vertices': predicted_vertices,
            'edge_probs': edge_probs,
            'edge_indices': edge_indices,
            'global_features': global_features
        }

