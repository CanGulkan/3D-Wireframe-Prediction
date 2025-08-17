
import torch.nn as nn

class VertexPredictor(nn.Module):
    """Enhanced vertex predictor with massive capacity for total overfitting"""
    
    def __init__(self, global_feature_dim=512, num_vertices=32, vertex_dim=3):
        super(VertexPredictor, self).__init__()
        
        self.num_vertices = num_vertices
        self.vertex_dim = vertex_dim
        
        # Massive architecture for total overfitting
        self.vertex_mlp1 = nn.Sequential(
            nn.Linear(global_feature_dim, 2048),  # Much larger capacity
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # No dropout for overfitting
        )
        
        self.vertex_mlp2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # No dropout
        )
        
        self.vertex_mlp3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # No dropout
        )
        
        self.vertex_mlp4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # No dropout
        )
        
        # Final prediction layer
        self.final_layer = nn.Linear(512, num_vertices * vertex_dim)
        
        # Multiple residual connections for better fitting
        self.residual_proj1 = nn.Linear(global_feature_dim, 1024)
        self.residual_proj2 = nn.Linear(global_feature_dim, 512)
        
    def forward(self, global_features):
        # Enhanced forward pass with multiple residual connections
        x = self.vertex_mlp1(global_features)
        x = self.vertex_mlp2(x)
        
        # First residual connection
        residual1 = self.residual_proj1(global_features)
        x = self.vertex_mlp3(x) + residual1
        
        # Second residual connection
        residual2 = self.residual_proj2(global_features)
        x = self.vertex_mlp4(x) + residual2
        
        # Final prediction
        vertex_coords = self.final_layer(x)
        vertex_coords = vertex_coords.view(-1, self.num_vertices, self.vertex_dim)
        return vertex_coords