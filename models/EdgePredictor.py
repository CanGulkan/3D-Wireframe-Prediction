import torch
import torch.nn as nn

class EdgePredictor(nn.Module):
    """Predict edge connectivity between vertices"""
    
    def __init__(self, vertex_dim=3, hidden_dim=128):
        super(EdgePredictor, self).__init__()
        
        # Edge features are concatenated vertex features
        self.edge_mlp = nn.Sequential(
            nn.Linear(vertex_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vertices):
        # vertices shape: (batch_size, num_vertices, vertex_dim)
        batch_size, num_vertices, vertex_dim = vertices.shape
        
        # Create all possible pairs of vertices
        edges = []
        edge_indices = []
        
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):  # Only upper triangle to avoid duplicate edges
                # Concatenate vertex features
                v1 = vertices[:, i, :]  # (batch_size, vertex_dim)
                v2 = vertices[:, j, :]  # (batch_size, vertex_dim)
                edge_feature = torch.cat([v1, v2], dim=1)  # (batch_size, vertex_dim * 2)
                edges.append(edge_feature)
                edge_indices.append((i, j))
        
        # Stack all edge features
        edge_features = torch.stack(edges, dim=1)  # (batch_size, num_edges, vertex_dim * 2)
        
        # Reshape for MLP processing
        batch_size, num_edges = edge_features.shape[:2]
        edge_features = edge_features.view(-1, vertex_dim * 2)
        
        # Predict edge probabilities
        edge_probs = self.edge_mlp(edge_features)  # (batch_size * num_edges, 1)
        edge_probs = edge_probs.view(batch_size, num_edges)
        
        return edge_probs, edge_indices
