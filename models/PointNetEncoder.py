
import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    """Massive PointNet encoder for total overfitting"""
    
    def __init__(self, input_dim=8, hidden_dims=[256, 512, 1024], output_dim=512):
        super(PointNetEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.0)  # No dropout for total overfitting
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Enhanced pooling - combine max and mean
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Larger feature fusion for more capacity
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_points, input_dim)
        batch_size, num_points, input_dim = x.shape
        
        # Reshape for MLP processing
        x = x.view(-1, input_dim)  # (batch_size * num_points, input_dim)
        
        # Apply MLP to each point
        point_features = self.mlp(x)  # (batch_size * num_points, output_dim)
        
        # Reshape back
        point_features = point_features.view(batch_size, num_points, -1)
        
        # Enhanced global pooling - combine max and mean
        # Transpose for pooling: (batch_size, output_dim, num_points)
        point_features_t = point_features.transpose(1, 2)
        
        # Max and average pooling
        max_features = self.global_max_pool(point_features_t).squeeze(-1)
        avg_features = self.global_avg_pool(point_features_t).squeeze(-1)
        
        # Combine features with more processing
        combined_features = torch.cat([max_features, avg_features], dim=1)
        global_features = self.feature_fusion(combined_features)
        
        return global_features, point_features