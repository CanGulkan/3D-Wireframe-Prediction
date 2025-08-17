import torch
import torch.nn as nn


class WireframeLoss(nn.Module):
    """Combined loss for vertex position and edge connectivity"""
    
    def __init__(self, vertex_weight=1.0, edge_weight=1.0):
        super(WireframeLoss, self).__init__()
        self.vertex_weight = vertex_weight
        self.edge_weight = edge_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions, targets):
        # Vertex position loss
        pred_vertices = predictions['vertices']
        target_vertices = targets['vertices']
        vertex_loss = self.mse_loss(pred_vertices, target_vertices)
        
        # Edge connectivity loss
        pred_edge_probs = predictions['edge_probs']
        target_edge_labels = targets['edge_labels']
        edge_loss = self.bce_loss(pred_edge_probs, target_edge_labels)
        
        # Combined loss
        total_loss = self.vertex_weight * vertex_loss + self.edge_weight * edge_loss
        
        return {
            'total_loss': total_loss,
            'vertex_loss': vertex_loss,
            'edge_loss': edge_loss
        }
