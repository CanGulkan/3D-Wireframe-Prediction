import numpy as np
import torch
from torch.utils.data import Dataset

class WireframeTransform:
    """Apply random transformations to point cloud and wireframe"""
    def __init__(self, noise_std=0.01, max_rotation=15, min_scale=0.9, max_scale=1.1):
        self.noise_std = noise_std
        self.max_rotation = max_rotation
        self.scale_range = (min_scale, max_scale)
    
    def __call__(self, pc, gt_edges):
        # Random rotation
        angle = np.random.uniform(-self.max_rotation, self.max_rotation)
        rot_mat = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Random scaling
        scale = np.random.uniform(*self.scale_range)
        
        # Apply transformations
        pc = (pc @ rot_mat) * scale
        gt_edges = (gt_edges @ rot_mat) * scale
        
        # Add noise only to point cloud
        if self.noise_std > 0:
            pc += torch.randn_like(pc) * self.noise_std
            
        return pc, gt_edges
