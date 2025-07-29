import torch.nn as nn
import torch

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class BWFormer(nn.Module):
    def __init__(self, E_pred=50, d_model=128, nhead=4, num_layers=2):  # Reduced sizes
        super().__init__()
        self.E_pred = E_pred

        # More efficient encoder
        self.encoder = nn.Sequential(
            nn.Linear(3, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

        # Transformer with batch_first and reduced layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=d_model*2  # Reduced from typical 4x
            ),
            num_layers=num_layers
        )

        # Heads with reduced capacity
        self.edge_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, E_pred * 2 * 3)
        )
        
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, E_pred),
            nn.Sigmoid()
        )

    def forward(self, pc):
        # Input: [N_points, 3]
        x = self.encoder(pc)  # [N, d_model]
        
        # Transformer expects [batch, seq, features]
        x = x.unsqueeze(0)  # [1, N, d_model]
        
        # Process in chunks if needed
        x = self.transformer(x)
        
        # Pool
        global_feat = x.mean(dim=1)  # [1, d_model]
        
        # Predictions
        edges = torch.tanh(self.edge_head(global_feat)).view(-1, 2, 3)
        conf = self.conf_head(global_feat).squeeze(0)
        
        return edges[:self.E_pred], conf[:self.E_pred]  # Ensure correct size