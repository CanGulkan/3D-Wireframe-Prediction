
import torch.nn as nn  
import torch

class BWFormer(nn.Module):
    def __init__(self, E_pred=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.E_pred = E_pred

        # Point encoder
        self.encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Positional encoding (optional)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )

        # Global token or pooled feature
        self.global_token = nn.Parameter(torch.randn(1, d_model))

        # Heads
        self.edge_head = nn.Linear(d_model, E_pred * 2 * 3)
        self.conf_head = nn.Linear(d_model, E_pred)
        self.quad_head = nn.Linear(d_model, E_pred * 4)

    def forward(self, pc):
        x = self.encoder(pc)                    # [N, d_model]
        x = x.unsqueeze(1)                      # [N, 1, d_model]
        x = x + self.pos_embed[:, :x.size(0)]   # [N, 1, d_model]

        x = x.squeeze(1).permute(1, 0, 2)       # [seq_len, batch, d_model]
        tf_out = self.transformer(x)            # [seq_len, batch, d_model]
        global_feat = tf_out.mean(dim=0)        # [batch, d_model]

        edges = self.edge_head(global_feat).view(self.E_pred, 2, 3)
        conf = self.conf_head(global_feat)
        quad = self.quad_head(global_feat).view(self.E_pred, 4)

        return edges, conf, quad
