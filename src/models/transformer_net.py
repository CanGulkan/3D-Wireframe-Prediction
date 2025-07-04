import torch
import torch.nn as nn

class TransformerWireframeNet(nn.Module):
    def __init__(self, E_pred, d_model=128, num_heads=8, num_layers=4):
        super().__init__()
        self.E_pred = E_pred

        # Point embedding
        self.pc_embed = nn.Linear(3, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable edge queries
        self.edge_queries = nn.Parameter(torch.randn(E_pred, d_model))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads
        self.edge_proj = nn.Linear(d_model, 6)  # [2x3] per edge
        self.conf_proj = nn.Linear(d_model, 1)  # confidence
        self.quad_proj = nn.Linear(d_model, 4)  # quadrant logits

    def forward(self, pc):
        # pc: [N, 3]
        x = self.pc_embed(pc)  # [N, d]
        x = self.encoder(x)    # [N, d]

        queries = self.edge_queries.unsqueeze(1)       # [E_pred, 1, d]
        memory = x.unsqueeze(1)                        # [N, 1, d]
        decoded = self.decoder(queries, memory)        # [E_pred, 1, d]
        decoded = decoded.squeeze(1)                   # [E_pred, d]

        edges = self.edge_proj(decoded).view(-1, 2, 3)
        conf = self.conf_proj(decoded).squeeze(-1)
        quad = self.quad_proj(decoded)

        return edges, conf, quad
