import torch
import torch.nn as nn                                      
import torch.nn.functional as F     

# ─── Part 4: Dummy Model for Testing ────────────────────────────────────────

class DummyWireframeNet(torch.nn.Module):
    """
    A dummy network that learns to regress fixed-length edges from point cloud input.
    """
    def __init__(self, E_pred):
        super().__init__()
        self.E_pred = E_pred
        self.fc = torch.nn.Linear(3, 3)  # dummy linear mapping

    def forward(self, pc):
        pts = pc[: self.E_pred * 2]     # select first 2*E_pred points
        out = self.fc(pts)              # apply linear layer
        edges = out.view(self.E_pred, 2, 3)  # reshape into edges
        conf = torch.zeros(self.E_pred, device=pc.device)       # dummy confidence
        quad = torch.zeros(self.E_pred, 4, device=pc.device)    # dummy quadrant logits
        return edges, conf, quad