import torch
from torch import nn

class MultiTaskHeads(nn.Module):
    """Heads consume fused features; make/model/year label maps must be supplied by dataset metadata."""
    def __init__(self, in_features: int, num_classes: int = 196, num_makes: int = 0, num_models: int = 0, num_years: int = 0):
        super().__init__()
        self.vehicle196=nn.Linear(in_features, num_classes)
        self.make=nn.Linear(in_features, num_makes) if num_makes else None
        self.model=nn.Linear(in_features, num_models) if num_models else None
        self.year=nn.Linear(in_features, num_years) if num_years else None
    def forward(self, x):
        out={"vehicle196": self.vehicle196(x)}
        for name in ("make", "model", "year"):
            head=getattr(self, name)
            if head is not None: out[name]=head(x)
        return out
