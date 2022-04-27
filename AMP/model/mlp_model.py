import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x