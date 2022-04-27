import torch.nn as nn
import torch
from .conv_lstm import ConvLSTM
from .mlp_model import MLP

class WorldDisc(nn.Module):
    def __init__(self):
        super(WorldDisc, self).__init__()
        hidden_dim = 8
        self.ConvLSTM = ConvLSTM(input_dim=3,
                 hidden_dim=hidden_dim,
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            MLP(128,64),
            nn.Linear(64,1)
        )


    def forward(self, input_tensor):
        b_s = input_tensor.shape[0]
        _, last_states = self.ConvLSTM(input_tensor)
        feat = last_states[0][0]
        feat = nn.Flatten()(self.conv(feat))
        out = self.classifier(feat)
        return out