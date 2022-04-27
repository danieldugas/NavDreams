import torch.nn as nn

from .mlp_model import MLP

class imageDisc(nn.Module):
    def __init__(self):
        super(imageDisc, self).__init__()
        hidden_dim = 3

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
        b_s, t_s, c, h, w = input_tensor.shape
        input_tensor = input_tensor.view(b_s*t_s,c,h,w)
        feat = nn.Flatten()(self.conv(input_tensor))
        out = self.classifier(feat)
        return out