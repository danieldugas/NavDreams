import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
from strictfire import StrictFire

_RS = 5
_H = 64
encoder_types = ["E2E", "N3D", "sequenceN3D"]
N_CLASSES = 6

def labels_to_onehot(labels):
    pass

class UnFlatten(nn.Module):
    def __init__(self, channels):
        super(UnFlatten, self).__init__()
        self.channels = channels

    def forward(self, input):
        return input.view(input.size(0), self.channels, 1, 1)

class Segmenter(nn.Module):
    def __init__(self, image_channels=1, fc_dim=1024, z_dim=64, gpu=True):
        self.gpu = gpu
        super(Segmenter, self).__init__()
        self.fc3 = nn.Linear(z_dim, fc_dim)

        self.decoder = nn.Sequential(
            UnFlatten(fc_dim),
            nn.ConvTranspose2d(fc_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x, labels=None):
        x = self.fc3(x)
        img = self.decoder(x)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy(img, labels)  # input-reconstruction loss
        return img, loss

def main(dry_run=False):
    archive_dir = os.path.expanduser("~/navrep3d_W/datasets/multitask/navrep3dalt_segmentation")

    for encoder_type in encoder_types:
        filenames = []
        for dirpath, dirnames, dirfilename in os.walk(archive_dir):
            for filename in [
                f
                for f in dirfilename
                if f.endswith("{}encodings_labels.npz".format(encoder_type))
            ]:
                filenames.append(os.path.join(dirpath, filename))
        for archive_file in filenames:
            archive_path = os.path.join(archive_dir, archive_file)
            data = np.load(archive_path)
            print("{} loaded.".format(archive_path))
            encodings = data["encodings"]
            labels = data["labels"]
            depths = data["depths"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]
            robotstates = data["robotstates"]

        model = Segmenter(N_CLASSES)
        is_train = True
        grad_norm_clip = 1.0
        losses = []

        # take over whatever gpus are on the system
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            model = torch.nn.DataParallel(model).to(device)

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ]
        weight_decay = 0.1  # only applied on matmul weights
        optim_groups = [
            {"params": params_decay, "weight_decay": weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(optim_groups)

        # place data on the correct device
        x = torch.tensor(encodings, dtype=torch.float)
        B, W, H, CH = labels.shape
        y_oh = F.one_hot(torch.tensor(labels[:, :, :, 2], dtype=torch.int64))
        y = torch.tensor(np.moveaxis(y_oh.detach().cpu().numpy(), -1, 1), dtype=torch.float)
        x = x.to(device)
        y = y.to(device)

        # forward the model
        with torch.set_grad_enabled(is_train):
            y_pred, loss = model(x, labels=y)
            loss = (
                loss.mean()
            )  # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())

        if is_train:
            # backprop and update the parameters
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            optimizer.step()

    globals().update(locals())


if __name__ == "__main__":
    StrictFire(main)
