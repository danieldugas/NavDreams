import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import typer
from pyniel.python_tools.path_tools import make_dir_if_not_exists
from navrep.models.gpt import save_checkpoint

from multitask_encode_dataset import encoder_types

_RS = 5
_H = 64
N_CLASSES = 6

def onehot_to_rgb(labels):
    W, H, CH = labels.shape
    colors = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1],
                       [1, 0, 1]], dtype=float)
    indices = np.argmax(labels, axis=-1)
    return colors[indices]

class UnFlatten(nn.Module):
    def __init__(self, channels):
        super(UnFlatten, self).__init__()
        self.channels = channels

    def forward(self, input):
        return input.view(input.size(0), self.channels, 1, 1)

class TaskLearner(nn.Module):
    def __init__(self, image_channels, label_is_onehot=True, fc_dim=1024, z_dim=64, gpu=True):
        self.gpu = gpu
        if label_is_onehot:
            self.loss_func = F.binary_cross_entropy
        else:
            self.loss_func = F.mse_loss
        super(TaskLearner, self).__init__()
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
#         B, Z = x.shape
#         B, CH, W, H = labels.shape
        x = self.fc3(x)
        img = self.decoder(x)

        loss = None
        if labels is not None:
            loss = self.loss_func(img, labels)  # input-reconstruction loss
        return img, loss

class MultitaskDataset(Dataset):
    def __init__(self, directory, task="segmentation",
                 filename_mask="encodings_labels.npz",
                 file_limit=None,
                 channel_first=True, as_torch_tensors=True,
                 ):
        self.channel_first = channel_first
        self.as_torch_tensors = as_torch_tensors
        self.task = task
        self.data = self._load_data(directory, filename_mask, file_limit=file_limit)
        size = self.__len__()
        if size == 0:
            raise ValueError
        print("data has %d steps." % size)

    def _load_data(self, directory, filename_mask, file_limit=None):
        # list all data files
        files = []
        if isinstance(directory, list):
            directories = directory
        elif isinstance(directory, str):
            directories = [directory]
        else:
            raise NotImplementedError
        for dir_ in directories:
            dir_ = os.path.expanduser(dir_)
            for dirpath, dirnames, filenames in os.walk(dir_):
                for filename in [
                    f
                    for f in filenames
                    if f.endswith(filename_mask)
                ]:
                    files.append(os.path.join(dirpath, filename))
        files = sorted(files)
        if file_limit is None:
            file_limit = len(files)
        data = {
            "encodings": [],
            "labels": [],
            "depths": [],
            "robotstates": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        arrays_dict = {}
        for path in files[:file_limit]:
            arrays_dict = np.load(path)
            for k in arrays_dict.keys():
                if k == "modelpath":
                    continue
                data[k].append(arrays_dict[k])
        for k in arrays_dict.keys():
            if k == "modelpath":
                continue
            data[k] = np.concatenate(data[k], axis=0)
        return data

    def _convert_obs(self, labels, depths):
        # labels to one-hot, then move channel to first axis
        ohlabels = F.one_hot(torch.tensor(labels[:, :, 2], dtype=torch.int64), num_classes=N_CLASSES)
        ohlabels = np.moveaxis(ohlabels.detach().cpu().numpy(), -1, 0).astype(float)
        depths01 = depths.astype(float) / 256.
        depths01 = np.moveaxis(depths01, -1, 0)
        return ohlabels, depths01

    def __len__(self):
        return len(self.data["encodings"])

    def __getitem__(self, idx):
        encodings = self.data["encodings"][idx]
        ohlabels, depths01 = self._convert_obs(self.data["labels"][idx] , self.data["depths"][idx])
        # outputs
        x = encodings
        if self.task == "segmentation":
            y = ohlabels
        elif self.task == "depth":
            y = depths01
        else:
            raise NotImplementedError
        # torch
        if self.as_torch_tensors:
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
        return x, y

def train_multitask(encoder_type, task="segmentation"):
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    if task == "segmentation":
        log_path = os.path.expanduser(
            "~/navrep3d/logs/multitask/{}_segmenter_train_log_{}.csv".format(encoder_type, START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d/models/multitask/{}_segmenter_{}".format(
            encoder_type, START_TIME))
        plot_path = os.path.expanduser("~/tmp_navrep3d/{}_segmenter_step".format(encoder_type))
    elif task == "depth":
        log_path = os.path.expanduser(
            "~/navrep3d/logs/multitask/{}_depth_train_log_{}.csv".format(encoder_type, START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d/models/multitask/{}_depth_{}".format(
            encoder_type, START_TIME))
        plot_path = os.path.expanduser("~/tmp_navrep3d/{}_depth_step".format(encoder_type))

    archive_dir = os.path.expanduser("~/navrep3d_W/datasets/multitask/navrep3dalt_segmentation")
    filename_mask = "{}encodings_labels.npz".format(encoder_type)
    make_dir_if_not_exists(os.path.dirname(checkpoint_path))
    make_dir_if_not_exists(os.path.dirname(log_path))
    make_dir_if_not_exists(os.path.expanduser("~/tmp_navrep3d"))

    full_dataset = MultitaskDataset(archive_dir, task=task, filename_mask=filename_mask)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    label_is_onehot = task == "segmentation"
    channels = N_CLASSES if label_is_onehot else 3

    model = TaskLearner(channels, label_is_onehot=label_is_onehot)
    print("trainable params: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # training params
    # optimization parameters
    max_steps = 100000
    PLOT_EVERY_N_STEPS = 1000
    max_epochs = max_steps  # don't stop based on epoch
    grad_norm_clip = 1.0

    # take over whatever gpus are on the system
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = torch.nn.DataParallel(model).to(device)

    # optimizer
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    global_step = 0
    values_logs = None
    start = time.time()
    for epoch in range(max_epochs):
        is_train = True
        model.train(is_train)
        loader = DataLoader(
            train_dataset,
            shuffle=is_train,
            batch_size=128,
            num_workers=8,
        )

        epoch_losses = []

        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (x, y) in pbar:
            global_step += 1

            # place data on the correct device
            x = x.to(device)
            y = y.to(device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                y_pred, loss = model(x, labels=y)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                epoch_losses.append(loss.item())

            if is_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step()

                # report progress
                pbar.set_description(
                    f"{encoder_type} epoch {epoch}: train loss {np.mean(epoch_losses):.5f}"
                )

                if global_step == 1 or global_step % PLOT_EVERY_N_STEPS == 0:
                    # save model
                    save_checkpoint(model, checkpoint_path)
                    # save plot
                    from matplotlib import pyplot as plt
                    plt.figure("training_status")
                    plt.clf()
                    plt.suptitle("training step {}".format(global_step))
                    f, axes = plt.subplots(2, 5, num="training_status", sharex=True, sharey=True)
                    for i, (ax0, ax1) in enumerate(axes.T):
                        if label_is_onehot:
                            ax0.imshow(onehot_to_rgb(np.moveaxis(y.cpu().numpy()[i], 0, -1)))
                            ax1.imshow(onehot_to_rgb(np.moveaxis(y_pred.detach().cpu().numpy()[i], 0, -1)))
                        else:
                            ax0.imshow(np.moveaxis(y.cpu().numpy()[i], 0, -1))
                            ax1.imshow(np.moveaxis(y_pred.detach().cpu().numpy()[i], 0, -1))
                    plt.savefig(plot_path + "{:07}.png".format(global_step))

        # Validation error
        is_train = False
        model.train(is_train)
        loader = DataLoader(
            test_dataset,
            shuffle=is_train,
            batch_size=128,
            num_workers=0,
        )
        epoch_losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (x, y) in pbar:
            # place data on the correct device
            x = x.to(device)
            y = y.to(device)
            # forward the model
            with torch.set_grad_enabled(is_train):
                y_pred, loss = model(x, labels=y)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                epoch_losses.append(loss.item())
                pbar.set_description(f"eval loss {np.mean(epoch_losses):.5f}")
        test_error = np.mean(epoch_losses)

        # log
        end = time.time()
        time_taken = end - start
        start = time.time()
        values_log = pd.DataFrame(
            [[global_step, np.mean(epoch_losses), test_error, time_taken]],
            columns=["step", "epoch_loss", "test_error", "train_time_taken"],
        )
        if values_logs is None:
            values_logs = values_log.copy()
        else:
            values_logs = values_logs.append(values_log, ignore_index=True)
        if log_path is not None:
            values_logs.to_csv(log_path)

        if global_step >= max_steps:
            break

def main():
    for encoder_type in encoder_types:
        train_multitask(encoder_type, task="depth")
    for encoder_type in encoder_types:
        train_multitask(encoder_type, task="segmentation")


if __name__ == "__main__":
    typer.run(main)
