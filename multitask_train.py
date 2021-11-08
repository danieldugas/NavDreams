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

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, channels):
        super(UnFlatten, self).__init__()
        self.channels = channels

    def forward(self, input):
        return input.view(input.size(0), self.channels, 1, 1)

class TaskLearner(nn.Module):
    def __init__(self, task_channels, from_image, label_is_onehot,
                 fc_dim=1024, z_dim=64, gpu=True, ):
        self.gpu = gpu
        self.from_image = from_image
        super(TaskLearner, self).__init__()

        # only used for baseline - train encoder + decoder
        if self.from_image:
            input_channels = 3
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2),
                nn.ReLU(),
                Flatten(),
            )
            self.fc1 = nn.Linear(fc_dim, z_dim)
            self.fc2 = nn.Linear(fc_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, fc_dim)

        self.decoder = nn.Sequential(
            UnFlatten(fc_dim),
            nn.ConvTranspose2d(fc_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, task_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        if label_is_onehot:
            self.loss_func = F.binary_cross_entropy
        else:
            self.loss_func = F.mse_loss

    def forward(self, x, labels=None):
        if self.from_image:
#             B, _3, W, H = x.shape
            h = self.encoder(x)
            mu, logvar = self.fc1(h), self.fc2(h)
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            if self.gpu:
                eps = torch.cuda.FloatTensor(*mu.size()).normal_()
            else:
                eps = torch.FloatTensor(*mu.size()).normal_()
            x = mu + std * eps

#         B, Z = x.shape
#         B, CH, W, H = labels.shape
        x = self.fc3(x)
        img = self.decoder(x)

        loss = None
        if labels is not None:
            loss = self.loss_func(img, labels)  # input-reconstruction loss
        return img, loss

class MultitaskDataset(Dataset):
    def __init__(self, directory, task, from_image, filename_mask,
                 file_limit=None,
                 channel_first=True, as_torch_tensors=True,
                 ):
        self.from_image = from_image
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
        if self.from_image:
            data = {
                "images": [],
                "labels": [],
                "depths": [],
                "robotstates": [],
                "actions": [],
                "rewards": [],
                "dones": [],
            }
        else:
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
            for k in data.keys():
                data[k].append(arrays_dict[k])
        for k in data.keys():
            data[k] = np.concatenate(data[k], axis=0)
        return data

    def _convert_obs(self, labels, depths):
        # labels to one-hot, then move channel to first axis
        W, H, CH = labels.shape # 0-255
        W, H, CH = depths.shape # 0-255
        ohlabels = F.one_hot(torch.tensor(labels[:, :, 2], dtype=torch.int64), num_classes=N_CLASSES)
        ohlabels = np.moveaxis(ohlabels.detach().cpu().numpy(), -1, 0).astype(float)
        depths01 = (depths[:, :, 0] / 256.
                    + depths[:, :, 1] / 256. / 256.
                    + depths[:, :, 2] / 256. / 256. / 256.).astype(float).reshape((1, W, H))
        return ohlabels, depths01

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
        if self.from_image:
            x = self.data["images"][idx] / 255.
            x = np.moveaxis(x, -1, 0)
        else:
            x = self.data["encodings"][idx]
        ohlabels, depths01 = self._convert_obs(self.data["labels"][idx] , self.data["depths"][idx])
        # outputs
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

def validate(model, test_dataset, device):
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
    model.train(True)
    return test_error

def train_multitask(encoder_type, task="segmentation", dry_run=False, gpu=True):
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
    if dry_run:
        log_path.replace(os.path.expanduser("~"), "/tmp")
        checkpoint_path.replace(os.path.expanduser("~"), "/tmp")
        plot_path.replace(os.path.expanduser("~"), "/tmp")
    from_image = encoder_type == "baseline"

    archive_dir = os.path.expanduser("~/navrep3d_W/datasets/multitask/navrep3dalt_segmentation")
    if from_image:
        filename_mask = "_images_labels.npz"
    else:
        filename_mask = "_{}encodings_labels.npz".format(encoder_type)
    make_dir_if_not_exists(os.path.dirname(checkpoint_path))
    make_dir_if_not_exists(os.path.dirname(log_path))
    make_dir_if_not_exists(os.path.expanduser("~/tmp_navrep3d"))

    full_dataset = MultitaskDataset(archive_dir, task, from_image, filename_mask)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))
    label_is_onehot = task == "segmentation"
    task_channels = N_CLASSES if label_is_onehot else 1

    model = TaskLearner(task_channels, from_image, label_is_onehot, gpu=gpu)
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
    losses = []
    for epoch in range(max_epochs):
        is_train = True
        model.train(is_train)
        loader = DataLoader(
            train_dataset,
            shuffle=is_train,
            batch_size=128,
            num_workers=8,
        )

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
                losses.append(loss.item())

            if is_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step()

                # report progress
                pbar.set_description(
                    f"{encoder_type} epoch {epoch}: train loss {np.mean(losses):.5f}"
                )

                if global_step == 1 or global_step % PLOT_EVERY_N_STEPS == 0:
                    # save model
                    save_checkpoint(model, checkpoint_path)
                    # save plot
                    from matplotlib import pyplot as plt
                    plt.figure("training_status")
                    plt.clf()
                    plt.suptitle("training step {}".format(global_step))
                    if encoder_type == "baseline":
                        f, axes = plt.subplots(3, 5, num="training_status", sharex=True, sharey=True)
                        axes = axes.reshape((3, 5))
                    else:
                        f, axes = plt.subplots(2, 5, num="training_status", sharex=True, sharey=True)
                        axes = axes.reshape((2, 5))
                    for i, axrow in enumerate(axes.T):
                        if encoder_type == "baseline":
                            ax0, ax1, ax2 = axrow
                            ax0.imshow(np.moveaxis(x.cpu().numpy()[i], 0, -1))
                        else:
                            ax1, ax2 = axrow
                        if label_is_onehot:
                            ax1.imshow(onehot_to_rgb(np.moveaxis(y.cpu().numpy()[i], 0, -1)))
                            ax2.imshow(onehot_to_rgb(np.moveaxis(y_pred.detach().cpu().numpy()[i], 0, -1)))
                        else:
                            ax1.imshow(np.moveaxis(y.cpu().numpy()[i], 0, -1))
                            ax2.imshow(np.moveaxis(y_pred.detach().cpu().numpy()[i], 0, -1))
                    plt.savefig(plot_path + "{:07}.png".format(global_step))
                    # log
                    end = time.time()
                    test_error = validate(model, test_dataset, device)
                    time_taken = end - start
                    start = time.time()
                    values_log = pd.DataFrame(
                        [[global_step, np.mean(losses), test_error, time_taken]],
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

def main(dry_run : bool = False, gpu : bool = True):
    for encoder_type in encoder_types + ["baseline"]:
        train_multitask(encoder_type, task="depth", dry_run=dry_run, gpu=gpu)
    for encoder_type in encoder_types + ["baseline"]:
        train_multitask(encoder_type, task="segmentation", dry_run=dry_run, gpu=gpu)


if __name__ == "__main__":
    typer.run(main)
