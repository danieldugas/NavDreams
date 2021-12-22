import os
import matplotlib
if os.path.expandvars("$MACHINE_NAME") in ["leonhard", "euler"]:
    matplotlib.use('agg')
import logging
import os
import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from pyniel.python_tools.path_tools import make_dir_if_not_exists
from strictfire import StrictFire

from navrep.models.gpt import GPT, GPTConfig, save_checkpoint, set_seed
from navrep.tools.wdataset import WorldModelDataset, scans_to_lidar_obs
from navrep.tools.test_worldmodel import mse

from navrep3d.auto_debug import enable_auto_debug

def gpt_worldmodel_error(gpt, test_dataset_folder, device):
    sequence_size = gpt.module.block_size
    batch_size = 128
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_size, lidar_mode="images",
                                   channel_first=True, as_torch_tensors=True, file_limit=64)
    batch_loader = DataLoader(seq_loader, shuffle=False, batch_size=batch_size)
    # iterate over batches
    batch_loader = tqdm(batch_loader, total=len(batch_loader))
    n_batches = 0
    sum_state_error = 0
    sum_lidar_error = 0
    for x, a, y, x_rs, y_rs, dones in batch_loader:

        # place data on the correct device
        x = x.to(device)
        x_rs = x_rs.to(device)
        a = a.to(device)
        y = y.to(device)
        y_rs = y_rs.to(device)
        dones = dones.to(device)

        y_pred_rec, y_rs_pred, _ = gpt(x, x_rs, a, dones)
        y_pred_rec = y_pred_rec.detach().cpu().numpy()
        y_rs_pred = y_rs_pred.detach().cpu().numpy()

        sum_lidar_error += mse(y_pred_rec, y.cpu().numpy())  # because binary cross entropy is inf for 0
        sum_state_error += mse(y_rs_pred, y_rs.cpu().numpy())  # mean square error loss
        n_batches += 1
    lidar_error = sum_lidar_error / n_batches
    state_error = sum_state_error / n_batches
    return lidar_error, state_error


_Z = _H = 64
_S = 32  # sequence length

class N3DWorldModelDataset(WorldModelDataset):
    """ same as a WorldModelDataset, but data regeneration is specialized for navrep3d """
    def _partial_regen(self, n_new_sequences=1, build_name=None):
        from navrep.scripts.make_vae_dataset import generate_vae_dataset, SemiRandomMomentumPolicy
        from navrep3d.navrep3danyenv import NavRep3DAnyEnv
        if self.regen in ["S", "SC", "Salt", "SCR", "R"]:
            if build_name is None:
                if self.regen == "S":
                    build_name = "./build.x86_64"
                elif self.regen == "Salt":
                    build_name = "./alternate.x86_64"
                elif self.regen == "SC":
                    build_names = ["./alternate.x86_64", "./city.x86_64", "./office.x86_64"]
                    build_name = np.random.choice(build_names)
                elif self.regen == "SCR":
                    build_names = [
                        "./alternate.x86_64", "./city.x86_64", "./office.x86_64", "staticasl", "rosbag"]
                    build_name = np.random.choice(build_names)
                elif self.regen == "R":
                    build_names = ["staticasl", "rosbag"]
                    build_name = np.random.choice(build_names)
                else:
                    raise NotImplementedError
            try:
                env = NavRep3DAnyEnv(verbose=0, collect_statistics=False,
                                     build_name=build_name, port=25005+np.random.randint(10),
                                     tolerate_corruption=False, randomize_difficulty=True)
                policy = SemiRandomMomentumPolicy()
                data = generate_vae_dataset(
                    env, n_sequences=n_new_sequences, policy=policy,
                    render=False, archive_dir=None)
            except: # noqa
                print("Failed to regenerate dataset {}. retrying.".format(build_name))
                self._partial_regen(n_new_sequences=n_new_sequences, build_name=build_name)
                return
            if self.pre_convert_obs:
                data["obs"] = scans_to_lidar_obs(
                    data["scans"], self.lidar_mode, self.rings_def, self.channel_first)
        else:
            print("Regen {} failed".format(self.regen))
            return
        for k in self.data.keys():
            N = len(data[k])  # should be the same for each key
            # check end inside loop to avoid having to pick an arbitrary key
            if self.regen_head_index + N > len(self.data[k]):
                self.regen_head_index = 0
            # replace data
            i = self.regen_head_index
            self.data[k][i : i + N] = data[k]
        print("Regenerated {} steps of {}".format(N, build_name))
        self.regen_head_index += N


def main(max_steps=222222, dataset="S", dry_run=False):
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    if dataset == "S":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dtrain")]
        log_path = os.path.expanduser(
            "~/navrep3d_W/logs/W/transformer_S_train_log_{}.csv".format(START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d_W/models/W/transformer_S")
        plot_path = os.path.expanduser("~/tmp_navrep3d/transformer_S_step")
    elif dataset == "Salt":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dalt")]
        log_path = os.path.expanduser(
            "~/navrep3d_W/logs/W/transformer_Salt_train_log_{}.csv".format(START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d_W/models/W/transformer_Salt")
        plot_path = os.path.expanduser("~/tmp_navrep3d/transformer_Salt_step")
    elif dataset == "SC":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dtrain"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3doffice")]
        log_path = os.path.expanduser(
            "~/navrep3d_W/logs/W/transformer_SC_train_log_{}.csv".format(START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d_W/models/W/transformer_SC")
        plot_path = os.path.expanduser("~/tmp_navrep3d/transformer_SC_step")
    elif dataset == "Random":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dalt")]
        log_path = os.path.expanduser(
            "~/navrep3d_W/logs/W/transformer_Random_train_log_{}.csv".format(START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d_W/models/W/transformer_Random")
        plot_path = os.path.expanduser("~/tmp_navrep3d/transformer_Random_step")
        max_steps = 0
    elif dataset == "SCR":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dalt"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3doffice"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")]
        log_path = os.path.expanduser(
            "~/navrep3d_W/logs/W/transformer_SCR_train_log_{}.csv".format(START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d_W/models/W/transformer_SCR")
        plot_path = os.path.expanduser("~/tmp_navrep3d/transformer_SCR_step")
    elif dataset == "R":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")]
        log_path = os.path.expanduser(
            "~/navrep3d_W/logs/W/transformer_R_train_log_{}.csv".format(START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d_W/models/W/transformer_R")
        plot_path = os.path.expanduser("~/tmp_navrep3d/transformer_R_step")
    else:
        raise NotImplementedError(dataset)

    if dry_run:
        log_path = log_path.replace(os.path.expanduser("~/navrep3d"), "/tmp/navrep3d")
        checkpoint_path = checkpoint_path.replace(os.path.expanduser("~/navrep3d"), "/tmp/navrep3d")

    make_dir_if_not_exists(os.path.dirname(checkpoint_path))
    make_dir_if_not_exists(os.path.dirname(log_path))
    make_dir_if_not_exists(os.path.expanduser("~/tmp_navrep3d"))

    # make deterministic
    set_seed(42)

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    mconf = GPTConfig(_S, _H)
    mconf.image_channels = 3
    train_dataset = N3DWorldModelDataset(
        dataset_dir, _S,
        pre_convert_obs=False,
        regen=dataset,
        lidar_mode="images",
    )
    if dry_run:
        train_dataset._partial_regen()

    # training params
    # optimization parameters
    max_epochs = max_steps  # don't stop based on epoch
    batch_size = 128
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    lr_decay = True  # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    weight_decay = 0.1  # only applied on matmul weights
    warmup_tokens = 512 * 20
    final_tokens = 200 * len(train_dataset) * _S
    num_workers = 0  # for DataLoader

    # create model
    model = GPT(mconf)
    print("GPT trainable params: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # increase stddev in random model weights
    if dataset == "Random":
        def randomize_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.2)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
        model.apply(randomize_weights)

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
    optim_groups = [
        {"params": params_decay, "weight_decay": weight_decay},
        {"params": params_nodecay, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    global_step = 0
    tokens = 0  # counter used for learning rate decay
    values_logs = None
    start = time.time()
    for epoch in range(max_epochs):
        is_train = True
        model.train(is_train)
        loader = DataLoader(
            train_dataset,
            shuffle=is_train,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (x, a, y, x_rs, y_rs, dones) in pbar:
            global_step += 1

            # place data on the correct device
            x = x.to(device)
            x_rs = x_rs.to(device)
            a = a.to(device)
            y = y.to(device)
            y_rs = y_rs.to(device)
            dones = dones.to(device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                y_pred, y_rs_pred, loss = model(x, x_rs, a, dones, targets=(y, y_rs))
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

                # decay the learning rate based on our progress
                if lr_decay:
                    tokens += (
                        a.shape[0] * a.shape[1]
                    )  # number of tokens processed this step
                    if tokens < warmup_tokens:
                        # linear warmup
                        lr_mult = float(tokens) / float(max(1, warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(tokens - warmup_tokens) / float(
                            max(1, final_tokens - warmup_tokens)
                        )
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    lr = learning_rate

                # report progress
                pbar.set_description(
                    f"epoch {epoch}: train loss {loss.item():.5f}. lr {lr:e}"
                )

                if global_step == 1 or global_step % 1000 == 0:
                    # save plot
                    from matplotlib import pyplot as plt
                    plt.figure("training_status")
                    plt.clf()
                    plt.suptitle("training step {}".format(global_step))
                    f, axes = plt.subplots(3, 5, num="training_status", sharex=True, sharey=True)
                    for i, (ax0, ax1, ax2) in enumerate(axes.T):
                        ax0.imshow(np.moveaxis(x.cpu().numpy()[0, 5 + i], 0, -1))
                        ax1.imshow(np.moveaxis(y.cpu().numpy()[0, 5 + i], 0, -1))
                        ax2.imshow(np.moveaxis(y_pred.detach().cpu().numpy()[0, 5 + i], 0, -1))
                        ax2.set_xlabel("Done {}".format(dones.cpu()[0, 5 + 1]))
                    plt.savefig(plot_path + "{:07}.png".format(global_step))

        lidar_e = None
        state_e = None
        if epoch % 20 == 0:
            lidar_e, state_e = gpt_worldmodel_error(model, dataset_dir, device)
            save_checkpoint(model, checkpoint_path)

        # log
        end = time.time()
        time_taken = end - start
        start = time.time()
        values_log = pd.DataFrame(
            [[global_step, loss.item(), lidar_e, state_e, time_taken]],
            columns=["step", "cost", "lidar_test_error", "state_test_error", "train_time_taken"],
        )
        if values_logs is None:
            values_logs = values_log.copy()
        else:
            values_logs = values_logs.append(values_log, ignore_index=True)
        if log_path is not None:
            values_logs.to_csv(log_path)

        if not is_train:
            logger.info("test loss: %f", np.mean(losses))

        if global_step >= max_steps:
            break

    print("Final evaluation")
    lidar_e, state_e = gpt_worldmodel_error(model, dataset_dir, device)
    save_checkpoint(model, checkpoint_path)


if __name__ == "__main__":
    enable_auto_debug()
    StrictFire(main)
