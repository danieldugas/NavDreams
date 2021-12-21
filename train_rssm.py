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
from pydreamer.models.dreamer import RSSMCore, MultiDecoder, MultiEncoder, init_weights_tf2, D, logavgexp

from navrep.models.gpt import GPT, GPTConfig, save_checkpoint, set_seed
from navrep.tools.wdataset import WorldModelDataset
from navrep.tools.test_worldmodel import mse

from navrep3d.auto_debug import enable_auto_debug
from train_gpt import N3DWorldModelDataset

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

# enum with two values, "original" and "navrep"
class AblationOptionType(object):
    ORIGINAL = "original"
    NAVREP = "navrep"

class Ablation(object):
    embedding_size = AblationOptionType.ORIGINAL
    hidden_state_size = AblationOptionType.ORIGINAL

# ----------------------------------------------
class RSSMWMConf(object):
    action_dim = 3
    actor_dist = 'onehot'
    actor_grad = 'reinforce'
    adam_eps = 1e-05
    adam_lr = 0.0003
    adam_lr_actor = 0.0001
    adam_lr_critic = 0.0001
    amp = True
    batch_length = 32
    batch_size = 50
    buffer_size = 10000000
    clip_rewards = None
    cnn_depth = 48
    data_workers = 1
    deter_dim = 2048
    device = 'cuda:0'
    enable_profiler = False
    entropy = 0.003
    env_action_repeat = 1
    env_id = 'NavRep3DStaticASLEnv'
    env_id_eval = None
    env_no_terminal = False
    env_time_limit = 27000
    eval_batch_size = 10
    eval_batches = 61
    eval_interval = 2000
    eval_samples = 10
    gamma = 0.995
    generator_prefill_policy = 'random'
    generator_prefill_steps = 1000
    generator_workers = 1
    generator_workers_eval = 1
    grad_clip = 200
    grad_clip_ac = 200
    gru_layers = 1
    gru_type = 'gru'
    hidden_dim = 1000
    imag_horizon = 15
    image_categorical = False
    image_channels = 3
    image_decoder = 'cnn'
    image_decoder_layers = 0
    image_decoder_min_prob = 0
    image_encoder = 'cnn'
    image_encoder_layers = 0
    image_key = 'image'
    image_size = 64
    image_weight = 1.0
    iwae_samples = 1
    keep_state = True
    kl_balance = 0.8
    kl_weight = 1.0
    lambda_gae = 0.95
    layer_norm = True
    limit_step_ratio = 0
    log_interval = 100
    logbatch_interval = 1000
    map_categorical = True
    map_channels = 4
    map_decoder = 'dense'
    map_hidden_dim = 1024
    map_hidden_layers = 4
    map_key = None
    map_model = 'none'
    map_size = 11
    map_stoch_dim = 64
    mem_loss_type = None
    mem_model = 'none'
    model = 'dreamer'
    n_env_steps = 200000000
    n_steps = 99000000
    offline_data_dir = None
    offline_eval_dir = None
    offline_prefill_dir = None
    offline_test_dir = None
    reset_interval = 200
    resume_id = None
    reward_decoder_categorical = None
    reward_decoder_layers = 4
    reward_input = False
    reward_weight = 1.0
    run_name = None
    save_interval = 500
    stoch_dim = 32
    stoch_discrete = 32
    target_interval = 100
    terminal_decoder_layers = 4
    terminal_weight = 1.0
    test_batch_size = 10
    test_batches = 61
    vecobs_weight = 1.0
    vecobs_size = 2
    verbose = False

# original
#   keep state
#   batch size = 50
#   sequence size = 50
#   ELU activations
#   loss
#   optimizer
#   grad clip
#   mixed precision
#   discrete actions

# equivalence
#   reset state
#   batch size = 128
#   sequence size = 32
#   no rewards


# y_pred, y_rs_pred, loss = model(x, x_rs, a, dones, targets=(y, y_rs))

class RSSMWorldModel(nn.Module):
    """ A prediction model based on DreamerV2's RSSM architecture """

    def __init__(self, conf, gpu=True):
        super().__init__()
        self.block_size = conf.batch_length
        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.kl_balance = None if conf.kl_balance == 0.5 else conf.kl_balance
        # Encoder
        self.encoder = MultiEncoder(conf)
        # Decoders
        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.decoder = MultiDecoder(features_dim, conf)
        # RSSM
        self.core = RSSMCore(embed_dim=self.encoder.out_dim,
                             action_dim=conf.action_dim,
                             deter_dim=conf.deter_dim,
                             stoch_dim=conf.stoch_dim,
                             stoch_discrete=conf.stoch_discrete,
                             hidden_dim=conf.hidden_dim,
                             gru_layers=conf.gru_layers,
                             gru_type=conf.gru_type,
                             layer_norm=conf.layer_norm)
        # Init
        for m in self.modules():
            init_weights_tf2(m)

    def get_block_size(self):
        return self.block_size

    def forward(self, img, state, action, dones, targets=None, h=None):
        """
        img: (batch, sequence, CH, W, H) [0, 1]
        action: (batch, sequence, A) [-inf, inf]
        state: (batch, sequence, S) [-inf, inf]
        dones:  (batch, sequence,) {0, 1}
        targets: None or (img_targets, state_targets)
            img_targets: same shape as img
            state_targets: same shape as state

        OUTPUTS
        img_pred: same shape as img
        state_pred: same shape as state
        loss: torch loss
        """
        b, t, CH, W, H = img.size()
        _, _, A = action.size()
        _, _, S = state.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # ------------------------------------------
        iwae_samples = 1 # always 1 for training
        # do_image_pred seems to be for logging only (nograd). in training loop:
        # do_image_pred=steps % conf.log_interval >= int(conf.log_interval * 0.9)  # 10% of batches
        do_image_pred = False
        do_open_loop = False # always closed loop for training. open loop for eval
        # obs.keys() = (['reset', 'action', 'reward', 'image', 'mission', 'terminal', 'map', 'map_seen_mask', 'map_coord', 'vecobs'])
        # actually_used = ["action", "reset", "terminal", "image", "vecobs", "reward"]
        # action is discrete onehot (T, B, 3)  [0 1 0] 
        # if obs terminal is 0 0 0 1 0 then obs reset is 0 0 0 0 1 (T, B)
        # image is 0-1, float16, (T, B, C, H, W)
        # vecobs is float, robotstate (T, B, 5)
        # reward is float, (T, B)
        obs = {}
        obs["action"] = action.moveaxis(1, 0)
        obs["terminal"] = dones.moveaxis(1, 0)
        obs["reset"] = torch.roll(obs["terminal"], 1, 0) > 0
        obs["image"] = img.moveaxis(1, 0)
        obs["vecobs"] = state.moveaxis(1, 0)
        obs["reward"] = obs["terminal"] * 0.0
        # in_state: Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)] -> h, z
        # we could maintain state across forward but don't to be consistent with other models
        in_state = self.core.init_state(b * iwae_samples)
        # Encoder
        embed = self.encoder(obs)
        # RSSM
        prior, post, post_samples, features, states, out_state = \
            self.core.forward(embed,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)
        # Decoder
        loss_reconstr, metrics, tensors = self.decoder.training_step(features, obs)
        # KL loss
        d = self.core.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (T,B,I)
        if iwae_samples == 1:
            # Analytic KL loss, standard for VAE
            if not self.kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(prior.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(post.detach()), dprior)
                loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + self.kl_balance * loss_kl_priograd
        else:
            # Sampled KL loss, for IWAE
            z = post_samples.reshape(dpost.batch_shape + dpost.event_shape)
            loss_kl = dpost.log_prob(z) - dprior.log_prob(z)
        # Total loss
        assert loss_kl.shape == loss_reconstr.shape
        loss_model_tbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_tbi, dim=2)
        # t+1 predictions (using next-step prior)
        with torch.no_grad():
            prior_samples = self.core.zdistr(prior).sample().reshape(post_samples.shape)
            features_prior = self.core.feature_replace_z(features, prior_samples)
            _, _, tens = self.decoder.training_step(features_prior, obs, extra_metrics=True)
            img_pred = tens["image_rec"].moveaxis(1, 0)
            state_pred = tens["vecobs_rec"].moveaxis(1, 0)
#         return loss_model.mean(), features, states, out_state, metrics, tensors
        x = features
        if h is not None:
            h[0] = states[0]
        return img_pred, state_pred, loss_model
    # -----------------------------

    def _to_correct_device(self, tensor):
        raise NotImplementedError
    def encode(self, img):
        raise NotImplementedError
    def encode_mu_logvar(self, img):
        raise NotImplementedError
    def decode(self, z):
        raise NotImplementedError
    def get_h(self, gpt_sequence):
        raise NotImplementedError
    def get_next(self, gpt_sequence):
        raise NotImplementedError


# In the pydreamer version, _Z is around 1500, _H is large too.
# two options: constrain _Z and _H (ablation) or use original size
_Z = _H = 64
_S = 32  # sequence length


def main(max_steps=222222, dataset="SCR", dry_run=False, ablation=None):
    namestring = "RSSM_A{}".format(ablation)
    if ablation is None:
        raise ValueError("ablation must be specified")
    elif ablation == 0:
        # original
        ablation = Ablation()
    else:
        raise NotImplementedError
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    if dataset == "SCR":
        dataset_dir = [os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dalt"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3doffice"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl"),
                       os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")]
        log_path = os.path.expanduser(
            "~/navrep3d_W/logs/W/{}_SCR_train_log_{}.csv".format(namestring, START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep3d_W/models/W/{}_SCR".format(namestring))
        plot_path = os.path.expanduser("~/tmp_navrep3d/{}_SCR_step".format(namestring))
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

#     mconf = GPTConfig(_S, _H)
    mconf = RSSMWMConf()
    if ablation.embedding_size != AblationOptionType.ORIGINAL:
        raise NotImplementedError
    if ablation.hidden_state_size != AblationOptionType.ORIGINAL:
        raise NotImplementedError
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
    model = RSSMWorldModel(mconf)
    print("RSSM trainable params: {}".format(
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
