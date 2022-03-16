import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from navrep.models.torchvae import VAE
from pydreamer.models.dreamer import RSSMCore, init_weights_tf2, D
from navdreams.worldmodel import WorldModel

ablation = 1
# ablated the encoder / decoder to the transformer one

# ----------------------------------------------
class RSSMWMConf(object):
    # most of these are not relevant for our use here, but are kept for reference
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
    kl_weight = 0.01
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

class RSSMWorldModel(WorldModel):
    """ A prediction model based on DreamerV2's RSSM architecture """

    def __init__(self, conf, gpu=True):
        super().__init__(gpu)
        self.block_size = conf.batch_length
        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.kl_balance = None if conf.kl_balance == 0.5 else conf.kl_balance
        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        # Encoder
        self.convVAE = VAE(z_dim=features_dim, gpu=gpu, image_channels=conf.image_channels)
        self.vecobs_emb = nn.Linear(conf.vecobs_size, features_dim)
        # Decoders
        self.vecobs_head = nn.Linear(features_dim, conf.vecobs_size)

        # RSSM
        self.core = RSSMCore(embed_dim=features_dim,
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

    def forward(self, img, vecobs, action, dones, targets=None, h=None):
        """
        img: (batch, sequence, CH, W, H) [0, 1]
        action: (batch, sequence, A) [-inf, inf]
        vecobs: (batch, sequence, V) [-inf, inf]
        dones:  (batch, sequence,) {0, 1}
        targets: None or (img_targets, vecobs_targets)
            img_targets: same shape as img
            vecobs_targets: same shape as vecobs
        h: None or []
            if None, will be ignored
            if [] will be filled with RNN state (batch, sequence, H)

        OUTPUTS
        img_pred: same shape as img
        vecobs_pred: same shape as vecobs
        loss: torch loss
        """
        b, t, CH, W, H = img.size()
        _, _, A = action.size()
        _, _, V = vecobs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # ------------------------------------------
        iwae_samples = 1 # always 1 for training
        # do_image_pred seems to be for logging only (nograd). in training loop:
        # do_image_pred=steps % conf.log_interval >= int(conf.log_interval * 0.9)  # 10% of batches
        # do_image_pred = False
        do_open_loop = False # always closed loop for training. open loop for eval
        # obs.keys() = (['reset', 'action', 'reward', 'image', 'mission', 'terminal', 'map', 'map_seen_mask', 'map_coord', 'vecobs']) # noqa
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
        obs["vecobs"] = vecobs.moveaxis(1, 0)
        obs["reward"] = obs["terminal"] * 0.0
        # in_state: Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)] -> h, z
        # we could maintain state across forward but don't to be consistent with other models
        in_state = self.core.init_state(b * iwae_samples)
        # Encoder
        img_embed, _, _ = self.convVAE.encode(obs["image"].reshape(b * t, CH, W, H))
        img_embed = img_embed.view(t, b, -1)
        vecobs_embed = self.vecobs_emb(obs["vecobs"].reshape(b * t, V)).view(t, b, -1)
        embed = img_embed + vecobs_embed
        # RSSM
        prior, post, post_samples, features, states, out_state = \
            self.core.forward(embed,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)
        # Decoder
        img_rec = self.convVAE.decode(features.view(b * t, -1)).view(t, b, CH, W, H)
        vecobs_rec = self.vecobs_head(features.view(b * t, -1)).view(t, b, V)
        img_rec_loss = F.mse_loss(img_rec, obs["image"])  # samestep-reconstruction loss
        STATE_NORM_FACTOR = 25.  # maximum typical goal distance, meters
        vecobs_pred_loss = F.mse_loss(vecobs_rec, obs["vecobs"]) / STATE_NORM_FACTOR**2
        loss_reconstr = img_rec_loss + vecobs_pred_loss
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
        loss_model = self.kl_weight * loss_kl.mean() + loss_reconstr
        # t+1 predictions (using next-step prior)
        with torch.no_grad():
            prior_samples = self.core.zdistr(prior).sample().reshape(post_samples.shape)
            features_prior = self.core.feature_replace_z(features, prior_samples)
            img_pred = self.convVAE.decode(features_prior.view(b * t, -1)).view(t, b, CH, W, H)
            vecobs_pred = self.vecobs_head(features_prior.view(b * t, -1)).view(t, b, V)
            img_pred = img_pred.moveaxis(1, 0)
            vecobs_pred = vecobs_pred.moveaxis(1, 0)
        # x = features
        if h is not None:
            h_states = states[0] # T, B, I, H
            h[0] = h_states.view((t, b, -1)).moveaxis(1, 0)
        return img_pred, vecobs_pred, loss_model
    # -----------------------------

    def encode_mu_logvar(self, img):
        """
        img: numpy (batch, W, H, CH)


        OUTPUTS
        mu: (batch, Z)
        logvar: (batch, Z)
        """
        b, W, H, CH = img.shape

        img_t = torch.tensor(np.moveaxis(img, -1, 1), dtype=torch.float)
        img_t = self._to_correct_device(img_t) # B, CH, W, H

        raise NotImplementedError
        # correct way would be to run RSSM on sequence of length 1, get the features
        embed = self.encoder.encoder_image.forward(img_t.view((1, b, CH, W, H)))  # (T,B,E)
        z = embed.view((b, -1))

        mu = z.detach().cpu().numpy()
        logvar = np.zeros_like(mu)
        return mu, logvar

    def decode(self, z):
        """
        z: numpy (batch, Z)

        OUTPUTS
        img_rec: (batch, W, H, CH)
        """
        b, Z = z.shape

        z_t = torch.tensor(z, dtype=torch.float)
        z_t = self._to_correct_device(z_t)

        # decoder expects features (T, B, I, Z) - T is sequence, I is samples (both irrelevant here)
        z_t = z_t.view((1, b, 1, Z))
        raise NotImplementedError
        img_rec_t = self.decoder.image.forward(z_t) # t, b, I, CH, W, H
        T, B, I, CH, W, H = img_rec_t.shape
        img_rec_t = img_rec_t.view((B, CH, W, H))

        img_rec = np.moveaxis(img_rec_t.detach().cpu().numpy(), 1, -1)
        return img_rec
