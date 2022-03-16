import numpy as np
import copy
import torch
from pydreamer.models.dreamer import RSSMCore, MultiDecoder, MultiEncoder, init_weights_tf2, D, logavgexp
from navdreams.worldmodel import WorldModel

# original loss, decoder encoder, everything
ablation = 0

# ----------------------------------------------
class RSSMA0WMConf(object):
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

class RSSMA0WorldModel(WorldModel):
    """ A prediction model based on DreamerV2's RSSM architecture """

    def __init__(self, conf, gpu=True):
        super(RSSMA0WorldModel, self).__init__(gpu)
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
        h: None or []
            if None, will be ignored
            if [] will be filled with RNN state (batch, sequence, H)

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
        # x = features
        if h is not None:
            h_states = states[0] # T, B, I, H
            h[0] = h_states.view((t, b, -1)).moveaxis(1, 0)
        return img_pred, state_pred, loss_model
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
        raise NotImplementedError(
            "decoder input is not same size as encoder output, so decoding is not implemented yet")
        # this will fail if z_t is not size 3072
        img_rec_t = self.decoder.image.forward(z_t) # t, b, I, CH, W, H
        T, B, I, CH, W, H = img_rec_t.shape
        img_rec_t = img_rec_t.view((B, CH, W, H))

        img_rec = np.moveaxis(img_rec_t.detach().cpu().numpy(), 1, -1)
        return img_rec

    def sequence_to_end_state(self, real_sequence):
        _b = 1  # batch size
        img = np.array([d["obs"] for d in real_sequence])  # t, W, H, CH
        img = np.moveaxis(img, -1, 1)
        img = img.reshape((_b, *img.shape))
        img_t = torch.tensor(img, dtype=torch.float)
        img_t = self._to_correct_device(img_t)
        vecobs = np.array([d["state"] for d in real_sequence])  # t, 2
        vecobs = vecobs.reshape((_b, *vecobs.shape))
        vecobs_t = torch.tensor(vecobs, dtype=torch.float)
        vecobs_t = self._to_correct_device(vecobs_t)
        action = np.array([d["action"] for d in real_sequence])  # t, 3
        action = action.reshape((_b, *action.shape))
        action_t = torch.tensor(action, dtype=torch.float)
        action_t = self._to_correct_device(action_t)
        dones = np.zeros((_b, len(real_sequence)))
        dones_t = torch.tensor(dones, dtype=torch.float)
        dones_t = self._to_correct_device(dones_t)
        b, t, CH, W, H = img_t.size()
        _, _, A = action_t.size()
        _, _, S = vecobs_t.size()
        # ------------------------------------------
        iwae_samples = 1 # always 1 for training
        do_open_loop = False
        obs = {}
        obs["action"] = action_t.moveaxis(1, 0)
        obs["terminal"] = dones_t.moveaxis(1, 0)
        obs["reset"] = torch.roll(obs["terminal"], 1, 0) > 0
        obs["image"] = img_t.moveaxis(1, 0)
        obs["vecobs"] = vecobs_t.moveaxis(1, 0)
        obs["reward"] = obs["terminal"] * 0.0
        in_state = self.core.init_state(b * iwae_samples)
        embed = self.encoder(obs)
        _, _, _, _, _, out_state = \
            self.core.forward(embed,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)
        return out_state # (h, z)

    def get_next(self, *args, **kwargs):
        print("Warning!: RSSM get_next should not be used in a sequential fashion."
              + "Use fill_dream_sequence instead.")
        return super(RSSMA0WorldModel, self).get_next(*args, **kwargs)

    def generate_dream_sequence(self, in_state, actions):
        _b = 1  # batch size
        t, A = np.array(actions).shape
        assert A == 3
        action = actions
        action = action.reshape((_b, *action.shape))
        action_t = torch.tensor(action, dtype=torch.float)
        action_t = self._to_correct_device(action_t)
        dones = np.zeros((_b, t))
        dones_t = torch.tensor(dones, dtype=torch.float)
        dones_t = self._to_correct_device(dones_t)
        _, _, A = action_t.size()
        # ------------------------------------------
        iwae_samples = 1 # always 1 for training
        do_open_loop = True # always closed loop for training. open loop for eval
        obs = {}
        obs["action"] = action_t.moveaxis(1, 0)
        obs["terminal"] = dones_t.moveaxis(1, 0)
        obs["reset"] = torch.roll(obs["terminal"], 1, 0) > 0
        obs["reward"] = obs["terminal"] * 0.0
        # RSSM
        prior, post, post_samples, features, states, out_state = \
            self.core.forward(None,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)
        # Decoder
#         assert conf.image_decoder == 'cnn'
        image_decoded = self.decoder.image.forward(features)
        image_decoded = image_decoded.mean(dim=2)
        vecobs_decoded = self.decoder.vecobs.forward(features)
        vecobs_decoded = vecobs_decoded.mean.mean(dim=2)

        # torch to numpy
        img_pred_t = image_decoded.moveaxis(1, 0)
        vecobs_pred_t = vecobs_decoded.moveaxis(1, 0)
        img_pred = img_pred_t.detach().cpu().numpy()
        img_pred = img_pred[0, :]  # only batch
        img_pred = np.moveaxis(img_pred, 1, -1)
        img_pred = np.clip(img_pred, 0., 1.)
        vecobs_pred = vecobs_pred_t.detach().cpu().numpy()
        vecobs_pred = vecobs_pred[0, :]  # only batch

        dream_sequence = []
        for image, vecobs, action in zip(img_pred, vecobs_pred, list(actions[1:]) + [actions[-1]]):
            dream_sequence.append(dict(obs=image, state=vecobs, action=action))

        return dream_sequence

    def fill_dream_sequence_through_images(self, *args, **kwargs):
        return super(RSSMA0WorldModel, self).fill_dream_sequence(*args, **kwargs)

    def fill_dream_sequence(self, real_sequence, context_length):
        """ overrides the fill_dream_sequence method of the base class,
        to predict from last state instead of from last image pred """
        sequence_length = len(real_sequence)
        context_sequence = copy.deepcopy(real_sequence[:context_length])
        context_rnn_state = self.sequence_to_end_state(context_sequence)
        real_actions = [d['action'] for d in real_sequence]
        next_actions = real_actions[context_length-1:sequence_length-1]
        dream_sequence = self.generate_dream_sequence(context_rnn_state, np.array(next_actions))
        full_sequence = context_sequence + dream_sequence
        assert len(full_sequence) == sequence_length
        return full_sequence
