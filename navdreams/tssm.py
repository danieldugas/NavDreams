import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from navrep.models.torchvae import VAE
from navrep.models.gpt import Block
from pydreamer.models.dreamer import D

from navdreams.worldmodel import WorldModel

version = 2

_Z = 1024
_H = _Z + 32 * 32
_S = 32  # sequence length

class TSSMWMConf(object):
    adam_eps = 1e-05
    adam_lr = 0.0003
    amp = True

    stoch_dim = 32
    stoch_discrete = 32
    n_embd = _Z
    image_size = 64
    image_channels = 3
    vecobs_size = 2
    n_action = 3
    # optimizer
    grad_clip = 200
    # loss
    kld_weight = 0.01
    kl_balance = 0.8
    # transformer block params
    block_size = 32 # sequence length
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_layer = 8
    n_head = 8


class TSSMWorldModel(WorldModel):
    def __init__(self, config, gpu=True):
        super().__init__(gpu=gpu)
        self.conf = config

        self.prior_size = config.stoch_dim * config.stoch_discrete # stochastic state size * # categories
        self.sampled_state_size = config.n_embd + self.prior_size

        # input embedding stem
        self.convVAE = VAE(z_dim=config.n_embd, gpu=gpu, image_channels=config.image_channels)
        self.action_emb = nn.Linear(config.n_action, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.vecobs_emb = nn.Linear(config.vecobs_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # prior and posteriors
        self.dstate_to_nextprior = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, self.prior_size),
        )
        self.dstate_to_mix = nn.Linear(config.n_embd, config.n_embd)
        self.embedding_to_mix = nn.Linear(config.n_embd, config.n_embd)
        self.mix_to_post = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, self.prior_size),
        )
        def to_distribution(prior_or_posterior):
            logits = prior_or_posterior.reshape(
                prior_or_posterior.shape[:-1] + (config.stoch_dim, config.stoch_discrete))
            distr = D.OneHotCategoricalStraightThrough(logits=logits.float())  # forces float32 in AMP
            distr = D.independent.Independent(distr, 1) # This makes d.entropy() and d.kl() sum over stoch_dim
            return distr
        self.to_distribution = to_distribution
        # decoder head
        self.z_head = nn.Linear(self.sampled_state_size, config.n_embd)
        self.vecobs_head = nn.Linear(self.sampled_state_size, config.vecobs_size)

        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.apply(self._init_weights)

        self.gpu = gpu

        print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, img, vecobs, action, dones, targets=None, h=None):
        """
        img: (batch, sequence, CH, W, H) [0, 1]
        action: (batch, sequence, A) [-inf, inf]
        vecobs: (batch, sequence, S) [-inf, inf]
        dones:  (batch, sequence,) {0, 1}
        targets: None or (img_targets, vecobs_targets)
            img_targets: same shape as img
            vecobs_targets: same shape as vecobs

        OUTPUTS
        img_pred: same shape as img
        vecobs_pred: same shape as vecobs
        loss: torch loss
        """
        b, t, CH, W, H = img.size()
        _, _, A = action.size()
        _, _, S = vecobs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        E = self.n_embd
        FS = self.sampled_state_size

        # encode embedding with vae
        z, mu, logvar = self.convVAE.encode(img.view(b * t, CH, W, H))  # each image maps to a vector
        token_embeddings = z.view(b, t, E)
        vecobs_embeddings = self.vecobs_emb(vecobs.view(b * t, S)).view(b, t, E)
        action_embeddings = self.action_emb(action.view(b * t, A)).view(b, t, E)
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        # forward the GPT model
        full_embeddings = self.drop(
            token_embeddings + position_embeddings + action_embeddings + vecobs_embeddings)
        x = self.blocks(full_embeddings)
        deterministic_state = self.ln_f(x)
        # get posterior and nextprior from deterministic state
        posterior = self.mix_to_post(
            self.dstate_to_mix(deterministic_state.view(b * t, E))
            + self.embedding_to_mix(full_embeddings.view(b * t, E))
        ).view(b, t, self.prior_size)
        nextprior = self.dstate_to_nextprior(
            deterministic_state.view(b * t, E)).view(b, t, self.prior_size)
        # sample from posterior
        posterior_distr = self.to_distribution(posterior)
        posterior_sample = posterior_distr.rsample().view(b, t, self.prior_size)
        sampled_fullstate = torch.cat((deterministic_state, posterior_sample), -1)
        # store worldmodel embedding
        if h is not None:
            h[0] = sampled_fullstate # this is the predicted "world state"
        # decode embedding with vae
        z_pred = self.z_head(sampled_fullstate.view(b * t, FS))
        img_rec = self.convVAE.decode(z).view(b, t, CH, W, H)
        img_pred = self.convVAE.decode(z_pred).view(b, t, CH, W, H)
        vecobs_pred = self.vecobs_head(sampled_fullstate.view(b * t, FS)).view(b, t, S)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if t < 2:
                raise ValueError("Cannot calculate prediction loss, sequence length is < 2.")
            img_targets, vecobs_targets = targets
            img_pred_loss = F.mse_loss(img_pred, img_targets)  # prediction-reconstruction loss
            img_rec_loss = F.mse_loss(img_rec, img)  # samestep-reconstruction loss
            img_rec_weight = 0.1
            # prediction loss is the KL divergence between the prior and the posterior
            STATE_NORM_FACTOR = 25.  # maximum typical goal distance, meters
            vecobs_pred_loss = F.mse_loss(vecobs_pred, vecobs_targets) / STATE_NORM_FACTOR**2
            # posterior from prior prediction
            prior_1_to_n = nextprior[:, :-1, :]
            posterior_1_to_n = posterior[:, 1:, :]
            dprior = self.to_distribution(prior_1_to_n)
            dpost = self.to_distribution(posterior_1_to_n)
            dprior_nograd = self.to_distribution(prior_1_to_n.detach())
            dpost_nograd = self.to_distribution(posterior_1_to_n.detach())
            loss_kl_postgrad = D.kl.kl_divergence(dpost, dprior_nograd)
            loss_kl_priograd = D.kl.kl_divergence(dpost_nograd, dprior)
            loss_kl = (1 - self.conf.kl_balance) * loss_kl_postgrad + self.conf.kl_balance * loss_kl_priograd
            loss = (img_pred_loss
                    + img_rec_loss * img_rec_weight
                    + vecobs_pred_loss
                    + self.conf.kld_weight * loss_kl)

        return img_pred, vecobs_pred, loss

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

        z, mu, logvar = self.convVAE.encode(img_t)

        mu = mu.detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()
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

        img_rec_t = self.convVAE.decode(z_t) # b, CH, W, H

        img_rec = np.moveaxis(img_rec_t.detach().cpu().numpy(), 1, -1)
        return img_rec
