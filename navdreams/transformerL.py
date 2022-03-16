import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

from navrep.models.torchvae import VAE
from navrep.models.gpt import Block

from navdreams.worldmodel import WorldModel

logger = logging.getLogger(__name__)

version = 0

_Z = 1024
_H = _Z
_S = 32  # sequence length
STATE_NORM_FACTOR = 25.  # maximum typical goal distance, meters

class TransformerLWMConf(object):
    adam_eps = 1e-05
    adam_lr = 0.0003
    amp = True

    n_embd = 1024
    image_size = 64
    image_channels = 3
    vecobs_size = 2
    n_action = 3
    # optimizer
    grad_clip = 200
    # transformer block params
    block_size = 32 # sequence length
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_layer = 8
    n_head = 8

class TransformerLWorldModel(WorldModel):
    """ A transformer model with wider latent space """

    def __init__(self, config, gpu=True):
        super().__init__(gpu)

        # input embedding stem
        self.convVAE = VAE(z_dim=config.n_embd, gpu=gpu, image_channels=config.image_channels)
        self.action_emb = nn.Linear(config.n_action, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.vecobs_emb = nn.Linear(config.vecobs_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.z_head = nn.Linear(config.n_embd, config.n_embd)
        self.vecobs_head = nn.Linear(config.n_embd, config.vecobs_size)

        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.apply(self._init_weights)

        self.gpu = gpu

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

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

        # encode embedding with vae
        z, mu, logvar = self.convVAE.encode(img.view(b * t, CH, W, H))  # each image maps to a vector
        token_embeddings = z.view(b, t, self.n_embd)
        vecobs_embeddings = self.vecobs_emb(vecobs.view(b * t, S)).view(b, t, self.n_embd)
        action_embeddings = self.action_emb(action.view(b * t, A)).view(b, t, self.n_embd)
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        # forward the GPT model
        x = self.drop(token_embeddings + position_embeddings + action_embeddings + vecobs_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        # store worldmodel embedding
        if h is not None:
            h[0] = x
        # decode embedding with vae
        z_pred = self.z_head(x.view(b * t, self.n_embd))
        img_rec = self.convVAE.decode(z).view(b, t, CH, W, H)
        img_pred = self.convVAE.decode(z_pred).view(b, t, CH, W, H)
        vecobs_pred = self.vecobs_head(x.view(b * t, self.n_embd)).view(b, t, S)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            img_targets, vecobs_targets = targets
            rec_loss = F.mse_loss(img_rec, img)  # input-reconstruction loss
            img_loss_weight = 10.0
            pred_loss = F.mse_loss(img_pred, img_targets)  # reconstructed prediction loss
            vecobs_loss = F.mse_loss(vecobs_pred, vecobs_targets) / STATE_NORM_FACTOR**2
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # kullback leibler
            kld_tolerance = 0.5
            kld_weight = 0.001
            KLD = torch.max(KLD, kld_tolerance * torch.ones_like(KLD))
            loss = rec_loss + kld_weight * KLD + pred_loss * img_loss_weight + vecobs_loss

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
        img_t = self._to_correct_device(img_t)

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
