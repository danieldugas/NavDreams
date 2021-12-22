import numpy as np
import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """ abstract world model class """
    gpu = True

    def __init__(self):
        super().__init__()

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
        raise NotImplementedError

    def encode_mu_logvar(self, img):
        """
        img: numpy (batch, W, H, CH)


        OUTPUTS
        mu: (batch, Z)
        logvar: (batch, Z)
        """
        raise NotImplementedError

    def decode(self, z):
        """
        z: numpy (batch, Z)

        OUTPUTS
        img_rec: (batch, W, H, CH)
        """
        raise NotImplementedError

    def _to_correct_device(self, tensor):
        if self.gpu:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                return tensor.to(device)
            else:
                print("WARNING: model created with gpu enabled, but no gpu found")
        return tensor

    def get_h(self, gpt_sequence):
        """ for compat with encodedenv
        gpt sequence is a list of dicts, one for each step in the sequence.
        each dict has
        "obs": numpy image (W, H, CH) [0, 1]
        "state": numpy (2,) [-inf, inf]
        "action": numpy (3,) [-inf, inf]
        """
        _b = 1  # batch size
        img = np.array([d["obs"] for d in gpt_sequence])  # t, W, H, CH
        img = np.moveaxis(img, -1, 1)
        img = img.reshape((_b, *img.shape))
        img_t = torch.tensor(img, dtype=torch.float)
        img_t = self._to_correct_device(img_t)
        state = np.array([d["state"] for d in gpt_sequence])  # t, 2
        state = state.reshape((_b, *state.shape))
        state_t = torch.tensor(state, dtype=torch.float)
        state_t = self._to_correct_device(state_t)
        action = np.array([d["action"] for d in gpt_sequence])  # t, 3
        action = action.reshape((_b, *action.shape))
        action_t = torch.tensor(action, dtype=torch.float)
        action_t = self._to_correct_device(action_t)
        dones = np.zeros((_b, len(gpt_sequence), 1))
        dones_t = torch.tensor(dones, dtype=torch.float)
        dones_t = self._to_correct_device(dones_t)
        h_container = [None]
        self.forward(img_t, state_t, action_t, dones_t, h=h_container)
        h = h_container[0].detach().cpu().numpy()
        h = h[0, -1]  # only batch, last item in sequence
        return h

    def get_next(self, gpt_sequence):
        """ for compat with encodedenv
        gpt sequence is a list of dicts, one for each step in the sequence.
        each dict has
        "obs": numpy image (W, H, CH) [0, 1]
        "state": numpy (2,) [-inf, inf]
        "action": numpy (3,) [-inf, inf]
        output:
        img_pred: (W, H, CH)
        state_pred: (2,)
        """
        _b = 1  # batch size
        img = np.array([d["obs"] for d in gpt_sequence])  # t, W, H, CH
        img = np.moveaxis(img, -1, 1)
        img = img.reshape((_b, *img.shape))
        img_t = torch.tensor(img, dtype=torch.float)
        img_t = self._to_correct_device(img_t)
        state = np.array([d["state"] for d in gpt_sequence])  # t, 2
        state = state.reshape((_b, *state.shape))
        state_t = torch.tensor(state, dtype=torch.float)
        state_t = self._to_correct_device(state_t)
        action = np.array([d["action"] for d in gpt_sequence])  # t, 3
        action = action.reshape((_b, *action.shape))
        action_t = torch.tensor(action, dtype=torch.float)
        action_t = self._to_correct_device(action_t)
        dones = np.zeros((_b, len(gpt_sequence), 1))
        dones_t = torch.tensor(dones, dtype=torch.float)
        dones_t = self._to_correct_device(dones_t)
        h_container = [None]
        img_pred_t, state_pred_t, _ = self.forward(img_t, state_t, action_t, dones_t, h=h_container)
        img_pred = img_pred_t.detach().cpu().numpy()
        img_pred = img_pred[0, -1]  # only batch, last item in sequence
        img_pred = np.moveaxis(img_pred, 0, -1)
        state_pred = state_pred_t.detach().cpu().numpy()
        state_pred = state_pred[0, -1]  # only batch, last item in sequence
        return img_pred, state_pred
