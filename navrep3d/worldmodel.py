import copy
import numpy as np
import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """ abstract world model class """
    gpu = True

    def __init__(self, gpu):
        self.gpu = gpu
        super().__init__()

    def get_block_size():
        raise NotImplementedError

    def forward(self, img, vecobs, action, dones, targets=None, h=None):
        """
        img: (batch, sequence, CH, W, H) [0, 1]
        action: (batch, sequence, A) [-inf, inf]
        vecobs: (batch, sequence, S) [-inf, inf]
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
        vecobs = np.array([d["state"] for d in gpt_sequence])  # t, 2
        vecobs = vecobs.reshape((_b, *vecobs.shape))
        vecobs_t = torch.tensor(vecobs, dtype=torch.float)
        vecobs_t = self._to_correct_device(vecobs_t)
        action = np.array([d["action"] for d in gpt_sequence])  # t, 3
        action = action.reshape((_b, *action.shape))
        action_t = torch.tensor(action, dtype=torch.float)
        action_t = self._to_correct_device(action_t)
        dones = np.zeros((_b, len(gpt_sequence)))
        dones_t = torch.tensor(dones, dtype=torch.float)
        dones_t = self._to_correct_device(dones_t)
        h_container = [None]
        self.forward(img_t, vecobs_t, action_t, dones_t, h=h_container)
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
        vecobs = np.array([d["state"] for d in gpt_sequence])  # t, 2
        vecobs = vecobs.reshape((_b, *vecobs.shape))
        vecobs_t = torch.tensor(vecobs, dtype=torch.float)
        vecobs_t = self._to_correct_device(vecobs_t)
        action = np.array([d["action"] for d in gpt_sequence])  # t, 3
        action = action.reshape((_b, *action.shape))
        action_t = torch.tensor(action, dtype=torch.float)
        action_t = self._to_correct_device(action_t)
        dones = np.zeros((_b, len(gpt_sequence)))
        dones_t = torch.tensor(dones, dtype=torch.float)
        dones_t = self._to_correct_device(dones_t)
        img_pred_t, vecobs_pred_t, _ = self.forward(img_t, vecobs_t, action_t, dones_t, h=None)
        img_pred = img_pred_t.detach().cpu().numpy()
        img_pred = img_pred[0, -1]  # only batch, last item in sequence
        img_pred = np.moveaxis(img_pred, 0, -1)
        img_pred = np.clip(img_pred, 0., 1.)
        vecobs_pred = vecobs_pred_t.detach().cpu().numpy()
        vecobs_pred = vecobs_pred[0, -1]  # only batch, last item in sequence
        return img_pred, vecobs_pred

    def fill_dream_sequence(self, real_sequence, context_length):
        """ Fills dream sequence based on context from real_sequence
            real_sequence is a list of dicts, one for each step in the sequence.
            each dict has
            "obs": numpy image (W, H, CH) [0, 1]
            "state": numpy (2,) [-inf, inf]
            "action": numpy (3,) [-inf, inf]

            context_length (int): number of steps of the real sequence to keep in the dream sequence

            output:
            dream_sequence: same length as the real_sequence, but observations and states are predicted
                    open-loop by the worldmodel, while actions are taken from the real sequence
            """
        T = self.get_block_size()
        sequence_length = len(real_sequence)
        if sequence_length > T:
            print("Warning: sequence_length > block_size ({} > {} in {})!".format(
                sequence_length, T, type(self).__name__))
        dream_sequence = copy.deepcopy(real_sequence[:context_length])
        dream_sequence[-1]['action'] = None
        real_actions = [d['action'] for d in real_sequence]
        next_actions = real_actions[context_length-1:sequence_length-1]
        for action in next_actions:
            dream_sequence[-1]['action'] = action * 1.
            img_npred, goal_pred = self.get_next(dream_sequence[-T:])
            # update sequence
            dream_sequence.append(dict(obs=img_npred, state=goal_pred, action=None))
        dream_sequence[-1]['action'] = next_actions[-1] * 1.
        return dream_sequence

class DummyWorldModel(WorldModel):
    def get_block_size(self):
        return 1024

    def forward(self, img, vecobs, action, dones, targets=None, h=None):
        img_pred = img * 1.
        vecobs_pred = vecobs * 1.
        loss = torch.tensor(0.0)
        if h is not None:
            raise NotImplementedError
        if targets is not None:
            raise NotImplementedError
        return img_pred, vecobs_pred, loss

class GreyDummyWorldModel(WorldModel):
    def get_block_size(self):
        return 1024

    def forward(self, img, vecobs, action, dones, targets=None, h=None):
        img_pred = img * 0. + 0.5
        vecobs_pred = vecobs * 0.
        loss = torch.tensor(0.0)
        if h is not None:
            raise NotImplementedError
        if targets is not None:
            raise NotImplementedError
        return img_pred, vecobs_pred, loss

