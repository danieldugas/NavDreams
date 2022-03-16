from __future__ import print_function
import numpy as np
import os
from gym import spaces

from navrep.models.gpt import GPT, GPTConfig, load_checkpoint

from navdreams.transformerL import TransformerLWMConf, TransformerLWorldModel
from navdreams.rssm_a0 import RSSMA0WMConf, RSSMA0WorldModel
from navdreams.tssm import TSSMWMConf, TSSMWorldModel

PUNISH_SPIN = True

""" W backends: GPT, RSSM_A0, TransformerL """
""" ENCODINGS: V_ONLY, VM, M_ONLY """
_G = 2  # goal dimensions
_A = 3  # action dimensions
_RS = 5  # robot state
_64 = 64  # image size
_C = 3 # channels in image
_L = 1080  # lidar size
NO_VAE_VAR = True

BLOCK_SIZE = 32  # sequence length (context)

class EnvEncoder(object):
    """ Generic class to encode the observations of an environment,
    look at EncodedEnv to see how it is typically used """
    def __init__(self,
                 backend, encoding,
                 wm_model_path=os.path.expanduser("~/navrep3d/models/W/transformer"),
                 e2e_model_path=os.path.expanduser("~/navrep3d/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip"), # noqa
                 gpu=False,
                 encoder_to_share_model_with=None,  # another EnvEncoder
                 ):
        LIDAR_NORM_FACTOR = None
        if backend == "GPT":
            from navrep.scripts.train_gpt import _Z, _H
        elif backend == "RSSM_A0":
            _Z = 1536
            _H = None
        elif backend == "TransformerL_V0":
            _Z = 1024
            _H = None
        elif backend == "TSSM_V2":
            _Z = 1024
            _H = None
        elif backend == "E2E":
            _Z = 64
            _H = None
        self._Z = _Z
        self._H = _H
        self.LIDAR_NORM_FACTOR = LIDAR_NORM_FACTOR
        self.encoding = encoding
        self.backend = backend
        if self.encoding == "V_ONLY":
            self.encoding_dim = _Z + _RS
        elif self.encoding == "VM":
            self.encoding_dim = _Z + _H + _RS
        elif self.encoding == "M_ONLY":
            self.encoding_dim = _H + _RS
        else:
            raise NotImplementedError
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.encoding_dim,), dtype=np.float32)
        # V + M Models
        if encoder_to_share_model_with is not None:
            self.vae = encoder_to_share_model_with.vae
            self.rnn = encoder_to_share_model_with.rnn
        else:
            # load world model
            if self.backend == "GPT":
                mconf = GPTConfig(BLOCK_SIZE, _H)
                mconf.image_channels = _C
                model = GPT(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                print("loaded WM at {}".format(wm_model_path))
                self.vae = model
                self.rnn = model
            elif self.backend == "TransformerL_V0":
                mconf = TransformerLWMConf()
                mconf.image_channels = 3
                model = TransformerLWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                self.vae = model
                self.rnn = model
                if self.encoding != "V_ONLY":
                    raise NotImplementedError
            elif self.backend == "RSSM_A0":
                mconf = RSSMA0WMConf()
                mconf.image_channels = 3
                model = RSSMA0WorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                self.vae = model
                self.rnn = model
                if self.encoding != "V_ONLY":
                    raise NotImplementedError
            elif self.backend == "TSSM_V2":
                mconf = TSSMWMConf()
                mconf.image_channels = 3
                model = TSSMWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                self.vae = model
                self.rnn = model
                if self.encoding != "V_ONLY":
                    raise NotImplementedError
            elif self.backend == "E2E":
                from stable_baselines3 import PPO
                import torch
                model = PPO.load(e2e_model_path)
                class Model(object):
                    def __init__(self, torch_model, gpu):
                        self.torch_model = torch_model
                        self.gpu = gpu
                    def _to_correct_device(self, tensor):
                        if self.gpu:
                            if torch.cuda.is_available():
                                device = torch.cuda.current_device()
                                return tensor.to(device)
                            else:
                                print("WARNING: model created with gpu enabled, but no gpu found")
                        return tensor
                    def encode_mu_logvar(self, img):
                        """ img is normalized [0-1] (that's what the sb3 model expects) """
                        b, W, H, CH = img.shape
                        tm = self.torch_model
                        img_t = torch.tensor(np.moveaxis(img, -1, 1), dtype=torch.float)
                        img_t = self._to_correct_device(img_t)
                        mu = tm.linear(tm.cnn(img_t))
                        mu = mu.detach().cpu().numpy()
                        logvar = np.zeros_like(mu)
                        return mu, logvar
                self.vae = Model(model.policy.features_extractor, gpu)
            else:
                raise NotImplementedError
        # other tools
        self.viewer = None
        # environment state variables
        self.reset()

    def reset(self):
        if self.encoding in ["VM", "M_ONLY"]:
            self.wm_sequence = []
        self.latest_z = np.zeros(self._Z)
        self.latest_image_obs = np.zeros((_64, _64, _C), dtype=np.uint8)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def _get_last_decoded_obs(self):
        nobs_pred = self.vae.decode(self.latest_z.reshape((1,self._Z))).reshape((_64, _64, _C))
        return nobs_pred

    def _normalize_obs(self, obs):
        """
        transforms the raw env obs (observation_space - eg image 0-255)
        to the format required by model (eg. image 0-1.)
        """
        return obs / 255.

    def _unnormalize_obs(self, nobs):
        return (nobs * 255).astype(np.uint8)

    def _encode_obs(self, obs, action):
        """
    obs is (image, other_obs)
    where image is (w, h, channel) with values 0-255 (uint8)
    and other_obs is (5,) - [goal_x, goal_y, vel_x, vel_y, vel_theta] all in robot frame
    """
        # obs to z, mu, logvar
        image_nobs = self._normalize_obs(obs[0])
        mu, logvar = self.vae.encode_mu_logvar(image_nobs.reshape((1, _64, _64, _C)))
        mu = mu[0]
        logvar = logvar[0]
        s = logvar.shape
        if NO_VAE_VAR:
            latest_z = mu * 1.
        else:
            latest_z = mu + np.exp(logvar / 2.0) * np.random.randn(*s)

        # encode obs through V + M
        self.latest_image_obs = obs[0]
        self.latest_z = latest_z
        if self.encoding == "V_ONLY":
            encoded_obs = np.concatenate([self.latest_z, obs[1]], axis=0)
        elif self.encoding in ["VM", "M_ONLY"]:
            # get h
            if self.backend in ["GPT", "VAE1DLSTM", "GPT1D"]:
                self.wm_sequence.append(dict(obs=image_nobs, state=obs[1][:2], action=action))
                self.wm_sequence = self.wm_sequence[:BLOCK_SIZE]
                h = self.rnn.get_h(self.wm_sequence)
            else:
                raise NotImplementedError
            # encoded obs
            if self.encoding == "VM":
                encoded_obs = np.concatenate([self.latest_z, obs[1], h], axis=0)
            elif self.encoding == "M_ONLY":
                encoded_obs = np.concatenate([h, obs[1]], axis=0)
        return encoded_obs

    def _render_side_by_side(self, mode="human", close=False, save_to_file=False):
        """ renders true and encoded image side by side """
        if close:
            self.viewer.close()
            return
        # rendering
        last_image = self._normalize_obs(self.latest_image_obs)
        last_pred = self._get_last_decoded_obs()
        # Window and viewport size
        image_size = _64  # grid cells
        padding = 4  # grid cells
        grid_size = 1  # px per grid cell
        WINDOW_W = (2 * image_size + 3 * padding) * grid_size
        WINDOW_H = (1 * image_size + 2 * padding) * grid_size
        VP_W = WINDOW_W
        VP_H = WINDOW_H
        from gym.envs.classic_control import rendering
        import pyglet
        from pyglet import gl
        # Create viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.rendering_iteration = 0
        # Render in pyglet
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()
        win.clear()
        gl.glViewport(0, 0, VP_W, VP_H)
        # colors
        bgcolor = np.array([0.4, 0.8, 0.4])
        # Green background
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
        gl.glVertex3f(0, VP_H, 0)
        gl.glVertex3f(VP_W, VP_H, 0)
        gl.glVertex3f(VP_W, 0, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glEnd()
        # rings - observation
        w_offset = 0
        for img in [last_image, last_pred]:
            for i in range(image_size):
                for j in range(image_size):
                    cell_color = img[i, j]
                    cell_y = (padding + i) * grid_size  # px
                    cell_x = (padding + j + w_offset) * grid_size  # px
                    cell_y = WINDOW_H - cell_y
                    gl.glBegin(gl.GL_QUADS)
                    gl.glColor4f(cell_color[0], cell_color[1], cell_color[2], 1.0)
                    gl.glVertex3f(cell_x+       0,  cell_y+grid_size, 0)  # noqa
                    gl.glVertex3f(cell_x+grid_size, cell_y+grid_size, 0)  # noqa
                    gl.glVertex3f(cell_x+grid_size, cell_y+        0, 0)  # noqa
                    gl.glVertex3f(cell_x+        0, cell_y+        0, 0)  # noqa
                    gl.glEnd()
            w_offset += image_size + padding
        if save_to_file:
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                "/tmp/encodedenv3d_{:04d}.png".format(self.rendering_iteration))
        # actualize
        win.flip()
        self.rendering_iteration += 1
        return self.viewer.isopen
