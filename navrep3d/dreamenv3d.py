from __future__ import print_function
import numpy as np
import os
import gym

from navrep.tools.rings import generate_rings
from navrep.envs.ianenv import IANEnv
from navrep.models.rnn import (reset_graph, sample_hps_params, MDNRNN,
                               rnn_init_state, rnn_next_state, MAX_GOAL_DIST)
from navrep.models.vae2d import ConvVAE
from navrep.models.vae1d import Conv1DVAE
from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
from navrep.models.gpt1d import GPT1D
from navrep.models.vae1dlstm import VAE1DLSTM, VAE1DLSTMConfig
from navrep.models.vaelstm import VAELSTM, VAELSTMConfig
from navrep.tools.wdataset import scans_to_lidar_obs

from navrep3d.navrep3dtrainenv import NavRep3DTrainEnv

PUNISH_SPIN = True

""" VM backends: VAE_LSTM, W backends: GPT, GPT1D, VAE1DLSTM """
""" ENCODINGS: V_ONLY, VM, M_ONLY """
_G = 2  # goal dimensions
_A = 3  # action dimensions
_RS = 5  # robot state
_64 = 64  # image size
_C = 3 # channels in image
_H = 64 # 36 # 120 # image height
_W = 64 # 36 # 160 # image width
NO_VAE_VAR = True

BLOCK_SIZE = 32  # sequence length (context)

class DreamEnv(object):
    """ Generic class for generating dreams from trained world models """
    def __init__(self,
                 gpt_model_path=os.path.expanduser("~/navrep3d_W/models/W/transformer_SC"),
                 gpu=False,
                 alongside_sim=False,
                 ):
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=255, shape=(_H, _W, _C), dtype=np.uint8),
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        ))
        # params
        self.DT = 0.2
        self.alongside_sim = alongside_sim
        # load world model
        mconf = GPTConfig(BLOCK_SIZE, _H)
        mconf.image_channels = _C
        model = GPT(mconf, gpu=gpu)
        load_checkpoint(model, gpt_model_path, gpu=gpu)
        self.worldmodel = model
        # other tools
        self.viewer = None
        self.simenv = None
        # environment state variables
        self.reset()

    def _sample_zero_state(self):
        if self.alongside_sim:
            if self.simenv is None:
                self.simenv = NavRep3DTrainEnv()
            obs = self.simenv.reset()
            image_obs, robot_state = obs
            image_nobs = self._normalize_obs(image_obs)
            goal_state = robot_state[:2]
        else:
            from zero import zero_image_nobs, zero_goal_state
            image_nobs = zero_image_nobs * 1.
            goal_state = zero_goal_state * 1.
        return image_nobs, goal_state

    def reset(self):
        image_nobs, goal_state = self._sample_zero_state()
        self.gpt_sequence = [dict(obs=image_nobs, state=goal_state, action=None)]
        self.latest_image_nobs = image_nobs
        self.last_action = np.array([0,0,0])
        self.zero_state = (image_nobs, goal_state)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def _get_viewer(self):
        return self.viewer

    def _get_dt(self):
        return self.DT

    def _normalize_obs(self, obs):
        """
        transforms the raw env obs (observation_space - eg image 0-255)
        to the format required by model (eg. image 0-1.)
        """
        return obs / 255.

    def _unnormalize_obs(self, nobs):
        return (nobs * 255).astype(np.uint8)

    def step(self, action):
        done = False
        self.gpt_sequence[-1]['action'] = action * 1.
        img_npred, goal_pred = self.worldmodel.get_next(self.gpt_sequence)

        # update sequence
        self.gpt_sequence.append(dict(obs=img_npred, state=goal_pred, action=None))
        self.gpt_sequence = self.gpt_sequence[:BLOCK_SIZE]

        # store for rendering
        self.latest_image_nobs = img_npred
        self.last_action = action * 1.

        img_pred = self._unnormalize_obs(img_npred)
        obs = (img_pred, goal_pred)

        if self.alongside_sim and self.simenv is not None:
            _, _, done, _ = self.simenv.step(action)

        return obs, 0, done, {}

    def render(self, mode="human", close=False, save_to_file=False):
        """ renders true and encoded image side by side """
        if self.alongside_sim and self.simenv is not None:
            self.simenv.render()
        if close:
            self.viewer.close()
            return
        # rendering
        last_image = self.latest_image_nobs
        # Window and viewport size
        padding = 4  # grid cells
        grid_size = 1  # px per grid cell
        WINDOW_W = (2 * _W + 3 * padding) * grid_size
        WINDOW_H = (1 * _H + 2 * padding) * grid_size
        VP_W = WINDOW_W
        VP_H = WINDOW_H
        from gym.envs.classic_control import rendering
        import pyglet
        from pyglet import gl
        # Create viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.rendering_iteration = 0
            self.score_label = pyglet.text.Label(
                '0000', font_size=8,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
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
        for img in [last_image, last_image]:
            for i in range(_H):
                for j in range(_W):
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
            w_offset += _W + padding
        # Text
        self.score_label.text = "A {:.1f} {:.1f} {:.1f}".format(
            self.last_action[0],
            self.last_action[1],
            self.last_action[2],
        )
        self.score_label.draw()
        if save_to_file:
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                "/tmp/dreamenv3d_{:04d}.png".format(self.rendering_iteration))
        # actualize
        win.flip()
        self.rendering_iteration += 1
        return self.viewer.isopen


if __name__ == "__main__":
    from strictfire import StrictFire
    from navrep.tools.envplayer import EnvPlayer
    np.set_printoptions(precision=1, suppress=True)
#     env = StrictFire(DreamEnv)
    env = DreamEnv(alongside_sim=True)
    player = EnvPlayer(env)
    player.run()
