from __future__ import print_function
import numpy as np
import os
import gym
from strictfire import StrictFire

from navrep.models.gpt import GPT, GPTConfig, load_checkpoint

from navdreams.navrep3dtrainenv import (NavRep3DTrainEnv, convert_continuous_to_discrete_action,
                                       convert_discrete_to_continuous_action)
from navdreams.rssm import RSSMWMConf, RSSMWorldModel
from navdreams.tssm import TSSMWMConf, TSSMWorldModel
from navdreams.transformerL import TransformerLWMConf, TransformerLWorldModel

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
                 wm_model_path=os.path.expanduser("~/navrep3d_W/models/W/transformer_SC"),
                 worldmodel_type="Transformer",
                 gpu=False,
                 alongside_sim=False,
                 discrete_worldmodel=False,
                 ):
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=255, shape=(_H, _W, _C), dtype=np.uint8),
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        ))
        # params
        self.DT = 0.2
        self.alongside_sim = alongside_sim
        self.discrete_worldmodel = discrete_worldmodel
        # load world model
        if worldmodel_type == "Transformer":
            mconf = GPTConfig(BLOCK_SIZE, _H)
            mconf.image_channels = _C
            model = GPT(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            self.worldmodel = model
        elif worldmodel_type == "RSSM":
            mconf = RSSMWMConf()
            mconf.image_channels = 3
            model = RSSMWorldModel(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            self.worldmodel = model
        elif worldmodel_type == "TSSM":
            mconf = TSSMWMConf()
            mconf.image_channels = 3
            model = TSSMWorldModel(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            self.worldmodel = model
        elif worldmodel_type == "TransformerL":
            mconf = TransformerLWMConf()
            mconf.image_channels = 3
            model = TransformerLWorldModel(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            self.worldmodel = model
        elif worldmodel_type == "dTransformerL":
            mconf = TransformerLWMConf()
            mconf.image_channels = 3
            mconf.n_action = 4
            model = TransformerLWorldModel(mconf, gpu=gpu)
            load_checkpoint(model, wm_model_path, gpu=gpu)
            self.worldmodel = model
        else:
            raise NotImplementedError

        # other tools
        self.viewer = None
        self.simenv = None
        # environment state variables
        self.reset()

    def _sample_zero_state(self):
        if self.alongside_sim:
            if self.simenv is None:
                self.simenv = NavRep3DTrainEnv(difficulty_mode="random")
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
        self.zero_state = (image_nobs, goal_state) # in case we want to store it to file later
        self.nondream_steps_to_go = 16

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
        continuous_action = action
        if self.discrete_worldmodel:
            discrete_action = convert_continuous_to_discrete_action(action)
            onehot_action = np.array([0, 0, 0, 0], dtype=np.uint8)
            onehot_action[discrete_action] = 1
            continuous_action = convert_discrete_to_continuous_action(discrete_action)
            action = onehot_action
        done = False
        self.gpt_sequence[-1]['action'] = action * 1.
        img_npred, goal_pred = self.worldmodel.get_next(self.gpt_sequence)

        if self.alongside_sim and self.simenv is not None:
            sim_obs, _, done, _ = self.simenv.step(continuous_action)
            if self.nondream_steps_to_go > 0:
                self.nondream_steps_to_go -= 1
                img, robotstate = sim_obs
                img_npred = img / 255.
                goal_pred = robotstate[:2]

        # update sequence
        self.gpt_sequence.append(dict(obs=img_npred, state=goal_pred, action=None))
        self.gpt_sequence = self.gpt_sequence[-BLOCK_SIZE:]

        # store for rendering
        self.latest_image_nobs = img_npred
        self.last_action = continuous_action * 1.

        img_pred = self._unnormalize_obs(img_npred)
        obs = (img_pred, goal_pred)

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

def main(wm_type="Transformer"):
    from navrep.tools.envplayer import EnvPlayer
    discrete = False
    if wm_type == "Transformer":
        wm_model_path = os.path.expanduser("~/navrep3d_W/models/W/transformer_SC")
    elif wm_type == "RSSM":
        wm_model_path = os.path.expanduser("~/navrep3d_W/models/W/RSSM_A1_SCR")
    elif wm_type == "TSSM":
        wm_model_path = os.path.expanduser("~/navrep3d_W/models/W/TSSM_V2_SCR")
    elif wm_type == "TransformerL":
        wm_model_path = os.path.expanduser("~/navrep3d_W/models/W/TransformerL_V0_SCR")
    elif wm_type == "dTransformerL":
        wm_model_path = os.path.expanduser("~/navrep3d_W/models/W/TransformerL_V0_dSalt")
        discrete = True
    else:
        raise NotImplementedError
    env = DreamEnv(alongside_sim=True, wm_model_path=wm_model_path, worldmodel_type=wm_type,
                   discrete_worldmodel=discrete)
    player = EnvPlayer(env)
    player.run()


if __name__ == "__main__":
    np.set_printoptions(precision=1, suppress=True)
    StrictFire(main)
