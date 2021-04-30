from gym import spaces
import numpy as np
import os

from navrep3dtrainenv import NavRep3DTrainEnv
from encodedenv3d import EnvEncoder

class NavRep3DTrainEncoder(EnvEncoder):
    def __init__(self, backend, encoding,
                 gpu=False, encoder_to_share_model_with=None):
        super(NavRep3DTrainEncoder, self).__init__(
            backend, encoding,
            rnn_model_path=os.path.expanduser("~/navrep3d/models/M/navrep3dtrainrnn.json"),
            vae_model_path=os.path.expanduser("~/navrep3d/models/V/navrep3dtrainvae.json"),
            gpt_model_path=os.path.expanduser("~/navrep3d/models/W/navrep3dtraingpt"),
            vaelstm_model_path=os.path.expanduser("~/navrep3d/models/W/navrep3dtrainvaelstm"),
            gpu=gpu,
            encoder_to_share_model_with=None,
        )


class NavRep3DTrainEncodedEnv(object):
    """ takes a (3) action as input
    outputs encoded obs (546) """
    def __init__(self, backend, encoding,
                 verbose=0, collect_statistics=False,
                 gpu=False, shared_encoder=None, encoder=None):
        if encoder is None:
            encoder = NavRep3DTrainEncoder(backend, encoding,
                                           gpu=gpu, encoder_to_share_model_with=shared_encoder)
        self.encoder = encoder
        self.env = NavRep3DTrainEnv(verbose=verbose, collect_statistics=collect_statistics)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def _get_dt(self):
        return self.env._get_dt()
    
    def _get_viewer(self):
        return self.encoder.viewer

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self, *args, **kwargs):
        self.encoder.reset()
        obs = self.env.reset(*args, **kwargs)
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

    def close(self):
        self.env.close()
        self.encoder.close()

    def render(self, mode="human", close=False, save_to_file=False):
        self.encoder._render_side_by_side(mode=mode, close=close, save_to_file=save_to_file)


if __name__ == "__main__":
    from navrep.tools.envplayer import EnvPlayer

    np.set_printoptions(precision=1, suppress=True)
    env = NavRep3DTrainEncodedEnv(verbose=1, backend="GPT", encoding="V_ONLY")
    player = EnvPlayer(env)
    player.run()
