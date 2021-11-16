import gym
import numpy as np
import dreamerv2.api as dv2
from navrep3d.navrep3dtrainenv import NavRep3DTrainEnvDiscrete

config = dv2.defaults.update({
    'logdir': '~/logdir/dreamerv2',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper for compatibility with dreamer
    """

    def __init__(self, env):
        super().__init__(env)

        _H = 64
        _W = 64

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(_H, _W, 3), dtype='uint8'),
            'mission': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })
        self.obs_space = self.observation_space

    def observation(self, obs):
        return {
            'mission': obs[1],
            'image': obs[0]
        }


env = RGBImgPartialObsWrapper(NavRep3DTrainEnvDiscrete())
dv2.train(env, config)
