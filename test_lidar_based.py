from gym.core import ObservationWrapper, ActionWrapper
from gym import spaces
import numpy as np
from navrep.envs.navreptrainencodedenv import NavRepTrainEncoder
from navrep.scripts.test_navrep import NavRepCPolicy, run_test_episodes
from strictfire import StrictFire

from navdreams.navrep3dtrainenv import NavRep3DTrainEnv

class NavRep3DTrainEnvLidarBased(NavRep3DTrainEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_lidar = True
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=(1080,), dtype=np.float32),
            spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        ))

class NavRepEncodedEnvWrapper(ObservationWrapper):
    """ takes a (2) action as input
    outputs encoded obs (546) """
    def __init__(self, env, backend="GPT", encoding="V_ONLY",
                 gpu=False, shared_encoder=None, encoder=None):
        super().__init__(env)
        if encoder is None:
            encoder = NavRepTrainEncoder(backend, encoding,
                                         gpu=gpu, encoder_to_share_model_with=shared_encoder)
        self.encoder = encoder
        self.observation_space = self.encoder.observation_space

    def observation(self, obs):
        action = self.unwrapped.last_action
        h = self.encoder._encode_obs(obs, action)
        return h

    def reset(self, *args, **kwargs):
        self.encoder.reset()
        return super(NavRepEncodedEnvWrapper, self).reset(*args, **kwargs)

    def close(self):
        self.encoder.close()
        return super(NavRepEncodedEnvWrapper, self).close()

class NavRepNoRotActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(NavRepNoRotActionWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def action(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        return action

    def reverse_action(self, action):
        action = np.array([action[0], action[1]])  # no rotation
        return action

def get_dt(*args, **kwargs):
    return 0.2

def main(render=False):
    env = NavRep3DTrainEnvLidarBased()
    env = NavRepEncodedEnvWrapper(env)
    env = NavRepNoRotActionWrapper(env)
    env._get_dt = get_dt
    policy = NavRepCPolicy()
    run_test_episodes(env, policy, render=render)


if __name__ == "__main__":
    StrictFire(main)
