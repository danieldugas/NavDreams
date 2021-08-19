from gym import spaces, Env
import numpy as np
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from functools import partial

from navrep3d.navrep3dtrainenv import NavRep3DTrainEnv
from navrep3d.navrep3dtrainenv import NavRep3DTrainEnvDiscrete, convert_discrete_to_continuous_action
from navrep3d.encodedenv3d import EnvEncoder

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


class NavRep3DTrainEncodedEnv(Env):
    """ takes a (3) action as input
    outputs encoded obs (546) """
    def __init__(self, backend, encoding,
                 verbose=0, collect_statistics=True, debug_export_every_n_episodes=0, port=25001,
                 gpu=False, shared_encoder=None, encoder=None):
        if encoder is None:
            encoder = NavRep3DTrainEncoder(backend, encoding,
                                           gpu=gpu, encoder_to_share_model_with=shared_encoder)
        self.encoder = encoder
        self.env = NavRep3DTrainEnv(verbose=verbose, collect_statistics=collect_statistics,
                                    debug_export_every_n_episodes=debug_export_every_n_episodes,
                                    port=port)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space
        self.episode_statistics = self.env.episode_statistics

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

class SubprocVecNavRep3DEncodedEnv(SubprocVecEnv):
    """ A SubprocVecEnv with multiple NavRep3DTrainEncodedEnv will fail: each encoder gets copied
    and the GPU memory fills up.

    SubprocVecEnv:
      simulators S S S S (unity + CPU)
                 | | | |  - each in own process
      encoders   E E E E (GPU)

    This:
      simulators S S S S (unity + CPU)
                 | | | |  - each in own process
                 ------
                    |
      encoder       E (GPU)


    This wrapped SubprocVecEnv allows encoding the output of the multiprocessed environments sequentially,
    which makes multiprocessed navrep3d encoded environments usable
    """
    def __init__(self, backend, encoding, n_envs,
                 verbose=0, collect_statistics=True, debug_export_every_n_episodes=0,
                 gpu=False, ):
        # create multiple encoder objects (to store distinct sequences) but with single encoding model
        self.encoders = []
        shared_encoder = None
        for i in range(n_envs):
            self.encoders.append(
                NavRep3DTrainEncoder(backend, encoding, gpu=gpu,
                                     encoder_to_share_model_with=shared_encoder)
            )
            if i == 0:
                shared_encoder = self.encoders[i]
        # create multiprocessed simulators
        env_init_funcs = [
            partial(
                lambda i: NavRep3DTrainEnv(
                    verbose=verbose, collect_statistics=collect_statistics,
                    debug_export_every_n_episodes=debug_export_every_n_episodes if i == 0 else 0,
                    port=25002+i
                ),
                i=k
            )
            for k in range(n_envs)
        ]
        super(SubprocVecNavRep3DEncodedEnv, self).__init__(env_init_funcs)
        self.simulator_obs_space = self.observation_space
        self.encoder_obs_space = self.encoders[0].observation_space
        self.observation_space = self.encoder_obs_space

    def step_async(self, actions):
        super(SubprocVecNavRep3DEncodedEnv, self).step_async(actions)
        self.last_actions = actions

    def step_wait(self):
        # hack: vecenv expects the simulator obs space to be set.
        # RL algo expects obs space to be the encoded obs space -> we switch them around
        self.observation_space = self.simulator_obs_space
        obs, rews, dones, infos = super(SubprocVecNavRep3DEncodedEnv, self).step_wait()
        self.observation_space = self.encoder_obs_space
        h = [encoder._encode_obs((imob, rsob), a)
             for imob, rsob, a, encoder in zip(obs[0], obs[1], self.last_actions, self.encoders)]
        return np.stack(h), rews, dones, infos

    def reset(self):
        for encoder in self.encoders:
            encoder.reset()
        self.observation_space = self.simulator_obs_space
        obs = super(SubprocVecNavRep3DEncodedEnv, self).reset()
        self.observation_space = self.encoder_obs_space
        h = [encoder._encode_obs((imob, rsob), np.array([0,0,0]))
             for imob, rsob, encoder in zip(obs[0], obs[1], self.encoders)]
        return np.stack(h)

class SubprocVecNavRep3DEncodedEnvDiscrete(SubprocVecEnv):
    """ Same as SubprocVecNavRep3DEncodedEnv but using discrete actions.
    Could have been a wrapper instead, but fear of spaghetti-code outweighed DRY """
    def __init__(self, backend, encoding, n_envs,
                 verbose=0, collect_statistics=True, debug_export_every_n_episodes=0,
                 gpu=False, ):
        # create multiple encoder objects (to store distinct sequences) but with single encoding model
        self.encoders = []
        shared_encoder = None
        for i in range(n_envs):
            self.encoders.append(
                NavRep3DTrainEncoder(backend, encoding, gpu=gpu,
                                     encoder_to_share_model_with=shared_encoder)
            )
            if i == 0:
                shared_encoder = self.encoders[i]
        # create multiprocessed simulators
        env_init_funcs = [
            partial(
                lambda i: NavRep3DTrainEnvDiscrete(
                    verbose=verbose, collect_statistics=collect_statistics,
                    debug_export_every_n_episodes=debug_export_every_n_episodes if i == 0 else 0,
                    port=25002+i
                ),
                i=k
            )
            for k in range(n_envs)
        ]
        super(SubprocVecNavRep3DEncodedEnv, self).__init__(env_init_funcs)
        self.simulator_obs_space = self.observation_space
        self.encoder_obs_space = self.encoders[0].observation_space
        self.observation_space = self.encoder_obs_space

    def step_async(self, actions):
        super(SubprocVecNavRep3DEncodedEnv, self).step_async(actions)
        self.last_actions = actions

    def step_wait(self):
        # hack: vecenv expects the simulator obs space to be set.
        # RL algo expects obs space to be the encoded obs space -> we switch them around
        self.observation_space = self.simulator_obs_space
        obs, rews, dones, infos = super(SubprocVecNavRep3DEncodedEnv, self).step_wait()
        self.observation_space = self.encoder_obs_space
        h = [encoder._encode_obs((imob, rsob), convert_discrete_to_continuous_action(a))
             for imob, rsob, a, encoder in zip(obs[0], obs[1], self.last_actions, self.encoders)]
        return np.stack(h), rews, dones, infos

    def reset(self):
        for encoder in self.encoders:
            encoder.reset()
        self.observation_space = self.simulator_obs_space
        obs = super(SubprocVecNavRep3DEncodedEnv, self).reset()
        self.observation_space = self.encoder_obs_space
        h = [encoder._encode_obs((imob, rsob), np.array([0,0,0]))
             for imob, rsob, encoder in zip(obs[0], obs[1], self.encoders)]
        return np.stack(h)


if __name__ == "__main__":
    from navrep.tools.envplayer import EnvPlayer

    np.set_printoptions(precision=1, suppress=True)
    env = NavRep3DTrainEncodedEnv(verbose=1, backend="GPT", encoding="V_ONLY")
    player = EnvPlayer(env)
    player.run()
