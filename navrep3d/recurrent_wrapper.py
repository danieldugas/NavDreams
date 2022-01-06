import numpy as np
from gym import spaces
from gym.core import ObservationWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from functools import partial

from navrep3d.navrep3dtrainenv import convert_discrete_to_continuous_action
from navrep3d.navrep3danyenv import NavRep3DAnyEnvDiscrete
from navrep3d.navrep3dtrainencodedenv import NavRep3DTrainEncoder

class RecurrentObsWrapper(ObservationWrapper):
    """
    Wrapper which turns observations into a sequence of observations
    """

    def __init__(self, env, n=10, concatenate=True):
        super().__init__(env)
        self.n = n
        self.obs_sequence = []
        self.observation_space = env.observation_space
        if concatenate:
            cat_dim = env.observation_space.shape[0] * n
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(cat_dim,), dtype=np.float32)
        else:
            raise NotImplementedError

    def observation(self, obs):
        if len(self.obs_sequence) == 0:
            self.obs_sequence = [obs] * self.n
        self.obs_sequence.append(obs)
        self.obs_sequence.pop(0)
        newobs = np.concatenate(self.obs_sequence, axis=0)
        return newobs

    def reset(self, *args, **kwargs):
        self.obs_sequence = []
        return super(RecurrentObsWrapper, self).reset(*args, **kwargs)

class SubprocVecNavRep3DEncodedSeqEnvDiscrete(SubprocVecEnv):
    """ Same as SubprocVecNavRep3DEncodedEnv but using discrete actions.
    Could have been a wrapper instead, but fear of spaghetti-code outweighed DRY """
    def __init__(self, backend, encoding, variant, n_envs,
                 verbose=0, collect_statistics=True, debug_export_every_n_episodes=0, build_name=None,
                 gpu=False, ):
        # create multiple encoder objects (to store distinct sequences) but with single encoding model
        build_names = build_name if isinstance(build_name, list) else [build_name] * n_envs
        self.encoders = []
        shared_encoder = None
        for i in range(n_envs):
            self.encoders.append(
                NavRep3DTrainEncoder(backend, encoding, variant, gpu=gpu,
                                     encoder_to_share_model_with=shared_encoder)
            )
            if i == 0:
                shared_encoder = self.encoders[i]
        # create multiprocessed simulators
        env_init_funcs = [
            partial(
                lambda i: RecurrentObsWrapper(NavRep3DAnyEnvDiscrete(
                    verbose=verbose, collect_statistics=collect_statistics, build_name=build_names[i],
                    debug_export_every_n_episodes=debug_export_every_n_episodes if i == 0 else 0,
                    port=25002+i
                )),
                i=k
            )
            for k in range(n_envs)
        ]
        super(SubprocVecNavRep3DEncodedSeqEnvDiscrete, self).__init__(env_init_funcs)
        self.simulator_obs_space = self.observation_space
        self.encoder_obs_space = self.encoders[0].observation_space
        self.observation_space = self.encoder_obs_space

    def step_async(self, actions):
        super(SubprocVecNavRep3DEncodedSeqEnvDiscrete, self).step_async(actions)
        self.last_actions = actions

    def step_wait(self):
        # hack: vecenv expects the simulator obs space to be set.
        # RL algo expects obs space to be the encoded obs space -> we switch them around
        self.observation_space = self.simulator_obs_space
        obs, rews, dones, infos = super(SubprocVecNavRep3DEncodedSeqEnvDiscrete, self).step_wait()
        self.observation_space = self.encoder_obs_space
        h = [encoder._encode_obs((imob, rsob), convert_discrete_to_continuous_action(a))
             for imob, rsob, a, encoder in zip(obs[0], obs[1], self.last_actions, self.encoders)]
        return np.stack(h), rews, dones, infos

    def reset(self):
        for encoder in self.encoders:
            encoder.reset()
        self.observation_space = self.simulator_obs_space
        obs = super(SubprocVecNavRep3DEncodedSeqEnvDiscrete, self).reset()
        self.observation_space = self.encoder_obs_space
        h = [encoder._encode_obs((imob, rsob), np.array([0,0,0]))
             for imob, rsob, encoder in zip(obs[0], obs[1], self.encoders)]
        return np.stack(h)
