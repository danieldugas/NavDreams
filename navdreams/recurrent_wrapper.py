from gym import spaces
from gym.core import ObservationWrapper
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from functools import partial
from stable_baselines3.common.vec_env import VecEnv


from navdreams.navrep3dtrainencodedenv import NavRep3DTrainEncoder
from navdreams.navrep3dtrainenv import convert_discrete_to_continuous_action
from navdreams.navrep3danyenv import NavRep3DAnyEnvDiscrete

class Sequencify(object):
    """ given a single input, it returns a fixed-length sequence of the last n inputs """
    def __init__(self, n=10):
        self.n = n
        self.sequence = []

    def sequencify(self, obs):
        if len(self.sequence) == 0:
            self.sequence = [obs] * self.n
        self.sequence.append(obs)
        self.sequence.pop(0)
        return self.sequence

    def reset(self):
        self.sequence = []


class RecurrentObsWrapper(ObservationWrapper):
    """
    Wrapper which turns observations into a sequence of observations
    """

    def __init__(self, env, n=10, concatenate=True):
        super().__init__(env)
        self.n = n
        self.concatenate = concatenate
        self.sequencifier = Sequencify(n=n)
        self.observation_space = env.observation_space
        if concatenate:
            cat_dim = env.observation_space.shape[0] * n # TODO FIX
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(cat_dim,), dtype=np.float32)
        else:
            raise NotImplementedError

    def observation(self, obs):
        sequence = self.sequencifier.sequencify(obs)
        if self.concatenate:
            newobs = np.concatenate(sequence, axis=0)
            return newobs

    def reset(self, *args, **kwargs):
        self.sequencifier.reset()
        return super(RecurrentObsWrapper, self).reset(*args, **kwargs)

class SubprocVecNavRep3DEncodedSeqEnvDiscrete(SubprocVecEnv):
    """ Same as SubprocVecNavRep3DEncodedEnv but using discrete actions.
    Could have been a wrapper instead, but fear of spaghetti-code outweighed DRY """
    def __init__(self, backend, encoding, variant, n_envs, n=10,
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
        self.sequencifiers = [Sequencify(n=n) for _ in range(n_envs)]
        # create multiprocessed simulators
        env_init_funcs = [
            partial(
                lambda i: NavRep3DAnyEnvDiscrete(
                    verbose=verbose, collect_statistics=collect_statistics, build_name=build_names[i],
                    debug_export_every_n_episodes=debug_export_every_n_episodes if i == 0 else 0,
                    port=25002+i
                ),
                i=k
            )
            for k in range(n_envs)
        ]
        super(SubprocVecNavRep3DEncodedSeqEnvDiscrete, self).__init__(env_init_funcs)
        self.simulator_obs_space = self.observation_space
        self.encoder_obs_space = self.encoders[0].observation_space
        assert len(self.encoder_obs_space.shape) == 1
        self.encoder_obs_space.shape = (self.encoder_obs_space.shape[0] * n, )
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
        h = [np.concatenate(sf.sequencify(f), axis=0) for sf, f in zip(self.sequencifiers, h)]
        return np.stack(h), rews, dones, infos

    def reset(self):
        for encoder in self.encoders:
            encoder.reset()
        for sequencifier in self.sequencifiers:
            sequencifier.reset()
        self.observation_space = self.simulator_obs_space
        obs = super(SubprocVecNavRep3DEncodedSeqEnvDiscrete, self).reset()
        self.observation_space = self.encoder_obs_space
        h = [encoder._encode_obs((imob, rsob), np.array([0,0,0]))
             for imob, rsob, encoder in zip(obs[0], obs[1], self.encoders)]
        h = [np.concatenate(sf.sequencify(f), axis=0) for sf, f in zip(self.sequencifiers, h)]
        return np.stack(h)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    verbose = True
    collect_statistics = True
    backend = "GPT"
    encoding = "V_ONLY"
    variant = "SCR"
    N_ENVS = 4
    build_names = "staticasl"
    render_mode = "human"
    step_by_step = False
    env = SubprocVecNavRep3DEncodedSeqEnvDiscrete(backend, encoding, variant, N_ENVS,
                                                  build_name=build_names,
                                                  debug_export_every_n_episodes=0)
    env.reset()
    for i in range(10):
        obs, _, _, _ = env.step(np.array([env.action_space.sample() for _ in range(N_ENVS)]))
        plt.plot(obs[0] + i)
    print(obs.shape)
    env.close()
    plt.show()
