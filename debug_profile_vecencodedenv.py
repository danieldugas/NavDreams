import numpy as np
from fire import Fire
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import gym

from navrep3d.navrep3dtrainencodedenv import SubprocVecNavRep3DEncodedEnv, NavRep3DTrainEncodedEnv, NavRep3DTrainEncoder

def debug_env_max_speed(env, n=2000, render=False):
    from tqdm import tqdm
    import numpy as np
    env.reset()
    n_envs = env.num_envs # len(env.remotes)
    n_episodes = 0
    for i in tqdm(range(n)):
        env.step_async(0.1*np.random.uniform(size=(n_envs,3)))
        _,_,done,_ = env.step_wait()
        if i % 10 == 0 and render:
            env.render()
        if np.any(done):
            env.reset()
            n_episodes += 1
    env.close()

# separate main function to define the script-relevant arguments used by Fire
def main(
    # NavRep3DTrainEnv args
    verbose=1, collect_statistics=True, debug_export_every_n_episodes=0, port=25001,
    # Player args
    render_mode='human', step_by_step=False,
    # encoder args
    backend="GPT", encoding="V_ONLY", no_gpu=False
):
    N_ENVS = 4
    np.set_printoptions(precision=1, suppress=True)
    env = SubprocVecNavRep3DEncodedEnv(backend, encoding, N_ENVS)
    debug_env_max_speed(env)
    shared_encoder = NavRep3DTrainEncoder(backend, encoding, gpu=not no_gpu)
    env = DummyVecEnv([
        lambda: NavRep3DTrainEncodedEnv(backend, encoding, verbose=0,
                                        gpu=not no_gpu, shared_encoder=shared_encoder,
                                        debug_export_every_n_episodes=debug_export_every_n_episodes, port=25002),
        lambda: NavRep3DTrainEncodedEnv(backend, encoding, verbose=0,
                                        gpu=not no_gpu, shared_encoder=shared_encoder,
                                        debug_export_every_n_episodes=0, port=25003),
        lambda: NavRep3DTrainEncodedEnv(backend, encoding, verbose=0,
                                        gpu=not no_gpu, shared_encoder=shared_encoder,
                                        debug_export_every_n_episodes=0, port=25004),
        lambda: NavRep3DTrainEncodedEnv(backend, encoding, verbose=0,
                                        gpu=not no_gpu, shared_encoder=shared_encoder,
                                        debug_export_every_n_episodes=0, port=25005),
    ])
    debug_env_max_speed(env)


if __name__ == "__main__":
    Fire(main)
