import os
import numpy as np
from strictfire import StrictFire
from stable_baselines3 import PPO

from navrep3d.navrep3dtrainencodedenv import SubprocVecNavRep3DEncodedEnvDiscrete
from plot_gym_training_progress import get_variant

def main(build_name="staticasl", render=True):
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daltencodedenv_2021_10_08__16_34_19_DISCRETE_PPO_GPT_V_ONLY_V64M64_Salt_ckpt.zip" # noqa

    backend = "GPT"
    encoding = "V_ONLY"
    model = PPO.load(MODELPATH)
    variant = get_variant(os.path.basename(MODELPATH))
    N_ENVS = 4
    env = SubprocVecNavRep3DEncodedEnvDiscrete(backend, encoding, variant, N_ENVS,
                                               build_name=build_name,
                                               debug_export_every_n_episodes=0)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if np.any(dones):
            env.reset()
        if render:
            if build_name == "rosbag":
                env.render(save_to_file=True, action_override=action)
            else:
                env.render(save_to_file=True)


if __name__ == "__main__":
    StrictFire(main)
