import numpy as np
from strictfire import StrictFire
from stable_baselines3 import PPO

from navrep3d.navrep3danyenv import NavRep3DAnyEnvDiscreteFlattened

def main(build_name="staticasl", render=True):
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" # noqa

    model = PPO.load(MODELPATH)
    env = NavRep3DAnyEnvDiscreteFlattened(verbose=0, build_name=build_name, debug_export_every_n_episodes=0)

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
