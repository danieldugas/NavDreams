from tqdm import tqdm
from strictfire import StrictFire
from stable_baselines3 import PPO

from navrep3d.navrep3danyenv import NavRep3DAnyEnvDiscreteFlattened

def main(build_name="./alternate.x86_64", render=True, difficulty_mode="hardest"):
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" # noqa

    model = PPO.load(MODELPATH)
    env = NavRep3DAnyEnvDiscreteFlattened(verbose=0, build_name=build_name, debug_export_every_n_episodes=0,
                                          difficulty_mode=difficulty_mode)

    successes = []
    N = 1000
    pbar = tqdm(range(N))
    for i in pbar:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if render:
                if build_name == "rosbag":
                    env.render(save_to_file=True, action_override=action)
                else:
                    env.render(save_to_file=True)
            if done:
                if reward > 50.:
                    if render:
                        print("Success!")
                    successes.append(1.)
                else:
                    if render:
                        print("Failure.")
                    successes.append(0.)
                pbar.set_description(f"Success rate: {sum(successes)/len(successes):.2f}")
                break


if __name__ == "__main__":
    StrictFire(main)
