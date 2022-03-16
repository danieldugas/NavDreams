import os
from strictfire import StrictFire
from stable_baselines3 import PPO
from tqdm import tqdm

from navdreams.navrep3dtrainencodedenv import EncoderObsWrapper
from navdreams.navrep3danyenv import NavRep3DAnyEnvDiscrete
from plot_gym_training_progress import get_variant

def main(build_name="./alternate.x86_64", difficulty_mode="random", model_path=None):
    render = True
    MODELPATH = "~/navrep3d/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    if model_path is not None:
        MODELPATH = model_path
    MODELPATH = os.path.expanduser(MODELPATH)

    backend = "GPT"
    encoding = "V_ONLY"
    model = PPO.load(MODELPATH)
    print("Loaded {}".format(MODELPATH))
    variant = get_variant(os.path.basename(MODELPATH))
    env = NavRep3DAnyEnvDiscrete(build_name=build_name,
                                 debug_export_every_n_episodes=0,
                                 difficulty_mode=difficulty_mode,
                                 render_trajectories=True)
    env = EncoderObsWrapper(env, backend=backend, encoding=encoding, variant=variant)
    env.reset()

    successes = []
    N = 1000
    pbar = tqdm(range(N))
    for i in pbar:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if render and i != 0:
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
