import os
from strictfire import StrictFire
from stable_baselines3 import PPO
from tqdm import tqdm

from navrep3d.navrep3dtrainencodedenv import EncoderObsWrapper
from navrep3d.navrep3danyenv import NavRep3DAnyEnvDiscrete
from plot_gym_training_progress import get_variant

def main(build_name="kozehd", render=True, difficulty_mode="easiest", model_path=None):
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daslencodedenv_2021_12_11__00_23_55_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daslencodedenv_2021_12_08__10_18_09_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    MODELPATH = "~/navrep3d/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa # kozehd - easiest: 10%
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdencodedenv_2022_01_17__12_55_53_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa # kozehd - easiest: 30%
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdencodedenv_2022_01_17__12_55_53_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" # noqa # kozehd - easiest: 50%
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa # kozehd - easiest: 50%
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" # noqa # kozehd - easiest: 65%
#     MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdremptyencodedenv_2022_01_30__22_12_30_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_bestckpt.zip" # noqa # kozehd - easiest: 51%
#     MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdremptyencodedenv_2022_01_30__22_12_30_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_ckpt.zip" # noqa # kozehd - easiest: 50%
    # untested
#     MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdrencodedenv_2022_01_28__06_44_50_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_ckpt.zip" # noqa # kozehd - easiest: ?
#     MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdrencodedenv_2022_01_25__21_34_44_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_ckpt.zip" # noqa # kozehd - easiest: ?
#     MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdrencodedenv_2022_01_26__08_56_54_DISCRETE_PPO_GPT_V_ONLY_V64M64_K_ckpt.zip" # noqa # kozehd - easiest: ?
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
                                 difficulty_mode=difficulty_mode)
    env = EncoderObsWrapper(env, backend=backend, encoding=encoding, variant=variant)

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
