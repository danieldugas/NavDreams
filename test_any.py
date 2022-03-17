import numpy as np
import os
from strictfire import StrictFire
from stable_baselines3 import PPO
from tqdm import tqdm
from pyniel.python_tools.path_tools import make_dir_if_not_exists

from navdreams.navrep3dtrainencodedenv import EncoderObsWrapper
from navdreams.navrep3danyenv import NavRep3DAnyEnvDiscrete, NavRep3DAnyEnvDiscreteFlattened
from plot_gym_training_progress import get_variant

def main(build_name="kozehd", render=False, difficulty_mode="easiest", model_path=None, n_episodes=1000):
    if n_episodes is True:
        raise ValueError("n_episodes not specified. Must be int")
    # e2e
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" # noqa
    # n3d
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3daslencodedenv_2021_12_11__00_23_55_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3daslencodedenv_2021_12_08__10_18_09_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa # kozehd - easiest: 10% # cathedral - easiest 93
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdencodedenv_2022_01_17__12_55_53_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa # kozehd - easiest: 30%
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdencodedenv_2022_01_17__12_55_53_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" # noqa # kozehd - easiest: 50%
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa # kozehd - easiest: 50%
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" # noqa # kozehd - easiest: 65%
#     MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdremptyencodedenv_2022_01_30__22_12_30_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_bestckpt.zip" # noqa # kozehd - easiest: 51%
#     MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdremptyencodedenv_2022_01_30__22_12_30_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_ckpt.zip" # noqa # kozehd - easiest: 50%
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3daslfixedencodedenv_2021_12_29__17_17_16_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    # untested
#     MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdrencodedenv_2022_01_28__06_44_50_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_ckpt.zip" # noqa # kozehd - easiest: ?
#     MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdrencodedenv_2022_01_25__21_34_44_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_ckpt.zip" # noqa # kozehd - easiest: ?
#     MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdrencodedenv_2022_01_26__08_56_54_DISCRETE_PPO_GPT_V_ONLY_V64M64_K_ckpt.zip" # noqa # kozehd - easiest: ?
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_02__17_18_59_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" # noqa # kozehdrs - easiest: >80%
    if model_path is not None:
        MODELPATH = model_path
    MODELPATH = os.path.expanduser(MODELPATH)

    if "E2E" in MODELPATH:
        model = PPO.load(MODELPATH)
        env = NavRep3DAnyEnvDiscreteFlattened(verbose=0, build_name=build_name,
                                              debug_export_every_n_episodes=0,
                                              difficulty_mode=difficulty_mode)
    else:
        backend = "GPT"
        encoding = "V_ONLY"
        assert encoding in MODELPATH
        model = PPO.load(MODELPATH)
        print("Loaded {}".format(MODELPATH))
        variant = get_variant(os.path.basename(MODELPATH))
        env = NavRep3DAnyEnvDiscrete(build_name=build_name,
                                     debug_export_every_n_episodes=0,
                                     difficulty_mode=difficulty_mode)
        env = EncoderObsWrapper(env, backend=backend, encoding=encoding, variant=variant)

    successes = []
    difficulties = []
    lengths = []
    causes = []
    pbar = tqdm(range(n_episodes))
    for i in pbar:
        obs = env.reset()
        steps = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
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
                difficulty = info["episode_scenario"]
                difficulties.append(difficulty)
                lengths.append(steps)
                causes.append(str(info["event"]))
                pbar.set_description("Success rate: {:.2f}, avg dif: {:.2f}".format(
                    sum(successes)/len(successes), np.mean(difficulties)))
                break

    bname = build_name.replace(".x86_64", "").replace("./", "")
    SAVEPATH = MODELPATH.replace("models/gym", "test").replace(".zip", "") + "_{}_{}_{}.npz".format(
        bname, difficulty_mode, n_episodes)
    make_dir_if_not_exists(os.path.dirname(SAVEPATH))
    np.savez(SAVEPATH,
             successes=np.array(successes),
             difficulties=np.array(difficulties),
             lengths=np.array(lengths),
             causes=np.array(causes))
    print("Saved to {}".format(SAVEPATH))

    env.close()


if __name__ == "__main__":
    StrictFire(main)
