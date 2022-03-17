import os
from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO

from navdreams.navrep3dtrainencodedenv import SubprocVecNavRep3DEncodedEnvDiscrete
from navdreams.sb3_callbacks import NavRep3DLogCallback

from plot_gym_training_progress import get_variant

if __name__ == "__main__":
    args, _ = parse_common_args()
    from_scratch = False

#     MODELPATH = "~/navdreams_data/results/models/gym/navrep3dtrainencodedenv_2021_10_02__12_44_20_DISCRETE_PPO_GPT_V_ONLY_V64M64_Salt_ckpt.zip" # noqa
    MODELPATH = "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_10_25__15_01_28_DISCRETE_PPO_GPT_V_ONLY_V64M64_SC_ckpt.zip" # noqa
    if from_scratch:
        MODELPATH = "~/navdreams_data/results/models/gym/navrep3daltencodedenv_from_scratch_DISCRETE_PPO_GPT_V_ONLY_V64M64_Random_ckpt.zip" # noqa
    MODELPATH = os.path.expanduser(MODELPATH)

    variant = get_variant(os.path.basename(MODELPATH))
    if "DISCRETE" in MODELPATH:
        env = SubprocVecNavRep3DEncodedEnvDiscrete("GPT", "V_ONLY", variant, 4,
                                                   build_name=["./city.x86_64",
                                                               "./city.x86_64",
                                                               "./office.x86_64",
                                                               "./office.x86_64"],
                                                   debug_export_every_n_episodes=170)
    else:
        raise NotImplementedError
    if from_scratch:
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        model = PPO.load(MODELPATH, env=env)

    BASE = os.path.expanduser("~/navdreams_data/results")
    TRAIN_STEPS = 2000000
    FILENAME = os.path.splitext(os.path.basename(MODELPATH))[0].replace("_ckpt", "")
    LOGDIR = os.path.join(BASE, "logs/finetune")
    SAVEDIR = os.path.join(BASE, "models/finetune")
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    LOGPATH = os.path.join(LOGDIR, FILENAME + ".csv")
    SAVEPATH = os.path.join(SAVEDIR, FILENAME + "_ckpt.zip")
    cb = NavRep3DLogCallback(logpath=LOGPATH, savepath=SAVEPATH, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)
