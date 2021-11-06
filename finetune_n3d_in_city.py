import os
from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO

from navrep3d.navrep3dtrainencodedenv import SubprocVecNavRep3DEncodedEnvDiscrete
from navrep3d.sb3_callbacks import NavRep3DLogCallback

from plot_gym_training_progress import get_variant

if __name__ == "__main__":
    args, _ = parse_common_args()

    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daltencodedenv_2021_10_08__16_34_19_DISCRETE_PPO_GPT_V_ONLY_V64M64_Salt_ckpt.zip" # noqa

    variant = get_variant(os.path.basename(MODELPATH))
    if "DISCRETE" in MODELPATH:
        env = SubprocVecNavRep3DEncodedEnvDiscrete("GPT", "V_ONLY", variant, 4,
                                                   build_name="./city.x86_64",
                                                   debug_export_every_n_episodes=170)
    else:
        raise NotImplementedError
    model = PPO.load(MODELPATH, env=env)

    HOME = os.path.expanduser("~")
    TRAIN_STEPS = 100000
    FILENAME = os.path.splitext(os.path.basename(MODELPATH))[0]
    LOGDIR = os.path.join(HOME, "finetune/logs/gym")
    SAVEDIR = os.path.join(HOME, "finetune/models/gym")
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    LOGPATH = os.path.join(LOGDIR, FILENAME + ".csv")
    SAVEPATH = os.path.join(SAVEDIR, FILENAME + ".zip")
    cb = NavRep3DLogCallback(logpath=LOGPATH, savepath=SAVEPATH, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)
