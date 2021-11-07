import os
from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from navrep3d.custom_policy import NavRep3DTrainEnvDiscreteFlattened
from navrep3d.sb3_callbacks import NavRep3DLogCallback

if __name__ == "__main__":
    args, _ = parse_common_args()

    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" # noqa

    if "DISCRETE" in MODELPATH:
        env = SubprocVecEnv([
            lambda: NavRep3DTrainEnvDiscreteFlattened(build_name="./city.x86_64",
                                                      debug_export_every_n_episodes=170, port=25002),
            lambda: NavRep3DTrainEnvDiscreteFlattened(build_name="./city.x86_64",
                                                      debug_export_every_n_episodes=0, port=25003),
            lambda: NavRep3DTrainEnvDiscreteFlattened(build_name="./office.x86_64",
                                                      debug_export_every_n_episodes=0, port=25004),
            lambda: NavRep3DTrainEnvDiscreteFlattened(build_name="./office.x86_64",
                                                      debug_export_every_n_episodes=0, port=25005),
        ])
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
