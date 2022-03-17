import os
from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from navdreams.custom_policy import NavRep3DTrainEnvDiscreteFlattened, NavRep3DTupleCNN
from navdreams.sb3_callbacks import NavRep3DLogCallback

if __name__ == "__main__":
    args, _ = parse_common_args()
    from_scratch = False

    MODELPATH = "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" # noqa
    if from_scratch:
        MODELPATH = "~/navdreams_data/results/models/gym/navrep3daltenv_from_scratch_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" # noqa
    MODELPATH = os.path.expanduser(MODELPATH)

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
    if from_scratch:
        _C = 64
        policy_kwargs = dict(
            features_extractor_class=NavRep3DTupleCNN,
            features_extractor_kwargs=dict(cnn_features_dim=_C),
        )
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
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
