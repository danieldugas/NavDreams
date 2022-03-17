import os
from navrep.tools.commonargs import parse_common_args
from datetime import datetime
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from navdreams.sb3_callbacks import NavRep3DLogCallback
from navdreams.navrep3dtrainenv import NavRep3DTrainEnv, check_running_unity_backends
from navdreams.custom_policy import NavRep3DTupleCNN, NavRep3DTrainEnvFlattened
from navdreams.auto_debug import enable_auto_debug

if __name__ == "__main__":
    enable_auto_debug()
    args, _ = parse_common_args()

    check_running_unity_backends()

    MILLION = 1000000
    TRAIN_STEPS = 60 * MILLION
    DIR = os.path.expanduser("~/navdreams_data/results/models/gym")
    LOGDIR = os.path.expanduser("~/navdreams_data/results/logs/gym")
    if args.dry_run:
        DIR = "/tmp/navdreams_data/results/models/gym"
        LOGDIR = "/tmp/navdreams_data/results/logs/gym"
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    CONTROLLER_ARCH = "_VCARCH_C64"
    LOGNAME = "navrep3dtrainenv_" + START_TIME + "_SAC" + "_E2E" + CONTROLLER_ARCH
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    if True:
        env = SubprocVecEnv([
            lambda: NavRep3DTrainEnvFlattened(verbose=0, debug_export_every_n_episodes=170, port=25002),
            lambda: NavRep3DTrainEnvFlattened(verbose=0, debug_export_every_n_episodes=0, port=25003),
            lambda: NavRep3DTrainEnvFlattened(verbose=0, debug_export_every_n_episodes=0, port=25004),
            lambda: NavRep3DTrainEnvFlattened(verbose=0, debug_export_every_n_episodes=0, port=25005),
        ])
    else:
        env = NavRep3DTrainEnvFlattened(verbose=0, debug_export_every_n_episodes=100)

    cb = NavRep3DLogCallback(logpath=LOGPATH, savepath=MODELPATH, verbose=1)

    policy_kwargs = dict(
        features_extractor_class=NavRep3DTupleCNN,
        features_extractor_kwargs=dict(cnn_features_dim=128),
    )
    model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, buffer_size=10000)
    model.learn(total_timesteps=TRAIN_STEPS, callback=cb)

    print("Saving model to {}".format(MODELPATH))
    model.save(MODELPATH)

    del model # remove to demonstrate saving and loading

    model = SAC.load(MODELPATH)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
        env.render()
