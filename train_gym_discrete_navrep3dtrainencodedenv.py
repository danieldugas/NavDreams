from datetime import datetime
import os
from fire import Fire

from stable_baselines3 import PPO

from navrep3d.navrep3dtrainencodedenv import (NavRep3DTrainEncoder,
                                              SubprocVecNavRep3DEncodedEnvDiscrete)
from navrep3d.sb3_callbacks import NavRep3DLogCallback
from navrep3d.auto_debug import enable_auto_debug

MILLION = 1000000

def main(backend="GPT", encoding="V_ONLY", variant="S", no_gpu=False, dry_run=False, n=60*MILLION):
    shared_encoder = NavRep3DTrainEncoder(backend, encoding, variant, gpu=not no_gpu)
    _Z = shared_encoder._Z
    _H = shared_encoder._H

    DIR = os.path.expanduser("~/navrep3d/models/gym")
    LOGDIR = os.path.expanduser("~/navrep3d/logs/gym")
    if dry_run:
        DIR = "/tmp/navrep3d/models/gym"
        LOGDIR = "/tmp/navrep3d/logs/gym"
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    ENCODER_ARCH = "_{}_{}_V{}M{}_{}".format(backend, encoding, _Z, _H, variant)
    LOGNAME = "navrep3dtrainencodedenv_" + START_TIME + "_DISCRETE_PPO" + ENCODER_ARCH
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    MODELPATH2 = os.path.join(DIR, "navrep3dtrainencodedenv_latest_DISCRETE_PPO_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    TRAIN_STEPS = n
    if TRAIN_STEPS is None:
        TRAIN_STEPS = 60 * MILLION

    if True:
        N_ENVS = 4
        env = SubprocVecNavRep3DEncodedEnvDiscrete(backend, encoding, variant, N_ENVS,
                                                   debug_export_every_n_episodes=170)
#     else:
#         env = NavRep3DTrainEncodedEnv(backend, encoding,
#                                       verbose=0,
#                                       gpu=not no_gpu,
#                                       shared_encoder=shared_encoder)
    cb = NavRep3DLogCallback(logpath=LOGPATH, savepath=MODELPATH, verbose=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)
    obs = env.reset()

    model.save(MODELPATH)
    model.save(MODELPATH2)
    print("Model '{}' saved".format(MODELPATH))

    del model
    env.close()

    model = PPO.load(MODELPATH)

    obs = env.reset()
    for i in range(512):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            env.reset()
#         env.render()

    print("exiting.")
    exit()


if __name__ == "__main__":
    enable_auto_debug()
    Fire(main)
