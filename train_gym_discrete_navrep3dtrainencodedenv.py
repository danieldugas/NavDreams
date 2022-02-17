from datetime import datetime
import os
from strictfire import StrictFire

from stable_baselines3 import PPO

from navrep3d.navrep3dtrainencodedenv import (NavRep3DTrainEncoder,
                                              SubprocVecNavRep3DEncodedEnvDiscrete)
from navrep3d.sb3_callbacks import NavRep3DLogCallback
from navrep3d.auto_debug import enable_auto_debug

MILLION = 1000000

def main(backend="GPT", encoding="V_ONLY", variant="S", no_gpu=False, dry_run=False, n=None, build_name=None):
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
    if build_name is None:
        build_name = "./alternate.x86_64"
    build_names = build_name
    if build_name == "./build.x86_64":
        ENV_NAME = "navrep3dtrainencodedenv_"
    elif build_name == "./alternate.x86_64":
        ENV_NAME = "navrep3daltencodedenv_"
    elif build_name == "./city.x86_64":
        ENV_NAME = "navrep3dcityencodedenv_"
    elif build_name == "./office.x86_64":
        ENV_NAME = "navrep3dofficeencodedenv_"
    elif build_name == "SC":
        ENV_NAME = "navrep3dSCencodedenv_"
        build_names = ["./alternate.x86_64", "./city.x86_64", "./office.x86_64", "./office.x86_64"]
    elif build_name == "SCR":
        ENV_NAME = "navrep3dSCRfixedencodedenv_"
        build_names = ["./alternate.x86_64", "./city.x86_64", "./office.x86_64", "staticasl"]
    elif build_name == "staticasl":
        ENV_NAME = "navrep3daslfixedencodedenv_"
    elif build_name == "unity3":
        ENV_NAME = "navrep3dunity3encodedenv_"
        build_names = ["staticasl", "staticasl", "cathedral", "gallery"]
    elif build_name == "cathedral":
        ENV_NAME = "navrep3dcathedralencodedenv_"
    elif build_name == "gallery":
        ENV_NAME = "navrep3dgalleryencodedenv_"
    elif build_name == "kozehd":
        ENV_NAME = "navrep3dkozehdrsencodedenv_"
    else:
        raise NotImplementedError
    LOGNAME = ENV_NAME + START_TIME + "_DISCRETE_PPO" + ENCODER_ARCH
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    MODELPATH2 = os.path.join(DIR, ENV_NAME + "latest_DISCRETE_PPO_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    TRAIN_STEPS = n
    if TRAIN_STEPS is None:
        TRAIN_STEPS = 5 * MILLION

    if True:
        N_ENVS = 4
        env = SubprocVecNavRep3DEncodedEnvDiscrete(backend, encoding, variant, N_ENVS,
                                                   build_name=build_names,
                                                   debug_export_every_n_episodes=170)
#     else:
#         env = NavRep3DTrainEncodedEnv(backend, encoding,
#                                       verbose=0,
#                                       gpu=not no_gpu,
#                                       shared_encoder=shared_encoder)
    cb = NavRep3DLogCallback(logpath=LOGPATH, savepath=MODELPATH, verbose=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)

    if False:
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

    env.close()


if __name__ == "__main__":
    enable_auto_debug()
    StrictFire(main)
